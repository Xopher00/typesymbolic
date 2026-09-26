"""Append-only decision/outcome/verdict/calibration journal. Rows are ids,
answers, and verdicts, never a raw domain payload, except `record_snapshot`
and the optional `state=`/`extra=` capture fields, which are replayed but
never indexed.

Writes enqueue to a background thread by default (`background_writes=False`
writes+indexes synchronously). The same path maintains a live `LabelIndex`
per calibration unit, so `calibrate.recalibrate()` needs no journal replay.
A verdict may arrive later, from another process; `refresh()` re-scans disk
for it. Rotation splits files by day; every read covers all of them.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .blobstore import BlobStore, offload, resolve
from .errors import TypesymbolicError
from .labels import DECISION, VERDICT, LabelIndex
from .question import Answer

if TYPE_CHECKING:
    from .domain import ActOutcome, Verdict
    from .gate import GateResult
    from .question import QuestionRef

OUTCOME = "outcome"
CALIBRATION = "calibration"
SNAPSHOT = "snapshot"

Rotation = Literal["none", "daily"]
DEFAULT_BLOB_MIN_BYTES = 2048

_OUTCOME_CORE_KEYS = frozenset({
    "type", "ts", "call_id", "gate_verdict", "gate_reason", "act_succeeded", "act_reasons",
    "key", "episode_id", "steps",
})


class JournalError(TypesymbolicError, ValueError):
    """An `extra` field collides with a core row column."""


class Journal:
    """Append-only JSONL under `root`. Fail-open on write."""

    def __init__(
        self, root: Path | str, *, enabled: bool = True, background_writes: bool = True,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC), fsync: bool = False,
        rotation: Rotation = "none", blob_min_bytes: int = DEFAULT_BLOB_MIN_BYTES,
    ) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.background_writes = background_writes
        self._clock = clock
        self._fsync = fsync
        self.rotation = rotation
        self._blob_min_bytes = blob_min_bytes
        self._blobs = BlobStore(self.root / "blobs")
        self._queue: queue.Queue[dict] = queue.Queue()
        self._labels = LabelIndex()
        self._index_lock = threading.Lock()
        self._writer: threading.Thread | None = None
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self._rebuild_index()
            if self.background_writes:
                self._writer = threading.Thread(target=self._run_writer, daemon=True)
                self._writer.start()

    def _path(self) -> Path:
        """The file new writes go to."""
        if self.rotation == "daily":
            return self.root / f"journal-{self._clock().strftime('%Y%m%d')}.jsonl"
        return self.root / "journal.jsonl"

    def _paths(self) -> list[Path]:
        """Every journal file on disk, oldest first."""
        if self.rotation == "daily":
            return sorted(self.root.glob("journal-*.jsonl"))
        path = self.root / "journal.jsonl"
        return [path] if path.exists() else []

    def _rebuild_index(self) -> None:
        """Reads files directly, not via `replay()`: the writer thread
        isn't running yet at construction time."""
        for path in self._paths():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        self._index_row(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def refresh(self) -> None:
        """Re-index from disk to pick up rows another process wrote.
        Idempotent; safe to call any time."""
        self._rebuild_index()

    def _run_writer(self) -> None:
        while True:
            row = self._queue.get()
            self._write_row(row)
            self._index_row(row)
            self._queue.task_done()

    def _write_row(self, row: dict) -> None:
        line = json.dumps(row, separators=(",", ":"), default=str)
        try:
            with open(self._path(), "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                if self._fsync:
                    handle.flush()
                    os.fsync(handle.fileno())
        except OSError as exc:
            print(f"journal write failed: {exc}")

    def _index_row(self, row: dict) -> None:
        with self._index_lock:
            self._labels.add_row(row)

    def _offload(self, value: Any) -> Any:
        return offload(value, store=self._blobs, min_bytes=self._blob_min_bytes)

    def _append(self, row_type: str, fields: dict) -> None:
        if not self.enabled:
            return
        row = {"type": row_type, "ts": self._clock().isoformat(timespec="milliseconds"), **fields}
        if self.background_writes:
            self._queue.put(row)
        else:
            self._write_row(row)
            self._index_row(row)

    def flush(self) -> None:
        """Block until every enqueued row is written and indexed. No-op
        when `background_writes=False`. Call via `asyncio.to_thread` from
        async code."""
        if self.enabled and self.background_writes:
            self._queue.join()

    def labeled_pairs(
        self, group: str, scale: str, *, engine: str,
        model_revision: str | None = None, any_revision: bool = False,
    ) -> list[tuple[float, bool]]:
        """(value, verified) pairs from the live index. `flush()` first for
        a complete view; `refresh()` first to see another process's writes.
        `any_revision=True` pools every model revision (jevdevice's opt-in
        `pool_revisions` calibration)."""
        with self._index_lock:
            return self._labels.pairs(
                group, scale, engine=engine, model_revision=model_revision, any_revision=any_revision,
            )

    def record_decision(
        self, *, call_id: str, engine: str, phase: str | None, answers: dict[str, Answer], model_revision: str | None = None,
        questions: dict[str, QuestionRef] | None = None, scope: dict | None = None,
        state: dict | None = None, extra: dict | None = None,
        error: str | None = None, elapsed_ms: float | None = None, asked: dict | None = None,
    ) -> None:
        """An answer key need not equal its `QuestionRef.qid` (e.g.
        `f"{qid}::{subject}"`, double colon -- a single `:` collides with
        `circuit.py`'s `"answer_id:option"` syntax). `state`/`extra`/`asked`
        are capture fields, offloaded to a blob when large, never read by
        the calibration index. `error` and empty `answers` together journal
        a failed ask."""
        fields = {
            "schema": 2, "call_id": call_id, "engine": engine, "model_revision": model_revision, "phase": phase,
            "scope": scope or {},
            "questions": {key: ref.model_dump(mode="json") for key, ref in (questions or {}).items()},
            "answers": {key: answer.model_dump(mode="json") for key, answer in answers.items()},
        }
        if state is not None:
            fields["state"] = self._offload(state)
        if extra is not None:
            fields["extra"] = self._offload(extra)
        if error is not None:
            fields["error"] = error
        if elapsed_ms is not None:
            fields["elapsed_ms"] = elapsed_ms
        if asked is not None:
            fields["asked"] = self._offload(asked)
        self._append(DECISION, fields)

    def record_outcome(
        self, *, call_id: str, gate: GateResult | None, outcome: ActOutcome | None = None,
        key: str | None = None, episode_id: str | None = None, extra: dict | None = None,
    ) -> None:
        """Gate and Act only -- Verify is `record_verdict()`. `gate=None`
        is a code-policy act with no gated judgment behind it. `extra`, if
        given, is merged flat into the row, each value offloaded on its own;
        a key colliding with a core column raises `JournalError`."""
        fields = {
            "call_id": call_id, "gate_verdict": gate.verdict if gate is not None else None,
            "gate_reason": gate.reason if gate is not None else None,
            "act_succeeded": outcome.succeeded if outcome is not None else None,
            "act_reasons": list(outcome.reasons) if outcome is not None else None,
            "key": key if key is not None else (outcome.key if outcome is not None else None),
            "episode_id": episode_id,
            "steps": [step.model_dump(mode="json") for step in outcome.steps] if outcome is not None else [],
        }
        for extra_key, value in (extra or {}).items():
            if extra_key in _OUTCOME_CORE_KEYS:
                raise JournalError(f"record_outcome: extra key {extra_key!r} collides with a core row column")
            fields[extra_key] = self._offload(value)
        self._append(OUTCOME, fields)

    def record_verdict(self, *, call_id: str, verdict: Verdict, tests: tuple[str, ...] | None = None) -> None:
        """A Verify result, callable any time, from any process. `tests`
        overrides `verdict.tests` when given."""
        final_tests = tests if tests is not None else (verdict.tests or ())
        observed_at = verdict.observed_at
        self._append(VERDICT, {
            "call_id": call_id, "tests": list(final_tests), "status": verdict.status,
            "reasons": list(verdict.reasons), "dedupe_key": verdict.dedupe_key,
            "observed_at": observed_at.isoformat(timespec="milliseconds") if isinstance(observed_at, datetime) else observed_at,
            "calibrate": verdict.calibrate,
        })

    def record_calibration(
        self, *, group: str, scale: str, engine: str, model_revision: str | None, current: float, proposed: float,
        applied: bool, n: int, precision: float | None, human_signoff: bool, note: str,
    ) -> None:
        """No `call_id`: never reaches `LabelIndex`."""
        self._append(CALIBRATION, {
            "group": group, "scale": scale, "engine": engine, "model_revision": model_revision,
            "current": current, "proposed": proposed, "applied": applied,
            "n": n, "precision": precision, "human_signoff": human_signoff, "note": note,
        })

    def record_snapshot(self, *, run_id: str, tags: dict | None = None, payload: Any) -> None:
        """Opaque per-run payload, replayed verbatim, never indexed."""
        self._append(SNAPSHOT, {"run_id": run_id, "tags": tags or {}, "payload": self._offload(payload)})

    def outcomes(self, *, call_id: str | None = None, episode_id: str | None = None) -> list[dict]:
        """Outcome rows, optionally filtered by `call_id`/`episode_id`.
        Flushes first, like `replay()`."""
        return [
            row for row in self.replay()
            if row.get("type") == OUTCOME
            and (call_id is None or row.get("call_id") == call_id)
            and (episode_id is None or row.get("episode_id") == episode_id)
        ]

    def replay(self):
        """Every row in write order, blob refs resolved. Flushes first.
        Torn lines are skipped, not fatal."""
        self.flush()
        for path in self._paths():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield resolve(row, store=self._blobs)
