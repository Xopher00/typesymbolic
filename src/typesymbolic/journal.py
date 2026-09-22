"""Append-only decision/outcome/calibration journal, domain-blind by
construction — decision/outcome/calibration rows are ids, answers, and
verdicts, never a domain payload (`record_outcome` reads
`GateResult`/`ActOutcome`/`Verdict` directly, so `ActOutcome.detail`
structurally never reaches a row); `record_snapshot()` is the deliberate
exception, see below. Storage root is caller-configured, not a hardcoded
dotfile, since one install may back more than one domain package.

Writes go through a background thread over a queue, not inline: a
`record_*` call only enqueues (no I/O on the caller's path), so a decision
loop judging many things a second never waits on a disk write to move on to
gating and acting. The same thread builds a live per-`(qid, engine,
model_revision)` index of (confidence, verified) pairs as it writes —
`labeled_pairs()` reads that index directly, no disk scan, so
`calibrate.recalibrate()` can run inline in the decision path without
costing a journal replay every time. `flush()` blocks until the queue
drains, for the one place that needs a guaranteed-complete view rather than
"complete as of the last processed row."

Concurrent writers are safe by row size, not by locking: one `open(..., "a")`
+ one short `write()` is atomic under O_APPEND because a row never carries a
domain payload. No rotation or pruning — never needed in practice.

`record_snapshot()` is the one row type this rule doesn't apply to: it holds
a caller-opaque payload, written and replayed verbatim, never parsed or
indexed — for a domain whose durable record is report-shaped (a whole run's
output) rather than a per-decision (confidence, verified) pair.

The live index is rebuilt from whatever's already on disk at construction,
synchronously, before the writer thread starts — a fresh-process-per-run
caller (a CLI, not a long-lived daemon) would otherwise never see labels
from its own prior runs.
"""

from __future__ import annotations

import json
import queue
import threading
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .domain import ActOutcome, Verdict
    from .gate import GateResult
    from .question import Answer

DECISION = "decision"
OUTCOME = "outcome"
CALIBRATION = "calibration"
SNAPSHOT = "snapshot"

IndexKey = tuple[str, str, "str | None"]  # (qid, engine, model_revision)


class Journal:
    """Append-only JSONL under `root`. Fail-open: a write failure must never
    break a live decision loop."""

    def __init__(self, root: Path | str, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self._queue: queue.Queue[dict] = queue.Queue()
        self._index: dict[IndexKey, list[tuple[float, bool]]] = defaultdict(list)
        self._pending: dict[tuple[str, str], tuple[float, str, str | None]] = {}  # (call_id, qid) -> (confidence, engine, model_revision)
        self._index_lock = threading.Lock()
        self._writer: threading.Thread | None = None
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self._rebuild_index()
            self._writer = threading.Thread(target=self._run_writer, daemon=True)
            self._writer.start()

    def _path(self) -> Path:
        return self.root / "journal.jsonl"

    def _rebuild_index(self) -> None:
        """Index whatever's already on disk, synchronously — reads the file
        directly rather than through `replay()`, since `replay()` flushes
        the queue and the writer thread isn't running yet at this point."""
        path = self._path()
        if not path.exists():
            return
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    self._index_row(json.loads(line))
                except json.JSONDecodeError:
                    continue

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
        except OSError as exc:
            print(f"journal write failed: {exc}")

    def _index_row(self, row: dict) -> None:
        with self._index_lock:
            if row["type"] == DECISION:
                call_id, engine, model_revision = row.get("call_id"), row.get("engine"), row.get("model_revision")
                if not call_id:
                    return
                for qid, answer in (row.get("answers") or {}).items():
                    confidence = answer.get("confidence")
                    if confidence is not None:
                        self._pending[(call_id, qid)] = (confidence, engine, model_revision)
            elif row["type"] == OUTCOME:
                call_id = row.get("call_id")
                if not call_id:
                    return
                verify_status = row.get("verify_status")
                verified = verify_status == "verified" if verify_status in ("verified", "failed") else None
                for key in [k for k in self._pending if k[0] == call_id]:
                    confidence, engine, model_revision = self._pending.pop(key)
                    if verified is not None:
                        self._index[(key[1], engine, model_revision)].append((confidence, verified))

    def _append(self, row_type: str, fields: dict) -> None:
        if not self.enabled:
            return
        row = {"type": row_type, "ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **fields}
        self._queue.put(row)

    def flush(self) -> None:
        """Block until every row enqueued so far has been written and
        indexed. Blocking: call via `asyncio.to_thread` from async code."""
        if self.enabled:
            self._queue.join()

    def labeled_pairs(self, qid: str, *, engine: str, model_revision: str | None = None) -> list[tuple[float, bool]]:
        """(confidence, verified) pairs for `qid`, from the live index —
        reflects every row the writer thread has processed so far; call
        `flush()` first for a guaranteed-complete view."""
        with self._index_lock:
            return list(self._index.get((qid, engine, model_revision), ()))

    def record_decision(
        self, *, call_id: str, engine: str, phase: str | None, answers: dict[str, Answer], model_revision: str | None = None,
    ) -> None:
        self._append(DECISION, {
            "call_id": call_id, "engine": engine, "model_revision": model_revision, "phase": phase,
            "answers": {qid: answer.model_dump(mode="json") for qid, answer in answers.items()},
        })

    def record_outcome(
        self, *, call_id: str, gate: GateResult, outcome: ActOutcome | None = None, verdict: Verdict | None = None,
    ) -> None:
        self._append(OUTCOME, {
            "call_id": call_id, "gate_verdict": gate.verdict, "gate_reason": gate.reason,
            "act_succeeded": outcome.succeeded if outcome is not None else None,
            "act_reasons": list(outcome.reasons) if outcome is not None else None,
            "verify_status": verdict.status if verdict is not None else None,
            "verify_reasons": list(verdict.reasons) if verdict is not None else None,
        })

    def record_calibration(
        self, *, qid: str, engine: str, model_revision: str | None, current: float, proposed: float,
        applied: bool, n: int, precision: float | None, human_signoff: bool, note: str,
    ) -> None:
        """No `call_id`: never enters `_pending`/`_index`, so it can't pool
        into a fit."""
        self._append(CALIBRATION, {
            "qid": qid, "engine": engine, "model_revision": model_revision,
            "current": current, "proposed": proposed, "applied": applied,
            "n": n, "precision": precision, "human_signoff": human_signoff, "note": note,
        })

    def record_snapshot(self, *, run_id: str, tags: dict | None = None, payload: Any) -> None:
        """Opaque per-run payload — written and replayed verbatim, never
        parsed or indexed. `payload` is domain-shaped by design; unlike
        every other row type here, it may carry raw domain data."""
        self._append(SNAPSHOT, {"run_id": run_id, "tags": tags or {}, "payload": payload})

    def replay(self):
        """Every row in write order, from disk — flushes first, so this
        always sees everything enqueued before the call, not just whatever
        the writer thread has gotten to. Torn/corrupt lines are skipped,
        not fatal."""
        self.flush()
        path = self._path()
        if not path.exists():
            return
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
