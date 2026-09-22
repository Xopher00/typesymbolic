"""Append-only decision/outcome/calibration journal, domain-blind by
construction — rows are ids, answers, and verdicts, never a domain payload
(`record_outcome` reads `GateResult`/`ActOutcome`/`Verdict` directly, so
`ActOutcome.detail` structurally never reaches a row). Storage root is
caller-configured, not a hardcoded dotfile, since one install may back more
than one domain package.

Concurrent writers are safe by row size, not by locking: one `open(..., "a")`
+ one short `write()` is atomic under O_APPEND because a row never carries a
domain payload. No rotation or pruning — never needed in practice.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .domain import ActOutcome, Verdict
    from .gate import GateResult
    from .question import Answer

DECISION = "decision"
OUTCOME = "outcome"
CALIBRATION = "calibration"


class Journal:
    """Append-only JSONL under `root`. Fail-open: a write failure must never
    break a live decision loop."""

    def __init__(self, root: Path | str, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.root / "journal.jsonl"

    def _append(self, row_type: str, fields: dict) -> None:
        if not self.enabled:
            return
        row = {"type": row_type, "ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **fields}
        line = json.dumps(row, separators=(",", ":"), default=str)
        try:
            with open(self._path(), "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            print(f"journal write failed: {exc}")

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
        """No `call_id`: `calibrate._labeled_pairs` skips rows that aren't
        `"decision"`, so this never pools into a fit."""
        self._append(CALIBRATION, {
            "qid": qid, "engine": engine, "model_revision": model_revision,
            "current": current, "proposed": proposed, "applied": applied,
            "n": n, "precision": precision, "human_signoff": human_signoff, "note": note,
        })

    def replay(self):
        """Every row in write order; torn/corrupt lines are skipped, not fatal."""
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
