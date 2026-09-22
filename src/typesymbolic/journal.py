"""Append-only decision/outcome journal, domain-blind by construction. A
decision row is one ask() call's answers (qid, answer value, confidence,
probabilities — judge output, not domain data) plus the concrete
`model_revision` that answered (never a requested alias like "jev-latest"),
so calibrate.py can tell whether pooled confidence data spans a model
change; an outcome row is one resolve_one() call's gate/act/verify
verdicts, joined to its decision row(s) by `call_id`.

No `state`/`questions` field, and no blob store for oversized values: a
domain-blind journal has no way to redact a domain-specific secret out of a
raw state dict, so it doesn't accept one in the first place. `record_outcome`
takes gate.py's `GateResult` and domain.py's `ActOutcome`/`Verdict` directly
rather than loose kwargs, so the payload exclusion is structural (it reads
`.verdict`/`.succeeded`/`.status`, never `.detail`) — not a discipline a
caller has to remember.

Storage root is caller-configured (`Journal(root=...)`), not a hardcoded
dotfile or env var, since one install may back more than one domain package.

Concurrent writers are safe by row size, not by locking: each row is one
`open(..., "a")` + one short `write()`, and O_APPEND makes that atomic
against other appenders — true because a row carries only ids, floats, and
short verdict strings, never a domain payload. No rotation or pruning: both
sibling implementations this package was extracted from shipped that knob
and never turned it on. One file, append-only, replayed in write order.
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

    def _append(self, row: dict) -> None:
        if not self.enabled:
            return
        line = json.dumps(row, separators=(",", ":"), default=str)
        try:
            with open(self._path(), "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            print(f"journal write failed: {exc}")

    def record_decision(
        self, *, call_id: str, engine: str, phase: str | None, answers: dict[str, Answer], model_revision: str | None = None,
    ) -> None:
        self._append({
            "type": DECISION, "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "call_id": call_id, "engine": engine, "model_revision": model_revision, "phase": phase,
            "answers": {qid: answer.model_dump(mode="json") for qid, answer in answers.items()},
        })

    def record_outcome(
        self, *, call_id: str, gate: GateResult, outcome: ActOutcome | None = None, verdict: Verdict | None = None,
    ) -> None:
        self._append({
            "type": OUTCOME, "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
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
        """Every recalibration attempt, applied or not — a cycle where
        nothing cleared the bar is as much a record as one that moved the
        threshold. Has no `call_id`; `calibrate._labeled_pairs` skips rows
        that aren't `"decision"`, so this never pools into a fit."""
        self._append({
            "type": CALIBRATION, "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "qid": qid, "engine": engine, "model_revision": model_revision,
            "current": current, "proposed": proposed, "applied": applied,
            "n": n, "precision": precision, "human_signoff": human_signoff, "note": note,
        })

    def replay(self):
        """Every row in write order; torn/corrupt lines are skipped, not
        fatal."""
        path = self._path()
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
