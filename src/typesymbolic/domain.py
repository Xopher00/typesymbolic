"""DomainAdapter protocol: what a domain plugin supplies to drive the ODAV
loop. No device- or report-shaped nouns here on purpose — `observe`/
`propose`/`act`/`verify` cover both a real-world mutation (run a command,
check its exit code) and a claim published into a report (render a finding,
check it against already-known facts) as the same four-step shape.

`act()` only ever runs on a candidate the gate has approved — a denied or
withheld pick never reaches it. `verify()` checks the outcome against
`facts_before` synchronously, using facts already in hand — a delayed,
cross-run comparison (e.g. against historical journal data) is a separate,
calibration-time operation, not this call.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class Facts(BaseModel):
    """One observation, computed once per `resolve_one()` call. `state` is
    exactly what a judge sees — a `question.py` Question is asked against it
    verbatim. `evidence` is domain-computed ground truth available to the
    gate's own checks (e.g. a groundedness or bounds check) without a second
    ask; it is not sent to the judge unless a caller folds it into `state`
    too."""

    state: dict[str, Any]
    evidence: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ActOutcome(BaseModel):
    """What happened when `act()` ran — a command's exit code, a published
    claim's text, whatever shape the domain plugin needs. `detail` is
    domain-blind on purpose: opaque to engine.py/gate.py/journal.py, read
    only by the domain plugin's own reporting or by a human debugging a
    journal row."""

    succeeded: bool
    detail: dict[str, Any] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()


class Verdict(BaseModel):
    """`verify()`'s result: whether the outcome held up against the facts
    checked, or why not."""

    status: Literal["verified", "failed", "escalated", "unconfirmed"]
    reasons: tuple[str, ...] = ()


class DomainAdapter(Protocol):
    """Everything engine.py needs from a domain plugin, and nothing more."""

    async def observe(self) -> Facts:
        """One observation. Called once per `resolve_one()`."""
        ...

    def propose(self, facts: Facts) -> dict[str, str]:
        """Real, live-enumerated candidates: id -> label. Becomes a Choice
        question's `criteria` — never invented by the judge or the caller."""
        ...

    async def act(self, chosen_id: str, facts: Facts) -> ActOutcome:
        """Run the gate-approved pick. Only called once the gate has
        approved `chosen_id`; a denied/withheld pick never reaches this."""
        ...

    async def verify(self, outcome: ActOutcome, facts_before: Facts) -> Verdict:
        """Check `outcome` against `facts_before`, using facts already in
        hand — no new observation. A delayed, cross-run backtest is a
        separate, calibration-time operation, not this call."""
        ...
