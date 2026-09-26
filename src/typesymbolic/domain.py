"""DomainAdapter protocol: what a domain plugin supplies to drive the ODAV
loop. `act()` only runs on a gate-approved candidate. `verify()` may check
synchronously, or return `status="unconfirmed"` and let a real `Verdict`
follow later via `journal.Journal.record_verdict()`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class Facts(BaseModel):
    """One observation. `state` is exactly what a judge sees. `evidence` is
    domain-computed ground truth for the gate's own checks, not sent to
    the judge unless folded into `state`."""

    state: dict[str, Any]
    evidence: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ActStep(BaseModel):
    """One step of a multi-step act (a planner tier, a repo-activity
    publish/drop/rank). `detail` is domain-blind, same as `ActOutcome`."""

    name: str
    succeeded: bool | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class ActOutcome(BaseModel):
    """What `act()` did. `detail` is domain-blind: opaque to engine/gate/
    journal, read only by the domain plugin itself. `key` names which
    journaled answer key this outcome acted on; `steps` breaks a
    multi-step act into its parts."""

    succeeded: bool
    detail: dict[str, Any] = Field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    key: str | None = None
    steps: tuple[ActStep, ...] = ()


class Verdict(BaseModel):
    """`verify()`'s result. Only `verified`/`failed` label anything.

    `tests` names which journaled answer keys this verdict speaks to;
    `None` lets the caller default it. `dedupe_key` makes re-recording the
    same observation a no-op. `observed_at` decides which of two verdicts
    for the same `(call_id, key)` supersedes the other.
    """

    status: Literal["verified", "failed", "escalated", "unconfirmed"]
    reasons: tuple[str, ...] = ()
    tests: tuple[str, ...] | None = None
    dedupe_key: str | None = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    calibrate: bool = True  # False: recorded but excluded from threshold fitting (e.g. a contradiction)


class DomainAdapter(Protocol):
    """Everything engine.py needs from a domain plugin."""

    async def observe(self) -> Facts:
        """One observation per `resolve_one()` call."""
        ...

    def propose(self, facts: Facts) -> dict[str, str]:
        """Live-enumerated candidates: id -> label. Never invented by the
        judge or caller."""
        ...

    async def act(self, chosen_id: str, facts: Facts) -> ActOutcome:
        """Runs only on a gate-approved `chosen_id`."""
        ...

    async def verify(self, outcome: ActOutcome, facts_before: Facts) -> Verdict:
        """Checks `outcome` against facts already in hand -- no new
        observation. A deferred check is a separate `Verdict`, not this call."""
        ...
