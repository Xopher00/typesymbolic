"""Two named confidence gates over the same primitive: `mutation_gate`
(act/needs_approval/deny) and `claim_gate` (publish/hedge/withhold). Each
takes a domain-computed bool (`denied`/`grounded`) that short-circuits
regardless of confidence, then falls through to a shared threshold decision
(`circuit.threshold_decision`) so both gates and `circuit.py`'s composable
ops compare confidence the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

from .circuit import threshold_decision


class GateVerdict:
    ACT = "act"
    NEEDS_APPROVAL = "needs_approval"
    DENY = "deny"
    PUBLISH = "publish"
    HEDGE = "hedge"
    WITHHOLD = "withhold"


def blocks_act(verdict: str) -> bool:
    """HEDGE still reaches act() (published, just caveated); NEEDS_APPROVAL
    blocks — a single-shot resolve() has no human-in-the-loop step."""
    return verdict in (GateVerdict.DENY, GateVerdict.WITHHOLD, GateVerdict.NEEDS_APPROVAL)


@dataclass
class GateResult:
    verdict: str
    reason: str
    confidence: float | None = None
    call_id: str | None = None  # journal linkage to the ask() that produced this verdict


def _confidence_verdict(
    confidence: float | None, threshold: float, *, ok_verdict: str, uncertain_verdict: str, call_id: str | None,
    band: float = 0.0, on_uncertain: str | None = None,
) -> GateResult:
    passes, uncertain = threshold_decision(confidence, threshold, band=band)
    settled_uncertain = on_uncertain or uncertain_verdict
    if passes is None:
        return GateResult(ok_verdict, "no confidence to gate", confidence, call_id)
    if uncertain:
        return GateResult(settled_uncertain, f"confidence {confidence:.2f} within {band} of threshold {threshold:.2f}", confidence, call_id)
    if passes:
        return GateResult(ok_verdict, f"confidence {confidence:.2f} >= threshold {threshold:.2f}", confidence, call_id)
    return GateResult(settled_uncertain, f"confidence {confidence:.2f} < threshold {threshold:.2f}", confidence, call_id)


def mutation_gate(
    confidence: float | None, threshold: float, *, denied: bool = False, band: float = 0.0,
    on_uncertain: str | None = None, call_id: str | None = None,
) -> GateResult:
    """`denied` is the domain plugin's own deny-list verdict — a denied
    action never runs even at confidence 1.0."""
    if denied:
        return GateResult(GateVerdict.DENY, "deny_listed", confidence, call_id)
    return _confidence_verdict(
        confidence, threshold, ok_verdict=GateVerdict.ACT, uncertain_verdict=GateVerdict.NEEDS_APPROVAL,
        call_id=call_id, band=band, on_uncertain=on_uncertain,
    )


def claim_gate(
    confidence: float | None, threshold: float, *, grounded: bool = True, band: float = 0.0,
    on_uncertain: str | None = None, call_id: str | None = None,
) -> GateResult:
    """`grounded` is the domain plugin's own groundedness check — an
    ungrounded claim is withheld even at confidence 1.0."""
    if not grounded:
        return GateResult(GateVerdict.WITHHOLD, "ungrounded", confidence, call_id)
    return _confidence_verdict(
        confidence, threshold, ok_verdict=GateVerdict.PUBLISH, uncertain_verdict=GateVerdict.HEDGE,
        call_id=call_id, band=band, on_uncertain=on_uncertain,
    )
