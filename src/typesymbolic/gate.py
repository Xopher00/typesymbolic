"""One confidence/deny primitive (`_confidence_verdict`), with two named
wrappers for the two policies a domain plugin picks from:

- `mutation_gate` -> act | needs_approval | deny: a hard deny short-circuits
  regardless of confidence, otherwise confidence >= threshold -> act, else
  needs_approval. The deny check itself is domain-specific (e.g. parsing a
  command against a deny-list) and stays in the domain plugin; this gate
  only consumes the resulting bool.

- `claim_gate` -> publish | hedge | withhold: a groundedness failure
  short-circuits to withhold regardless of confidence — an invented claim
  is never just hedged — otherwise confidence >= threshold -> publish, else
  hedge. The groundedness check itself is domain-specific and stays in the
  domain plugin; this gate only consumes the resulting bool.

Both wrappers call the same `_confidence_verdict` so calibrate.py's
tighten-only threshold refit has one shape of number to refit, not two.

`band` (default 0, off) widens "uncertain" to confidence within `band` of
`threshold` on either side — a 0.81 against threshold 0.8 is nominally a
pass, but a caller that wants passes to be *comfortable* passes, not
coin-flips that happened to clear the bar, can set a band instead of
hand-raising the threshold itself.

`on_uncertain`, given, overrides the verdict an uncertain call would
otherwise get (`needs_approval`/`hedge`) with a caller-chosen one from the
same gate's vocabulary — e.g. force `deny`/`withhold` instead of escalating,
or accept the risk and force `act`/`publish`. Left `None`, escalating is
still the only uncertain behavior, unchanged from before this existed.
"""

from __future__ import annotations

from dataclasses import dataclass


class GateVerdict:
    ACT = "act"
    NEEDS_APPROVAL = "needs_approval"
    DENY = "deny"
    PUBLISH = "publish"
    HEDGE = "hedge"
    WITHHOLD = "withhold"


def blocks_act(verdict: str) -> bool:
    """engine.py's default: HEDGE still reaches act() (a hedged claim is
    still published, just with a caveat); NEEDS_APPROVAL blocks because a
    single-shot resolve() has no human-in-the-loop step to resolve it."""
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
    settled_uncertain = on_uncertain if on_uncertain is not None else uncertain_verdict
    if confidence is None:
        return GateResult(ok_verdict, "no confidence to gate", confidence, call_id)
    if band > 0.0 and abs(confidence - threshold) < band:
        return GateResult(settled_uncertain, f"confidence {confidence:.2f} within {band} of threshold {threshold:.2f}", confidence, call_id)
    if confidence >= threshold:
        return GateResult(ok_verdict, f"confidence {confidence:.2f} >= threshold {threshold:.2f}", confidence, call_id)
    return GateResult(settled_uncertain, f"confidence {confidence:.2f} < threshold {threshold:.2f}", confidence, call_id)


def mutation_gate(
    confidence: float | None, threshold: float, *, denied: bool = False, band: float = 0.0,
    on_uncertain: str | None = None, call_id: str | None = None,
) -> GateResult:
    """`denied` is the domain plugin's own deny-list verdict, computed
    before this call — a denied action never runs even at confidence 1.0."""
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
    """`grounded` is the domain plugin's own groundedness check, computed
    before this call — an ungrounded claim is withheld even at confidence
    1.0."""
    if not grounded:
        return GateResult(GateVerdict.WITHHOLD, "ungrounded", confidence, call_id)
    return _confidence_verdict(
        confidence, threshold, ok_verdict=GateVerdict.PUBLISH, uncertain_verdict=GateVerdict.HEDGE,
        call_id=call_id, band=band, on_uncertain=on_uncertain,
    )
