"""Composable gates over calibrated `Answer`s — AND/OR/NOT, majority vote,
verify, ordinal bucketing. `gate.py` stays the simple, common-case entry
point; this module is for a decision that needs more than one threshold.

A `GateSpec` reads one or more answer ids, or an earlier gate id in the same
circuit (a small DAG, not a flat list), and produces a `CircuitResult`.

Ops (`input`/`inputs` is an answer id, "answer_id:option" for a choice/score
answer's probability of one option, or an earlier gate id):

    threshold  input: noul               value: bool, passes if p >= tau
    not        input: noul                value: bool, p = 1 - p_in
    and / or   inputs: k nouls            value: bool
               combine: "product" (default, independence) / "weak"
               (min/max, idempotent) / "strong" (bounded sum/difference,
               saturating) — the three continuous t-norms, not a menu to
               extend
    majority   inputs: k choices, same option set   value: option
    argmax     input: choice              value: option, uncertain if
               confidence < min_confidence
    verify     input: choice, check: noul  value: option, uncertain if
               P(check) < tau or confidence < min_confidence
    order      input: score, cutpoints: [c1, c2, ...]  value: bucket index
    confidence input: any answer (noul/choice/score)   value: that answer's
               own value, uncertain if confidence < min_confidence (banded)
               -- argmax/verify's confidence floor, usable standalone for a
               noul or score answer they don't cover

Every gate has `on_uncertain`: "abstain" | "escalate" | "default" (with a
`default` value); uncertain when the probability it acts on is within
`band` of `tau`, or an argmax/verify confidence check fails — surfaced in
`CircuitResult.outcome`, never silently resolved into a value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .question import Answer

OPS = ("threshold", "not", "and", "or", "majority", "argmax", "verify", "order", "confidence")
POLICIES = ("abstain", "escalate", "default")
OUTCOMES = ("decided", "abstain", "escalate", "default")
COMBINES = ("product", "weak", "strong")


class CircuitError(ValueError):
    """A gate spec was malformed, or referenced an answer that doesn't fit
    the op (e.g. `majority` over a non-choice answer)."""


@dataclass
class GateSpec:
    op: str
    input: str | None = None
    inputs: list[str] | None = None
    check: str | None = None
    tau: float = 0.5
    band: float = 0.1
    min_confidence: float = 0.0
    cutpoints: list[float] | None = None
    combine: str = "product"
    on_uncertain: str = "abstain"
    default: Any = None

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise CircuitError(f"unknown gate op {self.op!r}; expected one of {OPS}")
        if self.on_uncertain not in POLICIES:
            raise CircuitError(f"on_uncertain must be one of {POLICIES}, got {self.on_uncertain!r}")
        if self.combine not in COMBINES:
            raise CircuitError(f"combine must be one of {COMBINES}, got {self.combine!r}")
        if self.combine != "product" and self.op not in ("and", "or"):
            raise CircuitError(f"combine is only meaningful for 'and'/'or', not {self.op!r}")
        if self.op in ("threshold", "not", "argmax", "verify", "order", "confidence") and not self.input:
            raise CircuitError(f"gate op {self.op!r} needs `input`")
        if self.op in ("and", "or", "majority") and not self.inputs:
            raise CircuitError(f"gate op {self.op!r} needs `inputs`")
        if self.op == "verify" and not self.check:
            raise CircuitError("gate op 'verify' needs `check`")
        if self.op == "order" and self.cutpoints is None:
            raise CircuitError("gate op 'order' needs `cutpoints`")


@dataclass
class CircuitResult:
    value: Any
    p: float | None = None
    confidence: float | None = None
    uncertain: bool = False
    outcome: str = "decided"
    trace: list[str] = field(default_factory=list)


def result_key(result: CircuitResult) -> Any:
    """What a result routes on: its value when it decided (or fell back to
    its default), otherwise its outcome ("abstain" or "escalate")."""
    return result.value if result.outcome in ("decided", "default") else result.outcome


def threshold_decision(value: float | None, tau: float, *, band: float = 0.0) -> tuple[bool | None, bool]:
    """(passes, uncertain). `value=None` -> `(None, False)`. Shared by
    `gate.py`'s two named gates and this module's `threshold`/`not`/`and`/`or`."""
    if value is None:
        return None, False
    return value >= tau, band > 0.0 and abs(value - tau) < band


def _uncertain_note(uncertain: bool) -> str:
    return " (uncertain)" if uncertain else ""


def _noul_p(answers: dict[str, Answer], results: dict[str, CircuitResult], ref: str) -> tuple[float, str, bool]:
    if ref in results:
        r = results[ref]
        if r.p is None:
            raise CircuitError(f"gate {ref!r} carries no probability; use '{ref}:<value>'")
        return float(r.p), f"gate {ref} p={r.p:.2f}{_uncertain_note(r.uncertain)}", r.uncertain
    if ":" in ref:
        qid, opt = ref.split(":", 1)
        if qid in results:
            r = results[qid]
            p = float(r.p if r.p is not None else 1.0)
            match = str(r.value) == opt
            pm = p if match else max(0.0, 1.0 - p)
            return pm, f"gate {qid}={r.value} -> P({opt})={pm:.2f}{_uncertain_note(r.uncertain)}", r.uncertain
        a = answers[qid]
        if a.type in ("choice", "score") and a.probabilities:
            return float(a.probabilities[opt]), f"{qid}[{opt}] p={a.probabilities[opt]:.2f}", False
        raise CircuitError(f"{qid!r} is not a choice or score with probabilities")
    a = answers[ref]
    if a.type != "noul" or a.noul is None:
        raise CircuitError(f"{ref!r} is not a noul; use 'answer_id:option' for a choice/score")
    return float(a.noul), f"{ref} p={a.noul:.2f}", False


_COMBINE_NOTES = {
    "product": "under independence",
    "weak": "weak/min-max (idempotent)",
    "strong": "strong/bounded-sum (saturating)",
}


def _combine(op: str, how: str, ps: list[float]) -> tuple[float, str]:
    """`ps` is never empty: `__post_init__` requires `inputs`."""
    if how == "weak":
        p = min(ps) if op == "and" else max(ps)
    elif how == "strong":
        total = sum(ps)
        p = max(0.0, total - (len(ps) - 1)) if op == "and" else min(1.0, total)
    elif op == "and":
        p = 1.0
        for x in ps:
            p *= x
    else:
        q = 1.0
        for x in ps:
            q *= 1 - x
        p = 1 - q
    return p, _COMBINE_NOTES[how]


def _settle(g: GateSpec, value: Any, p: float | None, uncertain: bool, trace: list[str], confidence: float | None = None) -> CircuitResult:
    if not uncertain:
        return CircuitResult(value=value, p=p, confidence=confidence, trace=trace)
    if g.on_uncertain == "default":
        trace.append(f"uncertain -> default {g.default!r}")
        return CircuitResult(value=g.default, p=p, confidence=confidence, uncertain=True, outcome="default", trace=trace)
    trace.append(f"uncertain -> {g.on_uncertain}")
    return CircuitResult(value=None, p=p, confidence=confidence, uncertain=True, outcome=g.on_uncertain, trace=trace)


def evaluate_gates(gates: dict[str, GateSpec], answers: dict[str, Answer]) -> dict[str, CircuitResult]:
    """Evaluate every gate in declaration order; a gate may reference any
    earlier gate's id, so declaration order must respect dependencies (a
    forward reference raises `KeyError` from the missing lookup)."""
    results: dict[str, CircuitResult] = {}
    for gid, g in gates.items():
        trace: list[str] = []
        if g.op == "threshold":
            p, t, unc = _noul_p(answers, results, g.input)
            trace.append(t)
            passes, band_unc = threshold_decision(p, g.tau, band=g.band)
            results[gid] = _settle(g, passes, p, unc or band_unc, trace)
        elif g.op == "not":
            p, t, unc = _noul_p(answers, results, g.input)
            trace += [t, f"not -> p={1 - p:.2f}"]
            passes, band_unc = threshold_decision(1 - p, g.tau, band=g.band)
            results[gid] = _settle(g, passes, 1 - p, unc or band_unc, trace)
        elif g.op in ("and", "or"):
            ps: list[float] = []
            unc = False
            for ref in g.inputs or []:
                p, t, u = _noul_p(answers, results, ref)
                ps.append(p)
                unc = unc or u
                trace.append(t)
            p, note = _combine(g.op, g.combine, ps)
            trace.append(f"{g.op} {note} -> p={p:.2f}")
            passes, band_unc = threshold_decision(p, g.tau, band=g.band)
            results[gid] = _settle(g, passes, p, unc or band_unc, trace)
        elif g.op == "majority":
            votes: dict[str, list[float]] = {}
            for ref in g.inputs or []:
                a = answers[ref]
                if a.type != "choice" or a.choice is None or not a.probabilities:
                    raise CircuitError(f"majority input {ref!r} must be a choice")
                votes.setdefault(a.choice, []).append(float(a.probabilities[a.choice]))
                trace.append(f"{ref} -> {a.choice} ({a.probabilities[a.choice]:.2f})")
            n = len(g.inputs or [])
            winner, ps = max(votes.items(), key=lambda kv: (len(kv[1]), sum(kv[1])))
            margin = len(ps) / n
            p = sum(ps) / len(ps)
            trace.append(f"majority {winner} {len(ps)}/{n}, mean p={p:.2f}")
            results[gid] = _settle(g, winner, p, margin <= 0.5 or p < g.min_confidence, trace, confidence=margin)
        elif g.op == "argmax":
            a = answers[g.input]
            if a.type != "choice" or a.choice is None or not a.probabilities:
                raise CircuitError(f"argmax input {g.input!r} must be a choice")
            conf = float(a.confidence)
            p = float(a.probabilities[a.choice])
            trace.append(f"{g.input} -> {a.choice} p={p:.2f} conf={conf:.2f} (min {g.min_confidence})")
            results[gid] = _settle(g, a.choice, p, conf < g.min_confidence, trace, confidence=conf)
        elif g.op == "verify":
            a = answers[g.input]
            if a.type != "choice" or a.choice is None:
                raise CircuitError(f"verify input {g.input!r} must be a choice")
            conf = float(a.confidence)
            p_check, t, u = _noul_p(answers, results, g.check)
            trace += [f"{g.input} -> {a.choice} conf={conf:.2f}", f"check {t} (tau {g.tau})"]
            unc = u or conf < g.min_confidence or p_check < g.tau
            results[gid] = _settle(g, a.choice, p_check, unc, trace, confidence=conf)
        elif g.op == "order":
            a = answers[g.input]
            if a.type != "score" or a.score is None:
                raise CircuitError(f"order input {g.input!r} must be a score")
            s = float(a.score)
            cuts = g.cutpoints or []
            bucket = sum(1 for c in cuts if s >= c)
            near = any(abs(s - c) < g.band for c in cuts)
            trace.append(f"{g.input} score={s:.2f} cutpoints={cuts} -> bucket {bucket}" + (" (near a cutpoint)" if near else ""))
            results[gid] = _settle(g, bucket, None, near, trace, confidence=float(a.confidence))
        elif g.op == "confidence":
            a = answers[g.input]
            conf = float(a.confidence)
            value = a.noul if a.type == "noul" else a.choice if a.type == "choice" else a.score
            passes, band_unc = threshold_decision(conf, g.min_confidence, band=g.band)
            trace.append(f"{g.input} conf={conf:.2f} (min {g.min_confidence}){_uncertain_note(band_unc)}")
            results[gid] = _settle(g, value, conf, not passes or band_unc, trace, confidence=conf)
    return results
