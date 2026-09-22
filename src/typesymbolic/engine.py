"""The ODAV loop: observe -> propose candidates -> ask the judge (a
vocab-sourced Choice) -> gate -> act -> verify -> journal. Single-shot
`resolve_one()` only — a multi-step loop needs a second real domain plugin
to validate it against before it's worth building.

A domain's own deny/groundedness check often needs more than the candidate
id `propose()` returns (e.g. the literal command a pick would run), and
`act()` is the only place that gets built. `gate_extra` lets a domain plugin
compute its own denied/grounded flag from (chosen_id, facts) before act()
runs, without `resolve_one()` needing to know the gate's shape.

`verify_qid`, when given, asks the judge a second, Noul-shaped question in
the same batch as the primary pick — "is this supported?" — instead of the
domain hand-rolling its own groundedness heuristic. Its `Answer` reaches
`gate_extra` as a third argument, so grounding a claim is still a calibrated
judgment, not code guessing at support from raw evidence.

`store`, given, makes `threshold` the fallback for a question calibration
hasn't seen yet and reads the live value keyed on `(qid, judge.name,
result.model_revision)` — the same key `calibrate.recalibrate()` writes to.
Read per call and never cached here, since a cached value would ignore a
recalibration that just ran.

`circuit_gate_extra()` turns a `circuit.py` circuit into a ready-to-pass
`gate_extra`, so composing more than one confidence threshold into a gate
decision doesn't need a hand-written closure per call site.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .circuit import GateSpec, evaluate_gates, result_key
from .domain import ActOutcome, DomainAdapter, Facts, Verdict
from .gate import GateResult, blocks_act, mutation_gate
from .journal import Journal
from .judge import JudgeEngine
from .question import Answer, Choice, Noul
from .vocab import Vocabulary

if TYPE_CHECKING:
    from .calibration_store import CalibrationStore


@dataclass
class ResolveResult:
    status: str
    reasons: tuple[str, ...]
    call_id: str
    facts: Facts
    gate: GateResult | None
    outcome: ActOutcome | None
    verdict: Verdict | None


async def resolve_one(
    *, judge: JudgeEngine, vocab: Vocabulary, qid: str, domain: DomainAdapter, threshold: float,
    gate: Callable[..., GateResult] = mutation_gate,
    gate_extra: Callable[[str, Facts, Answer | None], dict[str, Any]] | None = None,
    verify_qid: str | None = None, store: CalibrationStore | None = None,
    journal: Journal | None = None, slots: dict[str, str] | None = None,
) -> ResolveResult:
    """One goal -> one gated action, end to end. `gate` defaults to
    `mutation_gate`; pass `claim_gate` for a publish-shaped domain.
    `gate_extra`, given the judge's pick, the observed facts, and the
    `verify_qid` answer (`None` if `verify_qid` wasn't given), supplies
    whatever extra kwarg that gate needs (`denied=...` for `mutation_gate`,
    `grounded=...` for `claim_gate`) — resolve_one() itself stays
    gate-shape-blind."""
    facts = await domain.observe()
    candidates = domain.propose(facts)
    call_id = str(uuid.uuid4())

    if not candidates:
        return ResolveResult("escalated", ("no candidates fit",), call_id, facts, None, None, None)

    template = vocab.ask(qid, **(slots or {}))
    if not isinstance(template, Choice):
        raise TypeError(f"{qid}: resolve_one() drives a Choice question; vocab returned {type(template).__name__}")
    questions = {qid: Choice(instructions=template.instructions, criteria=candidates)}

    if verify_qid is not None:
        verify_template = vocab.ask(verify_qid, **(slots or {}))
        if not isinstance(verify_template, Noul):
            raise TypeError(f"{verify_qid}: verify_qid must be a Noul question; vocab returned {type(verify_template).__name__}")
        questions[verify_qid] = verify_template

    result = await judge.ask_all(facts.state, questions)
    answer = result.answers[qid]
    verify_answer = result.answers.get(verify_qid) if verify_qid is not None else None
    if journal is not None:
        journal.record_decision(
            call_id=call_id, engine=judge.name, phase="decide", answers=result.answers,
            model_revision=result.model_revision,
        )

    chosen_id = answer.choice
    if chosen_id not in candidates:
        return ResolveResult(
            "escalated", (f"judge picked {chosen_id!r}, outside the live-enumerated candidates",),
            call_id, facts, None, None, None,
        )

    extra = gate_extra(chosen_id, facts, verify_answer) if gate_extra is not None else {}
    gate_threshold = (
        store.get(qid, engine=judge.name, model_revision=result.model_revision, default=threshold)
        if store is not None else threshold
    )
    gate_result = gate(answer.confidence, gate_threshold, call_id=call_id, **extra)

    if blocks_act(gate_result.verdict):
        if journal is not None:
            journal.record_outcome(call_id=call_id, gate=gate_result)
        return ResolveResult(gate_result.verdict, (gate_result.reason,), call_id, facts, gate_result, None, None)

    outcome = await domain.act(chosen_id, facts)
    verdict = await domain.verify(outcome, facts)
    if journal is not None:
        journal.record_outcome(call_id=call_id, gate=gate_result, outcome=outcome, verdict=verdict)

    return ResolveResult(verdict.status, verdict.reasons, call_id, facts, gate_result, outcome, verdict)


def circuit_gate_extra(
    gates: dict[str, GateSpec], *, key: str, kwarg: str,
    verify_key: str = "verify", extra: Callable[[Facts], dict[str, Answer]] | None = None,
) -> Callable[[str, Facts, Answer | None], dict[str, Any]]:
    """Turn a composed circuit into a ready-to-pass `gate_extra`: evaluates
    `gates` over the `verify_qid` answer (keyed as `verify_key`) plus
    whatever `extra(facts)` supplies, then returns `{kwarg: result_key(gates[key]) is True}`.
    Fails closed — an abstain/escalate outcome makes `result_key` return its
    outcome string, not `True`, so an uncertain circuit denies/withholds
    exactly like a `False` would."""
    def gate_extra(chosen_id: str, facts: Facts, verify_answer: Answer | None) -> dict[str, Any]:
        answers = dict(extra(facts)) if extra is not None else {}
        if verify_answer is not None:
            answers[verify_key] = verify_answer
        results = evaluate_gates(gates, answers)
        return {kwarg: result_key(results[key]) is True}
    return gate_extra
