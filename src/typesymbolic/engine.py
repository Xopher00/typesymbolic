"""The ODAV loop: observe -> propose -> ask the judge -> gate -> act ->
verify -> journal. Single-shot `resolve_one()` only — a multi-step loop
needs a second real domain plugin to validate it against.

`gate_extra`, given the pick, the facts, and the `verify_qid` answer (if
any), supplies whatever extra kwarg the chosen `gate` needs (`denied=...`
for `mutation_gate`, `grounded=...` for `claim_gate`) — `resolve_one()`
itself stays gate-shape-blind. `circuit_gate_extra()` builds a `gate_extra`
from a `circuit.py` circuit.

`store`, given, makes `threshold` the fallback and reads the live value
keyed on `(qid, judge.name, result.model_revision)`. With `journal` also
given, `resolve_one()` recalibrates that key inline before reading it,
whenever `journal`'s live index has grown past what `store` was last
calibrated against — so the threshold this qid gates on always reflects
everything verified about it before now, not just whatever a caller
remembered to recalibrate on a separate schedule. The freshness check costs
a `flush()` (bounded by however far behind the write queue currently is,
not by journal size) only on the call where it's actually needed.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .calibrate import recalibrate as _recalibrate
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
    gate_result: GateResult | None
    outcome: ActOutcome | None
    verdict: Verdict | None


async def resolve_one(
    *, judge: JudgeEngine, vocab: Vocabulary, qid: str, domain: DomainAdapter, threshold: float,
    gate: Callable[..., GateResult] = mutation_gate,
    gate_extra: Callable[[str, Facts, Answer | None], dict[str, Any]] | None = None,
    verify_qid: str | None = None, store: CalibrationStore | None = None,
    journal: Journal | None = None, slots: dict[str, str] | None = None,
) -> ResolveResult:
    """One goal -> one gated action, end to end."""
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
    if store is not None and journal is not None:
        await asyncio.to_thread(journal.flush)
        fresh_n = len(journal.labeled_pairs(qid, engine=judge.name, model_revision=result.model_revision))
        if fresh_n > store.get_n(qid, engine=judge.name, model_revision=result.model_revision):
            _recalibrate(journal=journal, store=store, qid=qid, engine=judge.name, model_revision=result.model_revision, default_threshold=threshold)
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
    """Build a `gate_extra` from a circuit: evaluates `gates` over the
    `verify_qid` answer (keyed as `verify_key`) plus `extra(facts)`, and
    returns `{kwarg: result_key(gates[key]) is True}` — an abstain/escalate
    outcome makes `result_key` return its outcome string, not `True`, so an
    uncertain circuit fails closed the same as an explicit `False`."""
    def _gate_extra(chosen_id: str, facts: Facts, verify_answer: Answer | None) -> dict[str, Any]:
        answers = dict(extra(facts)) if extra is not None else {}
        if verify_answer is not None:
            answers[verify_key] = verify_answer
        results = evaluate_gates(gates, answers)
        return {kwarg: result_key(results[key]) is True}
    return _gate_extra
