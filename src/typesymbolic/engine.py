"""The ODAV loop: observe -> decide -> gate -> act -> verify -> journal.

`ask_batch()` journals one batch per call_id, an error row on failure.
`decide()` wraps it for a `vocab`-driven `specs` map, keyed for per-key
gate/act/verify on the returned `BatchDecision`. `Episode` threads one
`episode_id` through several acts; `resolve_one()` composes these for a
single-shot Choice, its `gate_extra` optionally from `circuit_gate_extra()`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from . import calibrate
from .circuit import CircuitResult, GateSpec, evaluate_gates, result_key
from .domain import ActOutcome, DomainAdapter, Facts, Verdict
from .gate import GateResult, blocks_act, mutation_gate
from .journal import Journal
from .judge import AskResult, JudgeEngine
from .question import Answer, Choice, Noul, Question, QuestionRef
from .vocab import Vocabulary, calibration_unit

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


async def ask_batch(
    *, judge: JudgeEngine, questions: dict[str, Question], facts: Facts, journal: Journal | None = None,
    phase: str | None = None, refs: dict[str, QuestionRef] | None = None, scope: dict | None = None,
    capture: bool = False, extra: dict | None = None, call_id: str | None = None,
) -> tuple[str, AskResult]:
    """Ask a batch in one request, journaled under one `call_id`. On failure,
    journals an error row with empty answers, then re-raises."""
    if call_id is None:
        call_id = str(uuid.uuid4())
    state_cap = facts.state if capture else None
    asked_cap = ({k: q.model_dump(mode="json", exclude_none=True)
                  for k, q in questions.items()} if capture else None)
    try:
        result = await judge.ask_all(facts.state, questions)
    except Exception as error:
        if journal is not None:
            err_extra = dict(extra) if extra else {}
            journal.record_decision(
                call_id=call_id, engine=judge.name, phase=phase, answers={},
                questions=refs, scope=scope, state=state_cap, asked=asked_cap,
                error=str(error), extra=err_extra if err_extra else None,
            )
        raise
    if journal is not None:
        dec_extra = {"usage": result.usage} if result.usage else {}
        if extra:
            dec_extra.update(extra)
        journal.record_decision(
            call_id=call_id, engine=judge.name, phase=phase, answers=result.answers,
            model_revision=result.model_revision, questions=refs, scope=scope,
            state=state_cap, asked=asked_cap, elapsed_ms=result.elapsed_ms,
            extra=dec_extra if dec_extra else None,
        )
    return call_id, result


async def decide(
    *, judge: JudgeEngine, vocab: Vocabulary, specs: dict[str, tuple[str, dict | None, str | None, Any]],
    facts: Facts, journal: Journal | None = None, phase: str | None = None,
    scope: dict | None = None, store: CalibrationStore | None = None,
    capture: bool = False, extra: dict | None = None, call_id: str | None = None,
) -> BatchDecision:
    """One batched ask over `specs` = `{key: (qid, slots, subject, criteria)}`,
    each key independently gate/act/verify-able off the returned `BatchDecision`."""
    questions: dict[str, Question] = {}
    refs: dict[str, QuestionRef] = {}
    for key, (qid, slots, subject, criteria) in specs.items():
        question = vocab.ask(qid, criteria=criteria, **(slots or {}))
        questions[key] = question
        group, scale = calibration_unit(vocab, qid, question)
        refs[key] = QuestionRef(
            qid=qid, vocab_version=vocab.version, subject=subject, group=group, scale=scale,
        )
    call_id, result = await ask_batch(
        judge=judge, questions=questions, facts=facts, journal=journal, phase=phase, refs=refs, scope=scope,
        capture=capture, extra=extra, call_id=call_id,
    )
    return BatchDecision(
        call_id, result.answers, refs, result.model_revision, facts, judge.name, journal, store, scope,
    )


@dataclass
class BatchDecision:
    """One `decide()` result: several answer keys sharing one call_id."""

    call_id: str
    answers: dict[str, Answer]
    refs: dict[str, QuestionRef]
    model_revision: str | None
    facts: Facts
    engine: str
    journal: Journal | None = None
    store: CalibrationStore | None = None
    scope: dict | None = None
    _acted: list[str] = field(default_factory=list, init=False, repr=False)

    async def threshold(self, key: str, default: float) -> float:
        """The calibrated threshold for `key`'s unit, or `default` with no store."""
        if self.store is None:
            return default
        ref = self.refs[key]
        return await calibrate.current_threshold(
            journal=self.journal, store=self.store, group=ref.group, scale=ref.scale,
            engine=self.engine, model_revision=self.model_revision, default_threshold=default,
        )

    async def gate(
        self, key: str, gate_fn: Callable[..., GateResult], *, default: float, **kw: Any,
    ) -> GateResult:
        """Runs `gate_fn` on `key`'s answer at its calibrated threshold."""
        tau = await self.threshold(key, default)
        return gate_fn(self.answers[key].value(self.refs[key].scale), tau, call_id=self.call_id, **kw)

    async def circuit(self, gate_specs: dict[str, GateSpec]) -> dict[str, CircuitResult]:
        """`evaluate_gates`, with each gate id naming a key getting its `tau`
        resolved via `threshold()` first."""
        resolved: dict[str, GateSpec] = {}
        for gid, spec in gate_specs.items():
            if gid in self.refs:
                resolved[gid] = replace(spec, tau=await self.threshold(gid, spec.tau))
            else:
                resolved[gid] = spec
        return evaluate_gates(resolved, self.answers)

    def act(self, key: str, outcome: ActOutcome | None, gate: GateResult | None) -> None:
        """Journals Gate+Act for `key`; `outcome=None` records a blocked gate."""
        self._acted.append(key)
        if self.journal is not None:
            episode_id = (self.scope or {}).get("episode_id")
            self.journal.record_outcome(
                call_id=self.call_id, gate=gate, outcome=outcome, key=key, episode_id=episode_id,
            )

    def verify(self, verdict: Verdict) -> None:
        """Journals Verify; `verdict.tests` defaults to every acted key."""
        if self.journal is not None:
            tests = verdict.tests if verdict.tests is not None else tuple(self._acted)
            self.journal.record_verdict(call_id=self.call_id, verdict=verdict, tests=tests)


class Episode:
    """One `episode_id` threaded through several acts, e.g. a planner's tiers."""

    def __init__(self, journal: Journal | None, scope: dict | None = None) -> None:
        self.journal = journal
        self.episode_id = str(uuid.uuid4())
        self.scope = {**(scope or {}), "episode_id": self.episode_id}

    def outcome(
        self, outcome: ActOutcome, *, gate: GateResult | None = None,
        call_id: str | None = None, key: str | None = None,
    ) -> None:
        """Journals one Act step of this episode; `gate=None` is a code-policy act."""
        if self.journal is not None:
            self.journal.record_outcome(
                call_id=call_id or str(uuid.uuid4()), gate=gate, outcome=outcome,
                key=key, episode_id=self.episode_id,
            )


def _expect(vocab: Vocabulary, qid: str, slots: dict, kind: type, noun: str) -> Question:
    """`vocab.ask(qid)`, raising if it isn't a `kind` question."""
    question = vocab.ask(qid, **slots)
    if not isinstance(question, kind):
        raise TypeError(f"{qid}: {noun} vocab returned {type(question).__name__}")
    return question


async def resolve_one(
    *, judge: JudgeEngine, vocab: Vocabulary, qid: str, domain: DomainAdapter, threshold: float,
    gate: Callable[..., GateResult] = mutation_gate,
    gate_extra: Callable[[str, Facts, Answer | None], dict[str, Any]] | None = None,
    verify_qid: str | None = None, store: CalibrationStore | None = None,
    journal: Journal | None = None, slots: dict[str, str] | None = None,
) -> ResolveResult:
    """One goal -> one gated action, end to end, built on `decide`/`gate`/`act`/`verify`."""
    facts = await domain.observe()
    candidates = domain.propose(facts)
    if not candidates:
        return ResolveResult("escalated", ("no candidates fit",), str(uuid.uuid4()), facts, None, None, None)

    resolved_slots = slots or {}
    _expect(vocab, qid, resolved_slots, Choice, "resolve_one() drives a Choice question;")
    specs = {qid: (qid, resolved_slots, None, candidates)}
    if verify_qid is not None:
        _expect(vocab, verify_qid, resolved_slots, Noul, "verify_qid must be a Noul question;")
        specs[verify_qid] = (verify_qid, resolved_slots, None, None)

    decision = await decide(
        judge=judge, vocab=vocab, specs=specs, facts=facts, journal=journal, phase="decide", store=store,
    )
    answer = decision.answers[qid]
    verify_answer = decision.answers.get(verify_qid) if verify_qid is not None else None
    chosen_id = answer.choice
    if chosen_id not in candidates:
        reason = f"judge picked {chosen_id!r}, outside the live-enumerated candidates"
        return ResolveResult("escalated", (reason,), decision.call_id, facts, None, None, None)

    extra = gate_extra(chosen_id, facts, verify_answer) if gate_extra is not None else {}
    gate_result = await decision.gate(qid, gate, default=threshold, **extra)
    if blocks_act(gate_result.verdict):
        decision.act(qid, None, gate_result)
        return ResolveResult(
            gate_result.verdict, (gate_result.reason,), decision.call_id, facts, gate_result, None, None,
        )

    outcome = await domain.act(chosen_id, facts)
    verdict = await domain.verify(outcome, facts)
    decision.act(qid, outcome, gate_result)
    decision.verify(verdict)
    return ResolveResult(
        verdict.status, verdict.reasons, decision.call_id, facts, gate_result, outcome, verdict,
    )


def circuit_gate_extra(
    gates: dict[str, GateSpec], *, key: str, kwarg: str,
    verify_key: str = "verify", extra: Callable[[Facts], dict[str, Answer]] | None = None,
) -> Callable[[str, Facts, Answer | None], dict[str, Any]]:
    """Build a `gate_extra` from a circuit over the `verify_qid` answer
    (keyed as `verify_key`) plus `extra(facts)`. An abstain/escalate
    outcome fails closed, same as an explicit `False`."""
    def _gate_extra(chosen_id: str, facts: Facts, verify_answer: Answer | None) -> dict[str, Any]:
        answers = dict(extra(facts)) if extra is not None else {}
        if verify_answer is not None:
            answers[verify_key] = verify_answer
        results = evaluate_gates(gates, answers)
        return {kwarg: result_key(results[key]) is True}
    return _gate_extra
