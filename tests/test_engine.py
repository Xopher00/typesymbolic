import pytest

from typesymbolic.calibration_store import CalibrationStore
from typesymbolic.circuit import GateSpec, evaluate_gates, result_key
from typesymbolic.domain import ActOutcome, Facts, Verdict
from typesymbolic.engine import (
    Episode,
    ask_batch,
    circuit_gate_extra,
    decide,
    resolve_one,
)
from typesymbolic.gate import claim_gate, mutation_gate
from typesymbolic.journal import Journal
from typesymbolic.judge import ScriptedJudge
from typesymbolic.labels import LabelIndex
from typesymbolic.question import Answer, Choice, Noul, Score
from typesymbolic.vocab import FrozenVocabulary


class Vocab:
    version = "test-1"

    def ask(self, qid, **slots):
        if qid.startswith("verify."):
            return Noul(instructions=f"is the pick for {qid} supported?")
        return Choice(instructions=f"pick an option for {qid}", criteria={})


class DeviceAdapter:
    """Mutation-style: act() runs only when the gate says ACT."""

    def __init__(self) -> None:
        self.acted_on = None

    async def observe(self) -> Facts:
        return Facts(state={"goal": "turn on wifi"})

    def propose(self, facts: Facts) -> dict[str, str]:
        return {"toggle_service": "turns a radio on/off"}

    async def act(self, chosen_id: str, facts: Facts) -> ActOutcome:
        self.acted_on = chosen_id
        return ActOutcome(succeeded=True, detail={"exit_code": 0})

    async def verify(self, outcome: ActOutcome, facts_before: Facts) -> Verdict:
        return Verdict(status="verified")


class NoCandidatesAdapter(DeviceAdapter):
    def propose(self, facts: Facts) -> dict[str, str]:
        return {}


class ClaimAdapter:
    """Claim/publish-style: act() runs for PUBLISH and HEDGE, not WITHHOLD."""

    def __init__(self) -> None:
        self.acted_on = None

    async def observe(self) -> Facts:
        return Facts(state={"path": "core/auth.py"})

    def propose(self, facts: Facts) -> dict[str, str]:
        return {"low": "low risk", "high": "near-critical risk"}

    async def act(self, chosen_id: str, facts: Facts) -> ActOutcome:
        self.acted_on = chosen_id
        return ActOutcome(succeeded=True, detail={"claim": chosen_id})

    async def verify(self, outcome: ActOutcome, facts_before: Facts) -> Verdict:
        return Verdict(status="verified")


async def test_resolve_one_acts_and_verifies_on_confident_pick(tmp_path):
    judge = ScriptedJudge([{"kind.pick": Answer.from_choice("kind.pick", "toggle_service", {"toggle_service": 0.95}, confidence=0.95)}])
    domain = DeviceAdapter()
    journal = Journal(root=tmp_path)

    result = await resolve_one(judge=judge, vocab=Vocab(), qid="kind.pick", domain=domain, threshold=0.8, journal=journal)

    assert result.status == "verified"
    assert domain.acted_on == "toggle_service"
    rows = list(journal.replay())
    assert [r["type"] for r in rows] == ["decision", "outcome", "verdict"]
    verdict_row = rows[2]
    assert verdict_row["status"] == "verified"
    assert verdict_row["tests"] == ["kind.pick"]  # defaulted to the qid resolve_one() gated on


async def test_resolve_one_escalates_on_low_confidence_without_acting():
    judge = ScriptedJudge([{"kind.pick": Answer.from_choice("kind.pick", "toggle_service", {"toggle_service": 0.5}, confidence=0.5)}])
    domain = DeviceAdapter()

    result = await resolve_one(judge=judge, vocab=Vocab(), qid="kind.pick", domain=domain, threshold=0.8)

    assert result.status == "needs_approval"
    assert domain.acted_on is None


async def test_resolve_one_denies_regardless_of_confidence_via_gate_extra():
    judge = ScriptedJudge([{"kind.pick": Answer.from_choice("kind.pick", "toggle_service", {"toggle_service": 0.99}, confidence=0.99)}])
    domain = DeviceAdapter()

    result = await resolve_one(
        judge=judge, vocab=Vocab(), qid="kind.pick", domain=domain, threshold=0.8,
        gate=mutation_gate, gate_extra=lambda chosen_id, facts, verify: {"denied": True},
    )

    assert result.status == "deny"
    assert domain.acted_on is None


async def test_resolve_one_escalates_immediately_with_no_candidates():
    judge = ScriptedJudge([])  # never consulted -- no candidates short-circuits before ask_all()
    domain = NoCandidatesAdapter()

    result = await resolve_one(judge=judge, vocab=Vocab(), qid="kind.pick", domain=domain, threshold=0.8)

    assert result.status == "escalated"
    assert judge.calls == []


async def test_resolve_one_escalates_when_judge_invents_an_id():
    judge = ScriptedJudge([{"kind.pick": Answer.from_choice("kind.pick", "not_a_real_candidate", {"not_a_real_candidate": 0.9}, confidence=0.9)}])
    domain = DeviceAdapter()

    result = await resolve_one(judge=judge, vocab=Vocab(), qid="kind.pick", domain=domain, threshold=0.8)

    assert result.status == "escalated"
    assert "outside the live-enumerated candidates" in result.reasons[0]
    assert domain.acted_on is None


async def test_resolve_one_claim_gate_publishes_and_hedges_still_act():
    domain = ClaimAdapter()
    judge = ScriptedJudge([{"risk.pick": Answer.from_choice("risk.pick", "low", {"low": 0.5, "high": 0.5}, confidence=0.5)}])

    result = await resolve_one(
        judge=judge, vocab=Vocab(), qid="risk.pick", domain=domain, threshold=0.8,
        gate=claim_gate, gate_extra=lambda chosen_id, facts, verify: {"grounded": True},
    )

    # hedge still reaches act(), unlike mutation_gate's needs_approval
    assert result.gate_result.verdict == "hedge"
    assert domain.acted_on == "low"
    assert result.status == "verified"


async def test_resolve_one_claim_gate_withholds_ungrounded_claim():
    domain = ClaimAdapter()
    judge = ScriptedJudge([{"risk.pick": Answer.from_choice("risk.pick", "high", {"high": 0.99}, confidence=0.99)}])

    result = await resolve_one(
        judge=judge, vocab=Vocab(), qid="risk.pick", domain=domain, threshold=0.8,
        gate=claim_gate, gate_extra=lambda chosen_id, facts, verify: {"grounded": False},
    )

    assert result.status == "withhold"
    assert domain.acted_on is None


async def test_resolve_one_verify_qid_batches_a_second_question_and_grounds_the_gate():
    domain = ClaimAdapter()
    judge = ScriptedJudge([{
        "risk.pick": Answer.from_choice("risk.pick", "high", {"high": 0.95}, confidence=0.95),
        "verify.grounded": Answer.from_noul("verify.grounded", 0.1),
    }])

    result = await resolve_one(
        judge=judge, vocab=Vocab(), qid="risk.pick", domain=domain, threshold=0.8,
        gate=claim_gate, verify_qid="verify.grounded",
        gate_extra=lambda chosen_id, facts, verify: {"grounded": verify.noul >= 0.5},
    )

    assert judge.calls[0][1].keys() == {"risk.pick", "verify.grounded"}  # one batched ask_all(), not two
    assert result.status == "withhold"
    assert domain.acted_on is None


async def test_resolve_one_journals_refs_for_the_pick_and_verify_questions(tmp_path):
    journal = Journal(root=tmp_path)
    domain = ClaimAdapter()
    judge = ScriptedJudge([{
        "risk.pick": Answer.from_choice("risk.pick", "high", {"high": 0.95}, confidence=0.95),
        "verify.grounded": Answer.from_noul("verify.grounded", 0.9),
    }])

    await resolve_one(
        judge=judge, vocab=Vocab(), qid="risk.pick", domain=domain, threshold=0.8,
        gate=claim_gate, verify_qid="verify.grounded", journal=journal,
        gate_extra=lambda chosen_id, facts, verify: {"grounded": verify.noul >= 0.5},
    )

    decision = next(row for row in journal.replay() if row["type"] == "decision")
    assert decision["questions"]["risk.pick"] == {
        "qid": "risk.pick", "vocab_version": "test-1", "subject": None, "group": "risk.pick", "scale": "confidence",
    }
    assert decision["questions"]["verify.grounded"]["scale"] == "noul_p"


async def test_resolve_one_verify_qid_must_be_a_noul():
    domain = ClaimAdapter()
    judge = ScriptedJudge([{}])

    try:
        await resolve_one(judge=judge, vocab=Vocab(), qid="risk.pick", domain=domain, threshold=0.8, verify_qid="risk.pick")
        raise AssertionError("expected TypeError")
    except TypeError as error:
        assert "must be a Noul question" in str(error)


async def test_resolve_one_gates_on_the_stored_threshold(tmp_path):
    store = CalibrationStore(root=tmp_path)
    store.set("kind.pick|confidence", 0.97, engine="scripted", model_revision="scripted", default=0.8, n=25, precision=1.0)
    judge = ScriptedJudge([{"kind.pick": Answer.from_choice("kind.pick", "toggle_service", {"toggle_service": 0.9}, confidence=0.9)}])
    domain = DeviceAdapter()

    result = await resolve_one(judge=judge, vocab=Vocab(), qid="kind.pick", domain=domain, threshold=0.5, store=store)

    # 0.9 clears the passed threshold (0.5) but not the stored one (0.97)
    assert result.status == "needs_approval"
    assert domain.acted_on is None


async def test_resolve_one_falls_back_to_threshold_with_no_stored_entry(tmp_path):
    store = CalibrationStore(root=tmp_path)
    judge = ScriptedJudge([{"kind.pick": Answer.from_choice("kind.pick", "toggle_service", {"toggle_service": 0.9}, confidence=0.9)}])
    domain = DeviceAdapter()

    result = await resolve_one(judge=judge, vocab=Vocab(), qid="kind.pick", domain=domain, threshold=0.8, store=store)

    assert result.status == "verified"
    assert domain.acted_on == "toggle_service"


async def test_resolve_one_keys_the_store_lookup_on_model_revision(tmp_path):
    store = CalibrationStore(root=tmp_path)
    store.set("kind.pick|confidence", 0.97, engine="scripted", model_revision="a-different-revision", default=0.8, n=25, precision=1.0)
    judge = ScriptedJudge([{"kind.pick": Answer.from_choice("kind.pick", "toggle_service", {"toggle_service": 0.9}, confidence=0.9)}])
    domain = DeviceAdapter()

    result = await resolve_one(judge=judge, vocab=Vocab(), qid="kind.pick", domain=domain, threshold=0.8, store=store)

    # ScriptedJudge answers under model_revision="scripted", not "a-different-revision" -- the stored
    # entry isn't read, so the passed threshold (0.8) applies and 0.9 clears it.
    assert result.status == "verified"
    assert domain.acted_on == "toggle_service"


async def test_circuit_gate_extra_publishes_when_the_circuit_decides_true():
    domain = ClaimAdapter()
    judge = ScriptedJudge([{
        "risk.pick": Answer.from_choice("risk.pick", "high", {"high": 0.95}, confidence=0.95),
        "verify.grounded": Answer.from_noul("verify.grounded", 0.9),
    }])
    gates = {
        "policy": GateSpec(op="threshold", input="policy_ok", tau=0.5, band=0.0),
        "decision": GateSpec(op="and", inputs=["verify", "policy"], tau=0.5, band=0.0),
    }
    extra_gate = circuit_gate_extra(
        gates, key="decision", kwarg="grounded",
        extra=lambda facts: {"policy_ok": Answer.from_noul("policy_ok", 0.9)},
    )

    result = await resolve_one(
        judge=judge, vocab=Vocab(), qid="risk.pick", domain=domain, threshold=0.8,
        gate=claim_gate, verify_qid="verify.grounded", gate_extra=extra_gate,
    )

    assert result.gate_result.verdict == "publish"
    assert domain.acted_on == "high"


async def test_circuit_gate_extra_withholds_when_the_circuit_is_uncertain():
    domain = ClaimAdapter()
    judge = ScriptedJudge([{
        "risk.pick": Answer.from_choice("risk.pick", "high", {"high": 0.95}, confidence=0.95),
        "verify.grounded": Answer.from_noul("verify.grounded", 0.51),
    }])
    gates = {"decision": GateSpec(op="threshold", input="verify", tau=0.5, band=0.1)}
    extra_gate = circuit_gate_extra(gates, key="decision", kwarg="grounded")

    result = await resolve_one(
        judge=judge, vocab=Vocab(), qid="risk.pick", domain=domain, threshold=0.8,
        gate=claim_gate, verify_qid="verify.grounded", gate_extra=extra_gate,
    )

    assert result.status == "withhold"
    assert domain.acted_on is None


async def test_circuit_gate_extra_works_with_mutation_gate_too():
    domain = DeviceAdapter()
    judge = ScriptedJudge([{
        "kind.pick": Answer.from_choice("kind.pick", "toggle_service", {"toggle_service": 0.95}, confidence=0.95),
        "verify.safe": Answer.from_noul("verify.safe", 0.1),
    }])
    gates = {"decision": GateSpec(op="not", input="verify", tau=0.5, band=0.0)}
    extra_gate = circuit_gate_extra(gates, key="decision", kwarg="denied")

    result = await resolve_one(
        judge=judge, vocab=Vocab(), qid="kind.pick", domain=domain, threshold=0.8,
        gate=mutation_gate, verify_qid="verify.safe", gate_extra=extra_gate,
    )

    assert result.status == "deny"
    assert domain.acted_on is None


async def test_ask_batch_journals_a_heterogeneous_batch_under_one_call_id(tmp_path):
    journal = Journal(root=tmp_path)
    judge = ScriptedJudge([{
        "archetype": Answer.from_choice("archetype", "library", {"library": 0.9, "app": 0.1}, confidence=0.9),
        "drift": Answer.from_noul("drift", 0.8),
        "pollution": Answer.from_score("pollution", 0.0, {"0": "none"}, {"0": 1.0}, confidence=0.7),
    }])
    facts = Facts(state={"repo": "acme"})

    call_id, result = await ask_batch(judge=judge, questions={
        "archetype": Choice(instructions="what archetype?", criteria={"library": "a library", "app": "an app"}),
        "drift": Noul(instructions="has effort drifted?"),
        "pollution": Score(instructions="how polluted?", criteria=["none"]),
    }, facts=facts, journal=journal, phase="route")

    assert set(result.answers) == {"archetype", "drift", "pollution"}
    journal.flush()
    row = next(journal.replay())
    assert row["type"] == "decision"
    assert row["call_id"] == call_id
    assert row["phase"] == "route"
    assert set(row["answers"]) == {"archetype", "drift", "pollution"}


async def test_ask_batch_without_a_journal_does_not_raise():
    judge = ScriptedJudge([{"q": Answer.from_noul("q", 0.9)}])
    call_id, result = await ask_batch(
        judge=judge, questions={"q": Noul(instructions="?")}, facts=Facts(state={}),
    )
    assert result.answers["q"].noul == 0.9
    assert call_id


async def test_ask_batch_composes_with_evaluate_gates_for_a_routing_style_decision():
    """The shape ask_batch() exists for: many heterogeneous questions about
    one observation, each independently gated, with ordinary code routing
    on the combination -- no single candidate, no act()/verify()."""
    judge = ScriptedJudge([{
        "use_recent": Answer.from_noul("use_recent", 0.9),
        "gen_kind": Answer.from_choice("gen_kind", "generated", {"generated": 0.85, "hand_written": 0.15}, confidence=0.85),
    }])
    facts = Facts(state={"repo": "acme"})

    call_id, result = await ask_batch(judge=judge, questions={
        "use_recent": Noul(instructions="prioritize the recent window?"),
        "gen_kind": Choice(instructions="what kind of file?", criteria={"generated": "generated", "hand_written": "hand-written"}),
    }, facts=facts)

    gates = {
        "use_recent": GateSpec(op="threshold", input="use_recent", tau=0.5, band=0.0),
        "drop_generated": GateSpec(op="confidence", input="gen_kind", min_confidence=0.8, band=0.0),
    }
    gated = evaluate_gates(gates, result.answers)

    use_recent = result_key(gated["use_recent"]) is True
    drop_generated = result_key(gated["drop_generated"]) == "generated"

    assert call_id
    assert use_recent is True
    assert drop_generated is True


def _decide_vocab():
    return FrozenVocabulary("v1", {
        "pick": {"type": "choice", "instructions": "pick", "criteria": {}},
        "check": {"type": "noul", "instructions": "check?"},
    })


async def test_decide_act_and_verify_journal_key_episode_and_default_tests(tmp_path):
    journal = Journal(root=tmp_path)
    judge = ScriptedJudge([{
        "pick": Answer.from_choice("pick", "a", {"a": 0.9}, confidence=0.9),
        "check": Answer.from_noul("check", 0.8),
    }])
    specs = {"pick": ("pick", {}, None, {"a": "opt a"}), "check": ("check", {}, None, None)}

    decision = await decide(judge=judge, vocab=_decide_vocab(), specs=specs, facts=Facts(state={}), journal=journal, scope={"episode_id": "ep1"})
    decision.act("pick", ActOutcome(succeeded=True, detail={}), None)
    decision.verify(Verdict(status="verified"))

    rows = list(journal.replay())
    outcome_row = next(r for r in rows if r["type"] == "outcome")
    assert outcome_row["key"] == "pick"
    assert outcome_row["episode_id"] == "ep1"
    verdict_row = next(r for r in rows if r["type"] == "verdict")
    assert verdict_row["tests"] == ["pick"]  # defaulted to every acted key, "check" was never acted


async def test_batch_decision_circuit_resolves_tau_from_the_stored_threshold(tmp_path):
    store = CalibrationStore(root=tmp_path)
    store.set("check|noul_p", 0.9, engine="scripted", model_revision="scripted", default=0.5, n=20, precision=1.0)
    judge = ScriptedJudge([{"pick": Answer.from_choice("pick", "a", {"a": 0.9}, confidence=0.9), "check": Answer.from_noul("check", 0.8)}])
    specs = {"pick": ("pick", {}, None, {"a": "opt a"}), "check": ("check", {}, None, None)}

    decision = await decide(judge=judge, vocab=_decide_vocab(), specs=specs, facts=Facts(state={}), store=store)
    results = await decision.circuit({"check": GateSpec(op="threshold", input="check", tau=0.5, band=0.0)})

    # 0.8 clears the spec's own tau (0.5) but not the stored one (0.9) circuit() resolved instead
    assert result_key(results["check"]) is False


def test_episode_outcome_journals_a_code_policy_act_with_no_gate(tmp_path):
    journal = Journal(root=tmp_path)
    episode = Episode(journal, scope={"goal": "reorganize"})

    episode.outcome(ActOutcome(succeeded=True, detail={}), key="drop_generated")

    row = next(r for r in journal.replay() if r["type"] == "outcome")
    assert row["gate_verdict"] is None
    assert row["episode_id"] == episode.episode_id
    assert row["key"] == "drop_generated"


class FailingJudge:
    name = "failing"

    async def ask_all(self, state, questions):
        raise RuntimeError("judge unreachable")


async def test_ask_batch_journals_an_error_row_and_reraises(tmp_path):
    journal = Journal(root=tmp_path)

    with pytest.raises(RuntimeError, match="judge unreachable"):
        await ask_batch(judge=FailingJudge(), questions={"q": Noul(instructions="?")}, facts=Facts(state={}), journal=journal)

    row = next(journal.replay())
    assert row["type"] == "decision"
    assert row["answers"] == {}
    assert row["error"] == "judge unreachable"


async def test_ask_batch_never_sends_scope_to_the_judge():
    judge = ScriptedJudge([{"q": Answer.from_noul("q", 0.9)}])

    await ask_batch(judge=judge, questions={"q": Noul(instructions="?")}, facts=Facts(state={"repo": "acme"}), scope={"goal": "secret goal"})

    state_sent_to_judge = judge.calls[0][0]
    assert "goal" not in state_sent_to_judge


async def test_ask_batch_capture_writes_state_and_asked_omits_without(tmp_path):
    journal = Journal(root=tmp_path)
    judge = ScriptedJudge([
        {"q": Answer.from_noul("q", 0.9)},
        {"q": Answer.from_noul("q", 0.8)},
    ])
    facts = Facts(state={"repo": "acme"})

    await ask_batch(
        judge=judge, questions={"q": Noul(instructions="?")}, facts=facts, journal=journal, capture=True,
    )

    journal.flush()
    row = next(journal.replay())
    assert row["state"] == {"repo": "acme"}
    assert "q" in row["asked"]
    assert row["asked"]["q"]["type"] == "noul"

    journal.root.joinpath("journal.jsonl").unlink()
    journal._labels = LabelIndex()

    await ask_batch(
        judge=judge, questions={"q": Noul(instructions="?")}, facts=facts, journal=journal, capture=False,
    )

    journal.flush()
    row = next(iter(journal.replay()))
    assert "state" not in row
    assert "asked" not in row


async def test_ask_batch_extra_merges_with_usage(tmp_path):
    from typesymbolic.judge import AskResult
    journal = Journal(root=tmp_path)
    judge = ScriptedJudge([
        AskResult(
            answers={"q": Answer.from_noul("q", 0.9)},
            model_revision="scripted",
            usage={"input_tokens": 10, "output_tokens": 5},
        ),
    ])

    await ask_batch(
        judge=judge,
        questions={"q": Noul(instructions="?")},
        facts=Facts(state={}),
        journal=journal,
        extra={"custom": "value"},
    )

    journal.flush()
    row = next(journal.replay())
    assert row["extra"]["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert row["extra"]["custom"] == "value"


async def test_ask_batch_uses_supplied_call_id_and_decide_forwards_it(tmp_path):
    journal = Journal(root=tmp_path)
    judge = ScriptedJudge([
        {"q": Answer.from_noul("q", 0.9)},
        {"pick": Answer.from_choice("pick", "a", {"a": 0.9}, confidence=0.9)},
    ])
    supplied_id = "my-call-id-123"

    call_id, _ = await ask_batch(
        judge=judge,
        questions={"q": Noul(instructions="?")},
        facts=Facts(state={}),
        journal=journal,
        call_id=supplied_id,
    )

    assert call_id == supplied_id
    journal.flush()
    row = next(journal.replay())
    assert row["call_id"] == supplied_id

    journal.root.joinpath("journal.jsonl").unlink()
    journal._labels = LabelIndex()

    decision = await decide(
        judge=judge,
        vocab=_decide_vocab(),
        specs={"pick": ("pick", {}, None, {"a": "opt a"})},
        facts=Facts(state={}),
        journal=journal,
        call_id=supplied_id,
    )

    assert decision.call_id == supplied_id
    journal.flush()
    row = next(journal.replay())
    assert row["call_id"] == supplied_id
