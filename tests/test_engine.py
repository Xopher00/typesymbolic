from typesymbolic.calibration_store import CalibrationStore
from typesymbolic.circuit import GateSpec
from typesymbolic.domain import ActOutcome, Facts, Verdict
from typesymbolic.engine import circuit_gate_extra, resolve_one
from typesymbolic.gate import claim_gate, mutation_gate
from typesymbolic.journal import Journal
from typesymbolic.judge import ScriptedJudge
from typesymbolic.question import Answer, Choice, Noul


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
    assert [r["type"] for r in rows] == ["decision", "outcome"]


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
    store.set("kind.pick", 0.97, engine="scripted", model_revision="scripted", default=0.8, n=25, precision=1.0)
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
    store.set("kind.pick", 0.97, engine="scripted", model_revision="a-different-revision", default=0.8, n=25, precision=1.0)
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
