"""Exercises DomainAdapter end to end with two fakes covering the two shapes
act()/verify() need to support: a real-world mutation (run a command, check
its exit code) and a claim published into a report (render a finding, check
it against already-known facts)."""

from typesymbolic.domain import ActOutcome, ActStep, Facts, Verdict


class FakeDeviceAdapter:
    """Mutation-style: propose() offers action kinds, act() runs a command,
    verify() checks the device's own post-action exit code."""

    def __init__(self, exit_code: int = 0) -> None:
        self._exit_code = exit_code

    async def observe(self) -> Facts:
        return Facts(state={"goal": "turn on wifi", "action_options": {"toggle_service": "turns a radio on/off"}})

    def propose(self, facts: Facts) -> dict[str, str]:
        return facts.state["action_options"]

    async def act(self, chosen_id: str, facts: Facts) -> ActOutcome:
        return ActOutcome(succeeded=self._exit_code == 0, detail={"executed_command": "svc wifi enable", "exit_code": self._exit_code})

    async def verify(self, outcome: ActOutcome, facts_before: Facts) -> Verdict:
        if outcome.detail.get("exit_code") == 0:
            return Verdict(status="verified")
        return Verdict(status="failed", reasons=(f"exit_code={outcome.detail.get('exit_code')}",))


class FakeClaimAdapter:
    """Claim/publish-style: propose() offers frozen risk categories, act()
    "publishes" a claim into a report instead of mutating anything, verify()
    runs a same-run structural contradiction check against facts_before's
    evidence."""

    async def observe(self) -> Facts:
        return Facts(
            state={"path": "core/auth.py", "commits": 40},
            evidence={"dependent_files": 0, "bugfix_share": "0%"},
        )

    def propose(self, facts: Facts) -> dict[str, str]:
        return {"low": "low risk", "medium": "medium risk", "high": "near-critical risk"}

    async def act(self, chosen_id: str, facts: Facts) -> ActOutcome:
        return ActOutcome(succeeded=True, detail={"claim": f"{facts.state['path']}: {chosen_id} risk", "kind": chosen_id})

    async def verify(self, outcome: ActOutcome, facts_before: Facts) -> Verdict:
        if outcome.detail.get("kind") == "high" and facts_before.evidence.get("dependent_files") == 0:
            return Verdict(status="failed", reasons=("near-critical risk claimed but no dependents and no bugfix history",))
        return Verdict(status="verified")


async def test_device_adapter_resolves_and_verifies_ok():
    adapter = FakeDeviceAdapter(exit_code=0)
    facts = await adapter.observe()
    candidates = adapter.propose(facts)
    assert candidates == {"toggle_service": "turns a radio on/off"}

    outcome = await adapter.act("toggle_service", facts)
    assert outcome.succeeded

    verdict = await adapter.verify(outcome, facts)
    assert verdict.status == "verified"


async def test_device_adapter_surfaces_a_failed_command():
    adapter = FakeDeviceAdapter(exit_code=1)
    facts = await adapter.observe()
    outcome = await adapter.act("toggle_service", facts)
    assert not outcome.succeeded

    verdict = await adapter.verify(outcome, facts)
    assert verdict.status == "failed"
    assert "exit_code=1" in verdict.reasons[0]


async def test_claim_adapter_publish_and_verify_ok():
    adapter = FakeClaimAdapter()
    facts = await adapter.observe()
    candidates = adapter.propose(facts)
    assert "high" in candidates

    outcome = await adapter.act("medium", facts)
    assert outcome.succeeded
    assert outcome.detail["kind"] == "medium"

    verdict = await adapter.verify(outcome, facts)
    assert verdict.status == "verified"


async def test_claim_adapter_verify_catches_ungrounded_high_risk_claim():
    """A claim inconsistent with already-known structural facts
    (facts_before.evidence) is caught without any new I/O or model call —
    the "immediate check" reading of verify()."""
    adapter = FakeClaimAdapter()
    facts = await adapter.observe()

    outcome = await adapter.act("high", facts)
    verdict = await adapter.verify(outcome, facts)

    assert verdict.status == "failed"
    assert verdict.reasons


def test_act_outcome_carries_key_and_steps():
    outcome = ActOutcome(
        succeeded=True, key="pick",
        steps=(ActStep(name="drop", succeeded=True), ActStep(name="rank", succeeded=None, detail={"n": 3})),
    )
    assert outcome.key == "pick"
    assert [s.name for s in outcome.steps] == ["drop", "rank"]
    assert outcome.steps[1].succeeded is None
    assert outcome.steps[1].detail == {"n": 3}


def test_verdict_calibrate_defaults_true_and_can_be_disabled():
    assert Verdict(status="verified").calibrate is True
    assert Verdict(status="failed", calibrate=False).calibrate is False
