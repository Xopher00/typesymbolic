"""End to end: resolve_one() acts on a naively-confident pick, recalibrate()
tightens the threshold from what got journaled, and the next resolve_one()
call — reading the same store — escalates a pick it would previously have
acted on. No model calls."""

from typesymbolic.calibrate import recalibrate
from typesymbolic.calibration_store import CalibrationStore
from typesymbolic.domain import ActOutcome, Facts, Verdict
from typesymbolic.engine import resolve_one
from typesymbolic.journal import Journal
from typesymbolic.judge import ScriptedJudge
from typesymbolic.question import Answer, Choice


class FlakyDeviceAdapter:
    """A 0.6-confidence pick "fails" verification; a 0.95-confidence pick
    verifies clean — the signal a refit should learn to act on."""

    def __init__(self, confidence: float) -> None:
        self.confidence = confidence
        self.acted_on: list[str] = []

    async def observe(self) -> Facts:
        return Facts(state={"goal": "toggle wifi"})

    def propose(self, facts: Facts) -> dict[str, str]:
        return {"toggle_service": "turns a radio on/off"}

    async def act(self, chosen_id: str, facts: Facts) -> ActOutcome:
        self.acted_on.append(chosen_id)
        return ActOutcome(succeeded=True, detail={})

    async def verify(self, outcome: ActOutcome, facts_before: Facts) -> Verdict:
        return Verdict(status="verified" if self.confidence >= 0.9 else "failed")


class Vocab:
    version = "test-1"

    def ask(self, qid, **slots):
        return Choice(instructions="pick an action", criteria={})


async def _resolve_scripted(*, confidence: float, threshold: float, journal: Journal, store=None):
    domain = FlakyDeviceAdapter(confidence)
    judge = ScriptedJudge([{"kind.pick": Answer.from_choice("kind.pick", "toggle_service", {"toggle_service": confidence}, confidence=confidence)}])
    return await resolve_one(
        judge=judge, vocab=Vocab(), qid="kind.pick", domain=domain, threshold=threshold,
        journal=journal, store=store,
    )


async def test_self_calibration_tightens_and_the_next_resolve_one_uses_it(tmp_path):
    journal = Journal(root=tmp_path / "journal")
    store = CalibrationStore(root=tmp_path / "store")

    # loose starting threshold (0.5) lets every pick act; 5/20 verify failed.
    for _ in range(15):
        await _resolve_scripted(confidence=0.95, threshold=0.5, journal=journal, store=store)
    for _ in range(5):
        await _resolve_scripted(confidence=0.6, threshold=0.5, journal=journal, store=store)

    result = recalibrate(
        journal=journal, store=store, qid="kind.pick", engine="scripted", model_revision="scripted",
        default_threshold=0.5, min_precision=0.9, min_labels=15,
    )
    assert result.applied is True
    assert result.threshold > 0.6

    # same 0.6-confidence pick that acted before now escalates instead
    outcome = await _resolve_scripted(confidence=0.6, threshold=0.5, journal=journal, store=store)
    assert outcome.status == "needs_approval"
