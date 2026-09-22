import pytest

from typesymbolic.domain import ActOutcome, Facts, Verdict
from typesymbolic.engine import resolve_one
from typesymbolic.judge import ScriptedJudge
from typesymbolic.question import Answer, Choice, Noul, Score
from typesymbolic.vocab import FrozenVocabulary, VocabularyError


def test_noul_entry_builds_a_noul():
    vocab = FrozenVocabulary("v1", {"safe": {"type": "noul", "instructions": "is {thing} safe?"}})
    question = vocab.ask("safe", thing="this command")
    assert isinstance(question, Noul)
    assert question.instructions == "is this command safe?"


def test_choice_entry_builds_a_choice_with_criteria():
    vocab = FrozenVocabulary("v1", {
        "kind": {"type": "choice", "instructions": "pick a kind", "criteria": {"a": "option a", "b": "option b"}},
    })
    question = vocab.ask("kind")
    assert isinstance(question, Choice)
    assert question.criteria == {"a": "option a", "b": "option b"}


def test_score_entry_builds_a_score_with_ordered_criteria():
    vocab = FrozenVocabulary("v1", {
        "risk": {"type": "score", "instructions": "rate risk", "criteria": ["low", "medium", "high"]},
    })
    question = vocab.ask("risk")
    assert isinstance(question, Score)
    assert question.criteria == ["low", "medium", "high"]


def test_repeated_slot_fills_both_occurrences():
    vocab = FrozenVocabulary("v1", {"q": {"type": "noul", "instructions": "{name} then {name} again"}})
    question = vocab.ask("q", name="x")
    assert question.instructions == "x then x again"


def test_extra_unused_slots_are_tolerated():
    vocab = FrozenVocabulary("v1", {"q": {"type": "noul", "instructions": "hello {name}"}})
    question = vocab.ask("q", name="a", unused="b")
    assert question.instructions == "hello a"


def test_unknown_qid_raises_vocabulary_error_listing_known_ids():
    vocab = FrozenVocabulary("v1", {"a": {"type": "noul", "instructions": "?"}, "b": {"type": "noul", "instructions": "?"}})
    with pytest.raises(VocabularyError) as excinfo:
        vocab.ask("c")
    assert "a" in str(excinfo.value) and "b" in str(excinfo.value)


def test_missing_slot_raises_vocabulary_error_naming_it():
    vocab = FrozenVocabulary("v1", {"q": {"type": "noul", "instructions": "is {thing} safe?"}})
    with pytest.raises(VocabularyError) as excinfo:
        vocab.ask("q")
    assert "thing" in str(excinfo.value)


@pytest.mark.parametrize("entries", [
    {"q": {"type": "xor", "instructions": "?"}},
    {"q": {"type": "noul", "instructions": ""}},
    {"q": {"type": "noul"}},
    {"q": {"type": "noul", "instructions": "?", "criteria": {"true": "y"}}},
    {"q": {"type": "choice", "instructions": "?", "criteria": {"a": 1}}},
    {"q": {"type": "score", "instructions": "?", "criteria": []}},
    {"q": {"type": "score", "instructions": "?", "criteria": [1, 2]}},
    {"q": {"type": "noul", "instructions": "{}"}},
    {"q": {"type": "noul", "instructions": "{0}"}},
    {"q": {"type": "noul", "instructions": "{"}},
])
def test_construction_rejects_malformed_entries(entries):
    with pytest.raises(VocabularyError):
        FrozenVocabulary("v1", entries)


def test_construction_rejects_empty_entries():
    with pytest.raises(VocabularyError):
        FrozenVocabulary("v1", {})


def test_construction_rejects_empty_version():
    with pytest.raises(VocabularyError):
        FrozenVocabulary("", {"q": {"type": "noul", "instructions": "?"}})


def test_mutating_a_returned_choice_criteria_does_not_affect_later_asks():
    vocab = FrozenVocabulary("v1", {"kind": {"type": "choice", "instructions": "?", "criteria": {"a": "A"}}})
    question = vocab.ask("kind")
    question.criteria["a"] = "tampered"
    again = vocab.ask("kind")
    assert again.criteria == {"a": "A"}


class DeviceAdapter:
    async def observe(self) -> Facts:
        return Facts(state={"goal": "toggle wifi"})

    def propose(self, facts: Facts) -> dict[str, str]:
        return {"toggle_service": "turns a radio on/off"}

    async def act(self, chosen_id: str, facts: Facts) -> ActOutcome:
        return ActOutcome(succeeded=True, detail={"exit_code": 0})

    async def verify(self, outcome: ActOutcome, facts_before: Facts) -> Verdict:
        return Verdict(status="verified")


async def test_resolve_one_overwrites_declared_choice_criteria_with_live_candidates():
    vocab = FrozenVocabulary("v1", {
        "kind.pick": {"type": "choice", "instructions": "pick an action kind", "criteria": {"placeholder": "should never be sent"}},
    })
    judge = ScriptedJudge([{"kind.pick": Answer.from_choice("kind.pick", "toggle_service", {"toggle_service": 0.95}, confidence=0.95)}])
    domain = DeviceAdapter()

    result = await resolve_one(judge=judge, vocab=vocab, qid="kind.pick", domain=domain, threshold=0.8)

    assert result.status == "verified"
    _sent_state, sent_questions = judge.calls[0]
    assert sent_questions["kind.pick"].criteria == {"toggle_service": "turns a radio on/off"}
