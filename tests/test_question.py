import pytest

from typesymbolic.question import (
    Answer,
    Choice,
    InvalidAnswerError,
    Noul,
    Score,
    validate_answer,
)


def test_answer_abstained_defaults_false():
    answer = Answer.from_noul("q", 0.9)
    assert answer.abstained is False


def test_noul_confidence_is_not_synthesized():
    answer = Answer.from_noul("q", 0.9)
    assert answer.confidence is None


def test_noul_p_scale_reads_the_raw_probability():
    answer = Answer.from_noul("q", 0.05)
    assert answer.value("noul_p") == 0.05
    assert answer.value("noul_not_p") == pytest.approx(0.95)
    assert answer.value("confidence") is None
    assert answer.value("margin") is None


def test_choice_scale_values():
    answer = Answer.from_choice("q", "a", {"a": 0.7, "b": 0.3}, confidence=0.7)
    assert answer.value("confidence") == 0.7
    assert answer.value("margin") == pytest.approx(0.4)
    assert answer.value("noul_p") is None


def test_margin_undefined_for_fewer_than_two_probabilities():
    answer = Answer.from_choice("q", "a", {"a": 1.0}, confidence=1.0)
    assert answer.value("margin") is None


def test_answer_from_noul_abstained_passthrough():
    answer = Answer.from_noul("q", 0.5, abstained=True)
    assert answer.abstained is True


def test_answer_from_choice_abstained_passthrough():
    answer = Answer.from_choice("q", "opt", {"opt": 0.4}, abstained=True)
    assert answer.abstained is True


def test_answer_from_score_abstained_passthrough():
    answer = Answer.from_score("q", 1.0, {"0": "low"}, {"0": 1.0}, confidence=0.2, abstained=True)
    assert answer.abstained is True



def _choice():
    return Choice(instructions="pick", criteria={"a": "A", "b": "B", "c": None})


def _score():
    return Score(instructions="rate", criteria=["low", "mid", "high"])


def test_valid_answers_of_each_type_pass():
    validate_answer(Noul(instructions="?"), Answer.from_noul("n", 0.3))
    validate_answer(_choice(), Answer.from_choice("c", "b", {"a": 0.2, "b": 0.7, "c": 0.1}))
    validate_answer(_score(), Answer.from_score("s", 1.5, {}, {"0": 0.1, "1": 0.3, "2": 0.6}, 0.5))


def test_answers_without_distributions_only_check_their_value():
    validate_answer(_choice(), Answer(qid="c", type="choice", choice="a"))
    validate_answer(_score(), Answer(qid="s", type="score", score=2.0))


@pytest.mark.parametrize(
    ("question", "answer", "message"),
    [
        (Noul(instructions="?"), Answer.from_choice("x", "a", {"a": 1.0}), "choice answer to a noul"),
        (Noul(instructions="?"), Answer(qid="x", type="noul", noul=float("nan")), "finite probability"),
        (Noul(instructions="?"), Answer(qid="x", type="noul", noul=None), "finite probability"),
        (_choice(), Answer.from_choice("x", "z", {"a": 0.5, "b": 0.5, "c": 0.0}), "unknown option"),
        (_choice(), Answer.from_choice("x", "a", {"a": 0.5, "b": 0.5}), "complete distribution"),
        (_choice(), Answer.from_choice("x", "a", {"a": 0.5, "b": 0.3, "c": 0.3}), "sum to 1"),
        (_choice(), Answer.from_choice("x", "a", {"a": 0.2, "b": 0.7, "c": 0.1}), "highest-probability"),
        (_choice(), Answer(qid="x", type="choice", choice="a", probabilities={"a": 1.5, "b": -0.5, "c": 0.0}), "complete distribution"),
        (_score(), Answer(qid="x", type="score", score=2.5), "score must be in"),
        (_score(), Answer.from_score("x", 1.98, {}, {"0": 0.02, "1": 0.1, "2": 0.88}, 0.9), "weighted mean"),
        (_score(), Answer.from_score("x", 1.0, {}, {"0": 0.5, "1": 0.5}, 0.5), "complete distribution"),
    ],
)
def test_inconsistent_answers_are_rejected(question, answer, message):
    with pytest.raises(InvalidAnswerError, match=message):
        validate_answer(question, answer)


def test_ties_may_select_either_top_option():
    validate_answer(_choice(), Answer.from_choice("x", "a", {"a": 0.45, "b": 0.45, "c": 0.1}))


def test_declared_rounding_widens_the_tolerance_by_half_a_unit_per_value():
    rounded = Answer.from_choice("x", "a", {"a": 0.34, "b": 0.33, "c": 0.32})  # sums to 0.99
    with pytest.raises(InvalidAnswerError, match="sum to 1"):
        validate_answer(_choice(), rounded)
    validate_answer(_choice(), rounded, probability_decimals=2)

    off_by_rounding = Answer.from_score("x", 1.51, {}, {"0": 0.1, "1": 0.3, "2": 0.6}, 0.5)
    with pytest.raises(InvalidAnswerError, match="weighted mean"):
        validate_answer(_score(), off_by_rounding)
    validate_answer(_score(), off_by_rounding, probability_decimals=2, score_decimals=2)


def test_rounding_decimals_outside_zero_to_fifteen_are_a_caller_error():
    with pytest.raises(ValueError, match="decimals"):
        validate_answer(Noul(instructions="?"), Answer.from_noul("n", 0.5), probability_decimals=16)
