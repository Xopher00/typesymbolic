import pytest

from typesymbolic.question import Answer


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
