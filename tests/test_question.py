from typesymbolic.question import Answer


def test_answer_abstained_defaults_false():
    answer = Answer.from_noul("q", 0.9)
    assert answer.abstained is False


def test_answer_from_noul_abstained_passthrough():
    answer = Answer.from_noul("q", 0.5, abstained=True)
    assert answer.abstained is True


def test_answer_from_choice_abstained_passthrough():
    answer = Answer.from_choice("q", "opt", {"opt": 0.4}, abstained=True)
    assert answer.abstained is True


def test_answer_from_score_abstained_passthrough():
    answer = Answer.from_score("q", 1.0, {"0": "low"}, {"0": 1.0}, confidence=0.2, abstained=True)
    assert answer.abstained is True
