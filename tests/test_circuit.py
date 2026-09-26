import pytest

from typesymbolic.circuit import CircuitError, GateSpec, evaluate_gates, result_key
from typesymbolic.question import Answer


def test_threshold_passes_above_tau():
    answers = {"safe": Answer.from_noul("safe", 0.9)}
    gates = {"g": GateSpec(op="threshold", input="safe", tau=0.8, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].value is True
    assert results["g"].outcome == "decided"


def test_threshold_uncertain_within_band_abstains_by_default():
    answers = {"safe": Answer.from_noul("safe", 0.81)}
    gates = {"g": GateSpec(op="threshold", input="safe", tau=0.8, band=0.05)}
    results = evaluate_gates(gates, answers)
    assert results["g"].uncertain is True
    assert results["g"].outcome == "abstain"
    assert results["g"].value is None


def test_threshold_uncertain_with_default_policy_falls_back():
    answers = {"safe": Answer.from_noul("safe", 0.81)}
    gates = {"g": GateSpec(op="threshold", input="safe", tau=0.8, band=0.05, on_uncertain="default", default=False)}
    results = evaluate_gates(gates, answers)
    assert results["g"].outcome == "default"
    assert results["g"].value is False


def test_not_inverts_probability():
    answers = {"risky": Answer.from_noul("risky", 0.9)}
    gates = {"g": GateSpec(op="not", input="risky", tau=0.5, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].p == pytest.approx(0.1)
    assert results["g"].value is False


def test_and_multiplies_independent_probabilities():
    answers = {"a": Answer.from_noul("a", 0.9), "b": Answer.from_noul("b", 0.9)}
    gates = {"g": GateSpec(op="and", inputs=["a", "b"], tau=0.5, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].p == pytest.approx(0.81)
    assert results["g"].value is True


def test_or_combines_independent_probabilities():
    answers = {"a": Answer.from_noul("a", 0.5), "b": Answer.from_noul("b", 0.5)}
    gates = {"g": GateSpec(op="or", inputs=["a", "b"], tau=0.5, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].p == pytest.approx(0.75)


def test_weak_and_takes_the_minimum():
    answers = {"a": Answer.from_noul("a", 0.5), "b": Answer.from_noul("b", 0.5)}
    gates = {"g": GateSpec(op="and", inputs=["a", "b"], combine="weak", tau=0.4, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].p == pytest.approx(0.5)  # product would give 0.25, strong 0.0
    assert results["g"].value is True


def test_weak_and_is_idempotent_over_a_repeated_input():
    answers = {"a": Answer.from_noul("a", 0.6)}
    gates = {"g": GateSpec(op="and", inputs=["a", "a", "a"], combine="weak", tau=0.5, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].p == pytest.approx(0.6)  # product would decay to 0.216


def test_weak_or_takes_the_maximum():
    answers = {"a": Answer.from_noul("a", 0.5), "b": Answer.from_noul("b", 0.5)}
    gates = {"g": GateSpec(op="or", inputs=["a", "b"], combine="weak", tau=0.4, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].p == pytest.approx(0.5)  # product would give 0.75


def test_strong_and_collapses_when_the_inputs_do_not_jointly_saturate():
    answers = {"a": Answer.from_noul("a", 0.5), "b": Answer.from_noul("b", 0.5)}
    gates = {"g": GateSpec(op="and", inputs=["a", "b"], combine="strong", tau=0.4, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].p == pytest.approx(0.0)  # max(0, 1.0 - 1) -- the documented failure mode
    assert results["g"].value is False


def test_strong_and_survives_when_every_input_is_high():
    answers = {"a": Answer.from_noul("a", 0.9), "b": Answer.from_noul("b", 0.8)}
    gates = {"g": GateSpec(op="and", inputs=["a", "b"], combine="strong", tau=0.5, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].p == pytest.approx(0.7)  # max(0, 1.7 - 1)


def test_strong_or_saturates_at_one():
    answers = {"a": Answer.from_noul("a", 0.6), "b": Answer.from_noul("b", 0.7)}
    gates = {"g": GateSpec(op="or", inputs=["a", "b"], combine="strong", tau=0.5, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].p == pytest.approx(1.0)  # min(1, 1.3)


def test_product_stays_the_default_combine():
    answers = {"a": Answer.from_noul("a", 0.5), "b": Answer.from_noul("b", 0.5)}
    gates = {"g": GateSpec(op="and", inputs=["a", "b"], tau=0.4, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert gates["g"].combine == "product"
    assert results["g"].p == pytest.approx(0.25)


def test_trace_does_not_claim_independence_for_weak():
    answers = {"a": Answer.from_noul("a", 0.5), "b": Answer.from_noul("b", 0.5)}
    gates = {"g": GateSpec(op="and", inputs=["a", "b"], combine="weak")}
    results = evaluate_gates(gates, answers)
    assert not any("independence" in line for line in results["g"].trace)


def test_majority_picks_the_plurality_choice():
    answers = {
        "v1": Answer.from_choice("v1", "yes", {"yes": 0.9, "no": 0.1}),
        "v2": Answer.from_choice("v2", "yes", {"yes": 0.8, "no": 0.2}),
        "v3": Answer.from_choice("v3", "no", {"yes": 0.3, "no": 0.7}),
    }
    gates = {"g": GateSpec(op="majority", inputs=["v1", "v2", "v3"])}
    results = evaluate_gates(gates, answers)
    assert results["g"].value == "yes"
    assert results["g"].outcome == "decided"


def test_argmax_abstains_below_min_confidence():
    answers = {"pick": Answer.from_choice("pick", "billing", {"billing": 0.55, "technical": 0.45}, confidence=0.55)}
    gates = {"g": GateSpec(op="argmax", input="pick", min_confidence=0.8)}
    results = evaluate_gates(gates, answers)
    assert results["g"].uncertain is True
    assert results["g"].outcome == "abstain"


def test_verify_escalates_when_check_is_unsupported():
    answers = {
        "claim": Answer.from_choice("claim", "high", {"high": 0.95}, confidence=0.95),
        "supported": Answer.from_noul("supported", 0.1),
    }
    gates = {"g": GateSpec(op="verify", input="claim", check="supported", tau=0.5, on_uncertain="escalate")}
    results = evaluate_gates(gates, answers)
    assert results["g"].outcome == "escalate"


def test_verify_decides_when_check_is_supported():
    answers = {
        "claim": Answer.from_choice("claim", "high", {"high": 0.95}, confidence=0.95),
        "supported": Answer.from_noul("supported", 0.9),
    }
    gates = {"g": GateSpec(op="verify", input="claim", check="supported", tau=0.5)}
    results = evaluate_gates(gates, answers)
    assert results["g"].outcome == "decided"
    assert results["g"].value == "high"


def test_order_buckets_a_score_by_cutpoints():
    answers = {"risk": Answer.from_score("risk", 1.9, {"0": "low", "1": "medium", "2": "high"}, {"0": 0.05, "1": 0.15, "2": 0.8}, confidence=0.8)}
    gates = {"g": GateSpec(op="order", input="risk", cutpoints=[1.0, 2.0], band=0.05)}
    results = evaluate_gates(gates, answers)
    assert results["g"].value == 1


def test_gate_can_reference_an_earlier_gate():
    answers = {"a": Answer.from_noul("a", 0.9)}
    gates = {
        "first": GateSpec(op="threshold", input="a", tau=0.5),
        "second": GateSpec(op="not", input="first:True", tau=0.5),
    }
    results = evaluate_gates(gates, answers)
    assert results["second"].value is False  # not(True) -- second gate consumed the first's decided value


def test_result_key_routes_on_value_when_decided():
    answers = {"a": Answer.from_noul("a", 0.9)}
    results = evaluate_gates({"g": GateSpec(op="threshold", input="a", tau=0.5)}, answers)
    assert result_key(results["g"]) is True


def test_result_key_routes_on_outcome_when_uncertain():
    answers = {"a": Answer.from_noul("a", 0.51)}
    results = evaluate_gates({"g": GateSpec(op="threshold", input="a", tau=0.5, band=0.1)}, answers)
    assert result_key(results["g"]) == "abstain"


def test_gate_spec_rejects_unknown_op():
    with pytest.raises(CircuitError):
        GateSpec(op="xor", input="a")


def test_gate_spec_requires_input_for_threshold():
    with pytest.raises(CircuitError):
        GateSpec(op="threshold")


def test_gate_spec_requires_cutpoints_for_order():
    with pytest.raises(CircuitError):
        GateSpec(op="order", input="a")


def test_gate_spec_rejects_unknown_combine():
    with pytest.raises(CircuitError):
        GateSpec(op="and", inputs=["a", "b"], combine="drastic")


def test_gate_spec_rejects_combine_on_a_non_boolean_op():
    with pytest.raises(CircuitError):
        GateSpec(op="threshold", input="a", combine="weak")


def test_confidence_abstains_on_a_noul_with_no_native_confidence():
    answers = {"safe": Answer.from_noul("safe", 0.05)}
    gates = {"g": GateSpec(op="confidence", input="safe", min_confidence=0.5, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].uncertain is True
    assert results["g"].outcome == "abstain"


def test_confidence_abstains_below_min_confidence():
    answers = {"pick": Answer.from_choice("pick", "billing", {"billing": 0.55, "technical": 0.45}, confidence=0.55)}
    gates = {"g": GateSpec(op="confidence", input="pick", min_confidence=0.8, band=0.0)}
    results = evaluate_gates(gates, answers)
    assert results["g"].uncertain is True
    assert results["g"].outcome == "abstain"


def test_confidence_surfaces_a_score_answers_own_value():
    answers = {"risk": Answer.from_score("risk", 1.9, {"0": "low", "1": "medium", "2": "high"}, {"0": 0.05, "1": 0.15, "2": 0.8}, confidence=0.8)}
    gates = {"g": GateSpec(op="confidence", input="risk", min_confidence=0.5)}
    results = evaluate_gates(gates, answers)
    assert results["g"].value == 1.9
    assert results["g"].outcome == "decided"
