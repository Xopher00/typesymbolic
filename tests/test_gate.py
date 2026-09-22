from typesymbolic.gate import GateVerdict, claim_gate, mutation_gate


def test_mutation_gate_denies_regardless_of_confidence():
    result = mutation_gate(confidence=0.99, threshold=0.8, denied=True)
    assert result.verdict == GateVerdict.DENY
    assert result.reason == "deny_listed"


def test_mutation_gate_acts_above_threshold():
    result = mutation_gate(confidence=0.9, threshold=0.8)
    assert result.verdict == GateVerdict.ACT


def test_mutation_gate_needs_approval_below_threshold():
    result = mutation_gate(confidence=0.5, threshold=0.8)
    assert result.verdict == GateVerdict.NEEDS_APPROVAL


def test_mutation_gate_boundary_confidence_acts():
    result = mutation_gate(confidence=0.8, threshold=0.8)
    assert result.verdict == GateVerdict.ACT


def test_mutation_gate_carries_call_id_for_journal_linkage():
    result = mutation_gate(confidence=0.9, threshold=0.8, call_id="abc-123")
    assert result.call_id == "abc-123"


def test_claim_gate_withholds_ungrounded_claim_even_at_high_confidence():
    result = claim_gate(confidence=0.99, threshold=0.8, grounded=False)
    assert result.verdict == GateVerdict.WITHHOLD
    assert result.reason == "ungrounded"


def test_claim_gate_publishes_above_threshold():
    result = claim_gate(confidence=0.9, threshold=0.8)
    assert result.verdict == GateVerdict.PUBLISH


def test_claim_gate_hedges_below_threshold():
    result = claim_gate(confidence=0.5, threshold=0.8)
    assert result.verdict == GateVerdict.HEDGE


def test_claim_gate_publishes_with_no_confidence_to_gate():
    result = claim_gate(confidence=None, threshold=0.8)
    assert result.verdict == GateVerdict.PUBLISH
    assert result.reason == "no confidence to gate"


def test_mutation_gate_band_flags_a_pass_right_at_the_threshold_as_uncertain():
    result = mutation_gate(confidence=0.81, threshold=0.8, band=0.05)
    assert result.verdict == GateVerdict.NEEDS_APPROVAL
    assert "within" in result.reason


def test_mutation_gate_band_still_acts_comfortably_above_threshold():
    result = mutation_gate(confidence=0.95, threshold=0.8, band=0.05)
    assert result.verdict == GateVerdict.ACT


def test_mutation_gate_zero_band_is_the_old_behavior():
    result = mutation_gate(confidence=0.81, threshold=0.8, band=0.0)
    assert result.verdict == GateVerdict.ACT


def test_claim_gate_band_flags_a_borderline_hedge_boundary_as_uncertain():
    result = claim_gate(confidence=0.79, threshold=0.8, band=0.05)
    assert result.verdict == GateVerdict.HEDGE
    assert "within" in result.reason


def test_mutation_gate_on_uncertain_overrides_the_default_escalation():
    result = mutation_gate(confidence=0.5, threshold=0.8, on_uncertain=GateVerdict.DENY)
    assert result.verdict == GateVerdict.DENY


def test_claim_gate_on_uncertain_overrides_the_default_escalation():
    result = claim_gate(confidence=0.5, threshold=0.8, on_uncertain=GateVerdict.WITHHOLD)
    assert result.verdict == GateVerdict.WITHHOLD


def test_on_uncertain_none_keeps_old_behavior():
    result = mutation_gate(confidence=0.5, threshold=0.8, on_uncertain=None)
    assert result.verdict == GateVerdict.NEEDS_APPROVAL
