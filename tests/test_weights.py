from typesymbolic.weights import grid_search_weights


def test_grid_search_weights_picks_the_weighting_that_predicts_outcome():
    # `a` perfectly predicts outcome; `b` is pure noise -- the best weighting should load onto `a`.
    run = [
        {"a": 1.0, "b": 0.0, "outcome": 10},
        {"a": 0.9, "b": 1.0, "outcome": 9},
        {"a": 0.1, "b": 1.0, "outcome": 1},
        {"a": 0.0, "b": 0.0, "outcome": 0},
    ]
    result = grid_search_weights([run, run, run], names=("a", "b"), step=0.2, outcome_field="outcome")
    assert result["weights"]["a"] > result["weights"]["b"]
    assert result["consistent_runs"] == 3


def test_grid_search_weights_degrades_on_no_usable_runs():
    result = grid_search_weights([[{"a": 1.0}]], names=("a", "b"), outcome_field="outcome")
    assert result["weights"] is None
    assert result["usable_runs"] == 0
