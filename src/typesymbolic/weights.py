"""Grid-searched ranking-weight suggestion, zero model calls. Different
data shape than `calibrate.py`'s threshold fitting: per-item records with
a named outcome field, not (value, verified) pairs.
"""

from __future__ import annotations


def _grid(names: tuple[str, ...], step: float) -> list[dict[str, float]]:
    """Every weighting over `names` on a step-`step` grid that sums to 1,
    via recursion so `names` can be any length."""
    units_total = round(1 / step)

    def assign(remaining: tuple[str, ...], units_left: int):
        if len(remaining) == 1:
            yield {remaining[0]: units_left * step}
            return
        for units in range(units_left + 1):
            for rest in assign(remaining[1:], units_left - units):
                yield {remaining[0]: units * step, **rest}

    return list(assign(names, units_total))


def _consistent_count(weighting: dict[str, float], runs: list[list[dict]], *, outcome_field: str, margin: float) -> int:
    consistent = 0
    for rows in runs:
        scored = sorted(rows, key=lambda row: -sum(weighting[name] * row[name] for name in weighting))
        mid = len(scored) // 2
        top, bottom = scored[:mid], scored[mid:]
        if not top or not bottom:
            continue
        top_rate = sum(row[outcome_field] for row in top) / len(top)
        bottom_rate = sum(row[outcome_field] for row in bottom) / len(bottom)
        if top_rate > bottom_rate * (1 + margin):
            consistent += 1
    return consistent


def grid_search_weights(
    runs: list[list[dict]], names: tuple[str, ...], *, step: float = 0.1, outcome_field: str = "outcome", margin: float = 0.2,
    min_row_count: int = 4,
) -> dict:
    """Grid-searched weighting suggestion from `runs` (a list of runs, each
    a list of per-item records carrying every name in `names` plus
    `outcome_field`). Never mutates anything — a human applies the
    suggestion by hand."""
    usable = [rows for rows in runs if sum(1 for r in rows if all(n in r for n in names) and outcome_field in r) >= min_row_count]
    if not usable:
        return {
            "weights": None, "sample_size": len(runs), "usable_runs": 0, "consistent_runs": 0,
            "note": "no run carries every weighted field and an outcome; nothing to search",
        }
    grid = _grid(names, step)
    best_weighting, best_score = None, -1
    for weighting in grid:
        score = _consistent_count(weighting, usable, outcome_field=outcome_field, margin=margin)
        if score > best_score:
            best_score, best_weighting = score, weighting
    return {
        "weights": best_weighting, "sample_size": len(runs), "usable_runs": len(usable),
        "consistent_runs": best_score, "grid_size": len(grid),
        "note": f"grid search over {len(grid)} weightings against {len(usable)} usable run(s): best weighting called {best_score}/{len(usable)} runs consistent",
    }
