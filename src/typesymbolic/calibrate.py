"""Generic calibration utilities over plain journal rows, zero model calls:

- `tighten_only_threshold` joins decision rows' answer confidence to their
  outcome row's `verify_status` by `call_id`, sweeps ascending candidate
  thresholds for the loosest one clearing a precision floor on enough
  labeled rows, then only ever moves the incumbent threshold tighter unless
  a caller passes a recorded `human_signoff`. A threshold is only valid for
  the (engine, model_revision) it was fit against — pass either to isolate
  one judge's data; otherwise every pair present is pooled and flagged in
  the result if there's more than one.

- `grid_search_weights` grid-searches a set of named weights summing to 1,
  scored against how many runs (each a list of per-item records with an
  outcome field) a weighting would call "consistent" — its top-ranked
  half's outcome rate meaningfully exceeding the bottom half's. Its output
  is exactly as JSON-safe as a threshold and could persist through the same
  `CalibrationStore` `recalibrate()` writes to, but nothing in this
  package's domain-blind core consumes a weight dict yet (a weighted
  ranking is domain-specific), so no `recalibrate_weights()` exists to
  apply one automatically — building that ahead of a real caller would be
  exactly the speculative work this package elsewhere avoids.

- `recalibrate` is the one function here that mutates anything: it reads a
  journal, refits `qid`'s threshold, and writes the result to a
  `CalibrationStore` when it's safe to (a tightening move, always; a
  loosening move, only with `human_signoff`) — the automatic-calibration
  entry point `engine.resolve_one()`'s `store=` parameter reads back from.

Every other function here degrades to a documented no-op on thin or
malformed input rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .calibration_store import CalibrationStore
    from .journal import Journal

DEFAULT_CANDIDATE_THRESHOLDS = tuple(round(0.5 + 0.02 * i, 2) for i in range(26))  # 0.50 .. 1.00
DEFAULT_MIN_PRECISION = 0.9
DEFAULT_MIN_LABELS = 20


class PromotionError(RuntimeError):
    """A proposed threshold would loosen the gate without human sign-off."""


@dataclass(frozen=True)
class ThresholdProposal:
    qid: str
    current: float
    proposed: float  # never looser than `current` unless human_signoff was set
    n: int
    precision: float | None
    note: str


def _labeled_pairs(
    rows: list[dict], qid: str, *, engine: str | None = None, model_revision: str | None = None,
) -> tuple[list[tuple[float, bool]], set]:
    """(confidence, verified) pairs: one per decision row's `qid` answer,
    joined to its outcome row's `verify_status` by `call_id`. Rows without a
    verify_status, or with "escalated"/"unconfirmed" (no ground truth to fit
    against), are skipped. A threshold is only meaningful for the specific
    (engine, model_revision) it was fit against — two different engines
    could otherwise produce colliding `model_revision` strings — so both are
    tracked together, not `model_revision` alone. When `engine`/
    `model_revision` are given, only matching pairs are kept; either way,
    every (engine, model_revision) pair actually present in the matched rows
    is returned so a caller can tell whether it pooled across a model
    change."""
    verified_by_call: dict[str, bool] = {}
    for row in rows:
        if row.get("type") == "outcome" and row.get("verify_status") in ("verified", "failed"):
            verified_by_call[row["call_id"]] = row["verify_status"] == "verified"
    pairs = []
    revisions_seen = set()
    for row in rows:
        if row.get("type") != "decision":
            continue
        answer = (row.get("answers") or {}).get(qid)
        label = verified_by_call.get(row.get("call_id"))
        if not answer or label is None or answer.get("confidence") is None:
            continue
        row_identity = (row.get("engine"), row.get("model_revision"))
        revisions_seen.add(row_identity)
        if engine is not None and row_identity[0] != engine:
            continue
        if model_revision is not None and row_identity[1] != model_revision:
            continue
        pairs.append((answer["confidence"], label))
    return pairs, revisions_seen


def sweep_threshold(
    pairs: list[tuple[float, bool]], *, min_precision: float = DEFAULT_MIN_PRECISION,
    min_labels: int = DEFAULT_MIN_LABELS, candidates: tuple[float, ...] = DEFAULT_CANDIDATE_THRESHOLDS,
) -> tuple[float | None, int, float | None, str]:
    """The loosest `candidates` threshold clearing `min_precision` on
    `>= min_labels` approved pairs. Returns
    (proposed_threshold_or_None, n, precision_or_None, note)."""
    if len(pairs) < min_labels:
        return None, len(pairs), None, f"n={len(pairs)} < {min_labels} -- no proposal (keep collecting)"
    for t in candidates:
        approved = [(p, y) for p, y in pairs if p >= t]
        if len(approved) < min_labels:
            continue
        precision = sum(1 for _, y in approved if y) / len(approved)
        if precision >= min_precision:
            return t, len(approved), precision, f"threshold {t} clears precision {precision:.2f} on n={len(approved)}"
    return None, len(pairs), None, f"precision {min_precision} unreachable on this window (n={len(pairs)}) -- keep incumbent"


def _propose(
    rows: list[dict], qid: str, current: float, *, engine: str | None = None, model_revision: str | None = None,
    min_precision: float = DEFAULT_MIN_PRECISION, min_labels: int = DEFAULT_MIN_LABELS,
    candidates: tuple[float, ...] = DEFAULT_CANDIDATE_THRESHOLDS,
) -> tuple[ThresholdProposal, bool]:
    """The sweep, never raising: returns `(proposal, would_loosen)`, where
    `proposal.proposed` is the swept value even when it would loosen —
    `tighten_only_threshold()` and `recalibrate()` each decide what to do
    with a loosening proposal, this just computes it."""
    pairs, revisions_seen = _labeled_pairs(rows, qid, engine=engine, model_revision=model_revision)
    proposed, n, precision, note = sweep_threshold(pairs, min_precision=min_precision, min_labels=min_labels, candidates=candidates)
    if engine is None and model_revision is None and len(revisions_seen) > 1:
        seen = sorted(r for r in revisions_seen if r != (None, None))
        note += f" -- pooled across {seen}; pass engine=/model_revision= to isolate one"
    if proposed is None:
        return ThresholdProposal(qid, current, current, n, precision, note), False
    return ThresholdProposal(qid, current, proposed, n, precision, note), proposed < current


def tighten_only_threshold(
    rows: list[dict], qid: str, current: float, *, engine: str | None = None, model_revision: str | None = None,
    min_precision: float = DEFAULT_MIN_PRECISION, min_labels: int = DEFAULT_MIN_LABELS,
    candidates: tuple[float, ...] = DEFAULT_CANDIDATE_THRESHOLDS, human_signoff: bool = False,
) -> ThresholdProposal:
    """Refit `qid`'s gate threshold from journal `rows`. Raises
    `PromotionError` — instead of silently keeping `current` — when the
    sweep proposes a looser threshold and `human_signoff` isn't set, so a
    caller can't accidentally auto-apply a loosening it never reviewed.

    `engine`/`model_revision` isolate one judge's confidence data; left
    `None`, every (engine, model_revision) pair present is pooled and the
    returned `note` flags it if there's more than one, since a threshold fit
    against pooled data from different judges may not hold for either one."""
    proposal, would_loosen = _propose(
        rows, qid, current, engine=engine, model_revision=model_revision,
        min_precision=min_precision, min_labels=min_labels, candidates=candidates,
    )
    if would_loosen and not human_signoff:
        raise PromotionError(f"{qid}: proposed threshold {proposal.proposed} loosens below current {current} without human sign-off")
    return proposal


@dataclass(frozen=True)
class RecalibrationResult:
    proposal: ThresholdProposal
    applied: bool
    threshold: float  # what a gate should use now: the stored value, moved or not
    note: str


def recalibrate(
    *, journal: Journal, store: CalibrationStore, qid: str, engine: str, model_revision: str | None = None,
    default_threshold: float, min_precision: float = DEFAULT_MIN_PRECISION, min_labels: int = DEFAULT_MIN_LABELS,
    candidates: tuple[float, ...] = DEFAULT_CANDIDATE_THRESHOLDS, human_signoff: bool = False,
) -> RecalibrationResult:
    """Refit `qid`'s threshold from the journal and apply it through `store`.
    A tightening move applies automatically — tightening is safe by
    construction, so it needs no human in the loop, which is what makes a
    system built on this calibrate itself from what it logged. A loosening
    move is never applied without `human_signoff`.

    Unlike `tighten_only_threshold()` (which raises `PromotionError` for a
    direct caller who should see that), `recalibrate()` never raises: the
    sweep routinely proposes a value below a conservative incumbent even on
    healthy data, and a driver meant to be called repeatedly can't have every
    ordinary cycle raise. The refusal is recorded and the incumbent kept
    instead — same "degrade, don't break the caller" rule as
    `CalibrationStore`.

    Every cycle is journaled, applied or not.
    """
    current = store.get(qid, engine=engine, model_revision=model_revision, default=default_threshold)
    rows = list(journal.replay())
    proposal, would_loosen = _propose(
        rows, qid, current, engine=engine, model_revision=model_revision,
        min_precision=min_precision, min_labels=min_labels, candidates=candidates,
    )
    applied = proposal.proposed != current and not (would_loosen and not human_signoff)
    if applied:
        store.set(
            qid, proposal.proposed, engine=engine, model_revision=model_revision,
            default=default_threshold, n=proposal.n, precision=proposal.precision, note=proposal.note,
        )
    journal.record_calibration(
        qid=qid, engine=engine, model_revision=model_revision, current=current,
        proposed=proposal.proposed if applied else current, applied=applied,
        n=proposal.n, precision=proposal.precision, human_signoff=human_signoff, note=proposal.note,
    )
    return RecalibrationResult(proposal, applied, proposal.proposed if applied else current, proposal.note)


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
