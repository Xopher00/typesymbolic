"""Threshold calibration over plain journal rows, zero model calls.
Ranking-weight search lives in `weights.py`; `grid_search_weights` is
re-exported here for callers that import it from this module.

`tighten_only_threshold` sweeps candidates for the loosest one clearing a
precision floor for calibration unit `(group, scale)`, tightening only
unless `human_signoff` is set. `recalibrate` applies a tightening
threshold to a `CalibrationStore` and journals every cycle.
`current_threshold` wraps flush/check/recalibrate/read in one
`asyncio.to_thread()` hop, for a caller on an event loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import TypesymbolicError
from .labels import LabelIndex
from .vocab import unit_name
from .weights import grid_search_weights

__all__ = ["grid_search_weights"]  # re-export; everything else here is imported by name

if TYPE_CHECKING:
    from .calibration_store import CalibrationStore
    from .journal import Journal
    from .question import Scale

DEFAULT_CANDIDATES_BY_SCALE: dict[str, tuple[float, ...]] = {
    "noul_p": tuple(round(0.5 + 0.02 * i, 2) for i in range(26)),  # 0.50 .. 1.00
    "noul_not_p": tuple(round(0.5 + 0.02 * i, 2) for i in range(26)),
    "confidence": tuple(round(0.05 + 0.05 * i, 2) for i in range(20)),  # 0.05 .. 1.00
    "margin": tuple(round(0.05 + 0.05 * i, 2) for i in range(20)),
}
DEFAULT_MIN_PRECISION = 0.9
DEFAULT_MIN_LABELS = 20


class PromotionError(TypesymbolicError, RuntimeError):
    """A proposed threshold would loosen the gate without human sign-off."""


@dataclass(frozen=True)
class ThresholdProposal:
    group: str
    scale: str
    current: float
    proposed: float  # never looser than `current` unless human_signoff was set
    n: int
    precision: float | None
    note: str


def _labeled_pairs(
    rows: list[dict], group: str, scale: str, *, engine: str | None = None, model_revision: str | None = None,
) -> tuple[list[tuple[float, bool]], set]:
    """(value, verified) pairs for `(group, scale)`, via a `LabelIndex` built
    from an offline `journal.replay()` list. `engine`/`model_revision`
    filter when given; every pair seen is also reported, for pooling."""
    index = LabelIndex()
    for row in rows:
        index.add_row(row)
    return index.query(group, scale, engine=engine, model_revision=model_revision)


def sweep_threshold(
    pairs: list[tuple[float, bool]], *, min_precision: float = DEFAULT_MIN_PRECISION,
    min_labels: int = DEFAULT_MIN_LABELS, candidates: tuple[float, ...] = DEFAULT_CANDIDATES_BY_SCALE["noul_p"],
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


def _propose_from_pairs(
    pairs: list[tuple[float, bool]], group: str, scale: str, current: float, *, note_suffix: str = "",
    min_precision: float = DEFAULT_MIN_PRECISION, min_labels: int = DEFAULT_MIN_LABELS,
    candidates: tuple[float, ...] | None = None,
) -> tuple[ThresholdProposal, bool]:
    """The sweep, never raising: `(proposal, would_loosen)`. Callers decide
    what to do with a loosening proposal; this just computes it."""
    resolved_candidates = candidates if candidates is not None else DEFAULT_CANDIDATES_BY_SCALE[scale]
    proposed, n, precision, note = sweep_threshold(pairs, min_precision=min_precision, min_labels=min_labels, candidates=resolved_candidates)
    note += note_suffix
    if proposed is None:
        return ThresholdProposal(group, scale, current, current, n, precision, note), False
    return ThresholdProposal(group, scale, current, proposed, n, precision, note), proposed < current


def _propose(
    rows: list[dict], group: str, scale: str, current: float, *, engine: str | None = None, model_revision: str | None = None,
    min_precision: float = DEFAULT_MIN_PRECISION, min_labels: int = DEFAULT_MIN_LABELS,
    candidates: tuple[float, ...] | None = None,
) -> tuple[ThresholdProposal, bool]:
    """`_propose_from_pairs`, sourcing pairs from offline `rows`. Flags
    pooling across (engine, model_revision) when neither is given."""
    pairs, revisions_seen = _labeled_pairs(rows, group, scale, engine=engine, model_revision=model_revision)
    note_suffix = ""
    if engine is None and model_revision is None and len(revisions_seen) > 1:
        seen = sorted(r for r in revisions_seen if r != (None, None))
        note_suffix = f" -- pooled across {seen}; pass engine=/model_revision= to isolate one"
    return _propose_from_pairs(
        pairs, group, scale, current, note_suffix=note_suffix,
        min_precision=min_precision, min_labels=min_labels, candidates=candidates,
    )


def tighten_only_threshold(
    rows: list[dict], group: str, scale: Scale, current: float, *, engine: str | None = None, model_revision: str | None = None,
    min_precision: float = DEFAULT_MIN_PRECISION, min_labels: int = DEFAULT_MIN_LABELS,
    candidates: tuple[float, ...] | None = None, human_signoff: bool = False,
) -> ThresholdProposal:
    """Refit `(group, scale)`'s threshold from journal `rows`. Raises
    `PromotionError` on a looser proposal without `human_signoff`."""
    proposal, would_loosen = _propose(
        rows, group, scale, current, engine=engine, model_revision=model_revision,
        min_precision=min_precision, min_labels=min_labels, candidates=candidates,
    )
    if would_loosen and not human_signoff:
        raise PromotionError(f"{unit_name(group, scale)}: proposed threshold {proposal.proposed} loosens below current {current} without human sign-off")
    return proposal


@dataclass(frozen=True)
class RecalibrationResult:
    proposal: ThresholdProposal
    applied: bool
    threshold: float  # what a gate should use now: the stored value, moved or not
    note: str


def recalibrate(
    *, journal: Journal, store: CalibrationStore, group: str, scale: Scale, engine: str, model_revision: str | None = None,
    default_threshold: float, min_precision: float = DEFAULT_MIN_PRECISION, min_labels: int = DEFAULT_MIN_LABELS,
    candidates: tuple[float, ...] | None = None, human_signoff: bool = False, pool_revisions: bool = False,
) -> RecalibrationResult:
    """Refit `(group, scale)`'s threshold from `journal`'s live index and
    apply it through `store`: tightening applies automatically, loosening
    needs `human_signoff`. Unlike `tighten_only_threshold()`, never raises
    -- a refused proposal is journaled and the incumbent kept. `pool_revisions`
    counts labels from every model revision of the unit instead of only
    `model_revision`."""
    name = unit_name(group, scale)
    current = store.get(name, engine=engine, model_revision=model_revision, default=default_threshold)
    journal.flush()  # guarantee every row enqueued before this call is indexed
    pairs = journal.labeled_pairs(
        group, scale, engine=engine, model_revision=model_revision, any_revision=pool_revisions,
    )
    proposal, would_loosen = _propose_from_pairs(
        pairs, group, scale, current, min_precision=min_precision, min_labels=min_labels, candidates=candidates,
    )
    applied = proposal.proposed != current and (not would_loosen or human_signoff)
    resolved_threshold = proposal.proposed if applied else current
    if applied:
        store.set(
            name, proposal.proposed, engine=engine, model_revision=model_revision,
            default=default_threshold, n=proposal.n, precision=proposal.precision, note=proposal.note,
        )
    journal.record_calibration(
        group=group, scale=scale, engine=engine, model_revision=model_revision, current=current,
        proposed=resolved_threshold, applied=applied,
        n=proposal.n, precision=proposal.precision, human_signoff=human_signoff, note=proposal.note,
    )
    return RecalibrationResult(proposal, applied, resolved_threshold, proposal.note)


async def current_threshold(
    *, journal: Journal | None, store: CalibrationStore, group: str, scale: Scale, engine: str,
    model_revision: str | None = None, default_threshold: float, min_precision: float = DEFAULT_MIN_PRECISION,
    min_labels: int = DEFAULT_MIN_LABELS, candidates: tuple[float, ...] | None = None, human_signoff: bool = False,
    pool_revisions: bool = False,
) -> float:
    """The gate threshold for `(group, scale)`: `store`'s current value,
    refit first if `journal`'s live index has grown past what `store` was
    last fit against. With no `journal`, only reads `store`. `pool_revisions`
    counts labels from every model revision of the unit. All blocking work
    runs in one `asyncio.to_thread()` hop."""
    name = unit_name(group, scale)

    def _resolve() -> float:
        if journal is not None:
            journal.flush()
            fresh_n = len(journal.labeled_pairs(
                group, scale, engine=engine, model_revision=model_revision, any_revision=pool_revisions,
            ))
            if fresh_n > store.get_n(name, engine=engine, model_revision=model_revision):
                recalibrate(
                    journal=journal, store=store, group=group, scale=scale, engine=engine, model_revision=model_revision,
                    default_threshold=default_threshold, min_precision=min_precision, min_labels=min_labels,
                    candidates=candidates, human_signoff=human_signoff, pool_revisions=pool_revisions,
                )
        return store.get(name, engine=engine, model_revision=model_revision, default=default_threshold)

    return await asyncio.to_thread(_resolve)


