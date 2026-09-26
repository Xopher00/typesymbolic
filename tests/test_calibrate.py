import pytest

from typesymbolic.calibrate import (
    PromotionError,
    current_threshold,
    recalibrate,
    sweep_threshold,
    tighten_only_threshold,
)
from typesymbolic.calibration_store import CalibrationStore
from typesymbolic.domain import Verdict
from typesymbolic.gate import GateResult, GateVerdict
from typesymbolic.journal import Journal
from typesymbolic.question import Answer, QuestionRef


def _decision_row(call_id: str, confidence: float, engine: str | None = None, model_revision: str | None = None) -> dict:
    return {
        "type": "decision", "call_id": call_id, "engine": engine, "model_revision": model_revision,
        "questions": {"safe": {"qid": "safe", "group": "safe", "scale": "confidence"}},
        "answers": {"safe": {"qid": "safe", "type": "choice", "choice": "yes", "confidence": confidence}},
    }


def _verdict_row(call_id: str, status: str) -> dict:
    return {"type": "verdict", "call_id": call_id, "tests": ["safe"], "status": status, "observed_at": ""}


def _labeled_rows(n_good: int, n_bad: int, good_confidence: float = 0.95, bad_confidence: float = 0.6) -> list[dict]:
    rows = []
    for i in range(n_good):
        call_id = f"good-{i}"
        rows += [_decision_row(call_id, good_confidence), _verdict_row(call_id, "verified")]
    for i in range(n_bad):
        call_id = f"bad-{i}"
        rows += [_decision_row(call_id, bad_confidence), _verdict_row(call_id, "failed")]
    return rows


def test_sweep_threshold_finds_loosest_threshold_clearing_precision():
    pairs = [(0.95, True)] * 18 + [(0.6, False)] * 5 + [(0.6, True)] * 5
    threshold, n, precision, _note = sweep_threshold(pairs, min_precision=0.9, min_labels=15)
    # 0.62 is the loosest candidate above the bad pairs' 0.6 confidence -- the
    # first one that excludes them and so clears 100% precision.
    assert threshold == 0.62
    assert n == 18
    assert precision == 1.0


def test_sweep_threshold_returns_none_on_thin_sample():
    threshold, _n, _precision, note = sweep_threshold([(0.95, True)] * 3, min_labels=15)
    assert threshold is None
    assert "no proposal" in note


def test_tighten_only_threshold_raises_without_signoff_on_looser_proposal():
    rows = _labeled_rows(n_good=25, n_bad=0, good_confidence=0.95)
    with pytest.raises(PromotionError):
        tighten_only_threshold(rows, "safe", "confidence", current=0.99, min_labels=20)


def test_tighten_only_threshold_allows_looser_proposal_with_signoff():
    rows = _labeled_rows(n_good=25, n_bad=0, good_confidence=0.95)
    proposal = tighten_only_threshold(rows, "safe", "confidence", current=0.99, min_labels=20, human_signoff=True)
    assert proposal.proposed < proposal.current


def test_tighten_only_threshold_keeps_current_when_nothing_proposed():
    rows = _labeled_rows(n_good=2, n_bad=1)  # too thin to propose anything
    proposal = tighten_only_threshold(rows, "safe", "confidence", current=0.8, min_labels=20)
    assert proposal.proposed == 0.8
    assert proposal.n == 3


def test_tighten_only_threshold_tightens_when_precision_demands_it():
    rows = _labeled_rows(n_good=15, n_bad=5, good_confidence=0.95, bad_confidence=0.6)
    proposal = tighten_only_threshold(rows, "safe", "confidence", current=0.5, min_precision=0.9, min_labels=15)
    assert proposal.proposed > 0.6  # tightened above the bad pairs' confidence


def test_tighten_only_threshold_ignores_calibration_rows():
    rows = _labeled_rows(n_good=25, n_bad=0, good_confidence=0.95)
    rows.append({
        "type": "calibration", "group": "safe", "scale": "confidence", "engine": "jev", "model_revision": None,
        "current": 0.8, "proposed": 0.9, "applied": True, "n": 25, "precision": 1.0, "human_signoff": False, "note": "applied",
    })
    proposal = tighten_only_threshold(rows, "safe", "confidence", current=0.0, min_labels=20)
    assert proposal.n == 25


def test_tighten_only_threshold_flags_pooling_across_model_revisions():
    rows = [
        _decision_row("a", 0.95, engine="jev", model_revision="jev-1.13.0"), _verdict_row("a", "verified"),
        _decision_row("b", 0.95, engine="jev", model_revision="jev-1.14.0"), _verdict_row("b", "verified"),
    ]
    proposal = tighten_only_threshold(rows, "safe", "confidence", current=0.0, min_labels=2)
    assert "pooled across" in proposal.note
    assert "jev-1.13.0" in proposal.note and "jev-1.14.0" in proposal.note


def test_tighten_only_threshold_flags_pooling_across_different_engines():
    """Two different judge engines could produce colliding `model_revision`
    strings by coincidence -- calibrate.py must key on (engine,
    model_revision) together, not model_revision alone."""
    rows = [
        _decision_row("a", 0.95, engine="jev", model_revision="v1"), _verdict_row("a", "verified"),
        _decision_row("b", 0.5, engine="laya", model_revision="v1"), _verdict_row("b", "failed"),
    ]
    proposal = tighten_only_threshold(rows, "safe", "confidence", current=0.0, min_labels=2)
    assert "pooled across" in proposal.note
    assert "jev" in proposal.note and "laya" in proposal.note


def test_tighten_only_threshold_isolates_one_model_revision():
    rows = [
        _decision_row("a", 0.95, engine="jev", model_revision="jev-1.13.0"), _verdict_row("a", "verified"),
        _decision_row("b", 0.5, engine="jev", model_revision="jev-1.14.0"), _verdict_row("b", "failed"),
    ]
    proposal = tighten_only_threshold(rows, "safe", "confidence", current=0.0, model_revision="jev-1.13.0", min_labels=1)
    assert proposal.n == 1
    assert "pooled across" not in proposal.note


def test_tighten_only_threshold_isolates_one_engine():
    rows = [
        _decision_row("a", 0.95, engine="jev", model_revision="v1"), _verdict_row("a", "verified"),
        _decision_row("b", 0.5, engine="laya", model_revision="v1"), _verdict_row("b", "failed"),
    ]
    proposal = tighten_only_threshold(rows, "safe", "confidence", current=0.0, engine="jev", min_labels=1)
    assert proposal.n == 1
    assert "pooled across" not in proposal.note


def _populate_journal(journal: Journal, *, n_good: int, n_bad: int, good_confidence: float, bad_confidence: float) -> None:
    ref = QuestionRef(qid="safe", group="safe", scale="confidence")
    for i in range(n_good):
        call_id = f"good-{i}"
        answer = Answer.from_choice("safe", "yes", {"yes": good_confidence}, confidence=good_confidence)
        journal.record_decision(call_id=call_id, engine="jev", phase="decide", answers={"safe": answer}, questions={"safe": ref})
        journal.record_outcome(call_id=call_id, gate=GateResult(GateVerdict.ACT, "ok", good_confidence))
        journal.record_verdict(call_id=call_id, verdict=Verdict(status="verified"), tests=("safe",))
    for i in range(n_bad):
        call_id = f"bad-{i}"
        answer = Answer.from_choice("safe", "yes", {"yes": bad_confidence}, confidence=bad_confidence)
        journal.record_decision(call_id=call_id, engine="jev", phase="decide", answers={"safe": answer}, questions={"safe": ref})
        journal.record_outcome(call_id=call_id, gate=GateResult(GateVerdict.ACT, "ok", bad_confidence))
        journal.record_verdict(call_id=call_id, verdict=Verdict(status="failed"), tests=("safe",))


def test_recalibrate_applies_a_tightening_automatically_and_stores_it(tmp_path):
    journal = Journal(root=tmp_path / "journal")
    store = CalibrationStore(root=tmp_path / "store")
    _populate_journal(journal, n_good=15, n_bad=5, good_confidence=0.95, bad_confidence=0.6)

    result = recalibrate(
        journal=journal, store=store, group="safe", scale="confidence", engine="jev",
        default_threshold=0.5, min_precision=0.9, min_labels=15,
    )

    assert result.applied is True
    assert result.threshold > 0.6
    assert store.get("safe|confidence", engine="jev", default=0.5) == result.threshold


def test_recalibrate_writes_a_calibration_row_on_every_call_including_thin_samples(tmp_path):
    journal = Journal(root=tmp_path / "journal")
    store = CalibrationStore(root=tmp_path / "store")
    _populate_journal(journal, n_good=2, n_bad=1, good_confidence=0.95, bad_confidence=0.6)

    result = recalibrate(journal=journal, store=store, group="safe", scale="confidence", engine="jev", default_threshold=0.5, min_labels=20)

    assert result.applied is False
    calibration_rows = [r for r in journal.replay() if r["type"] == "calibration"]
    assert len(calibration_rows) == 1
    assert calibration_rows[0]["applied"] is False


def test_recalibrate_refuses_a_loosening_without_signoff(tmp_path):
    journal = Journal(root=tmp_path / "journal")
    store = CalibrationStore(root=tmp_path / "store")
    _populate_journal(journal, n_good=25, n_bad=0, good_confidence=0.95, bad_confidence=0.6)

    result = recalibrate(journal=journal, store=store, group="safe", scale="confidence", engine="jev", default_threshold=0.99, min_labels=20)

    assert result.applied is False
    assert store.get("safe|confidence", engine="jev", default=0.5) == 0.5  # unchanged -- nothing was ever written
    calibration_rows = [r for r in journal.replay() if r["type"] == "calibration"]
    assert calibration_rows[-1]["applied"] is False
    assert calibration_rows[-1]["note"]  # carries the refusal note, no exception was raised to get here


def test_recalibrate_applies_a_loosening_with_signoff(tmp_path):
    journal = Journal(root=tmp_path / "journal")
    store = CalibrationStore(root=tmp_path / "store")
    _populate_journal(journal, n_good=25, n_bad=0, good_confidence=0.95, bad_confidence=0.6)

    result = recalibrate(
        journal=journal, store=store, group="safe", scale="confidence", engine="jev",
        default_threshold=0.99, min_labels=20, human_signoff=True,
    )

    assert result.applied is True
    assert result.threshold < 0.99
    calibration_rows = [r for r in journal.replay() if r["type"] == "calibration"]
    assert calibration_rows[-1]["human_signoff"] is True


def test_recalibrate_uses_the_stored_value_as_current_not_the_default(tmp_path):
    journal = Journal(root=tmp_path / "journal")
    store = CalibrationStore(root=tmp_path / "store")
    store.set("safe|confidence", 0.7, engine="jev", default=0.5, n=1, precision=None)
    _populate_journal(journal, n_good=2, n_bad=1, good_confidence=0.95, bad_confidence=0.6)

    result = recalibrate(journal=journal, store=store, group="safe", scale="confidence", engine="jev", default_threshold=0.5, min_labels=20)

    assert result.applied is False
    assert result.threshold == 0.7  # kept the stored value, not default_threshold


def test_tighten_only_threshold_still_raises_directly():
    """tighten_only_threshold() raises PromotionError on a looser proposal
    without human_signoff, independent of recalibrate()'s non-raising
    behavior."""
    rows = _labeled_rows(n_good=25, n_bad=0, good_confidence=0.95)
    with pytest.raises(PromotionError):
        tighten_only_threshold(rows, "safe", "confidence", current=0.99, min_labels=20)


async def test_current_threshold_recalibrates_and_returns_the_refit_value(tmp_path):
    journal = Journal(root=tmp_path / "journal")
    store = CalibrationStore(root=tmp_path / "store")
    _populate_journal(journal, n_good=15, n_bad=5, good_confidence=0.95, bad_confidence=0.6)

    result = await current_threshold(
        journal=journal, store=store, group="safe", scale="confidence", engine="jev",
        default_threshold=0.5, min_precision=0.9, min_labels=15,
    )

    assert result > 0.6
    assert store.get("safe|confidence", engine="jev", default=0.5) == result


async def test_current_threshold_skips_recalibration_when_label_count_has_not_grown(tmp_path):
    """With `default_threshold` below every candidate and every labeled
    pair approved at the winning one, the first call applies (a genuine
    tightening) and stores `n` equal to the full label count -- so a
    second call, with no new labels journaled, sees `fresh_n == get_n()`
    and must not refit again."""
    journal = Journal(root=tmp_path / "journal")
    store = CalibrationStore(root=tmp_path / "store")
    _populate_journal(journal, n_good=20, n_bad=0, good_confidence=0.95, bad_confidence=0.6)
    journal.flush()
    first = await current_threshold(
        journal=journal, store=store, group="safe", scale="confidence", engine="jev",
        default_threshold=0.0, min_precision=0.9, min_labels=15,
    )
    assert first > 0.0  # confirms the first call actually applied a fit

    calibration_rows_before = [r for r in journal.replay() if r["type"] == "calibration"]
    second = await current_threshold(
        journal=journal, store=store, group="safe", scale="confidence", engine="jev",
        default_threshold=0.0, min_precision=0.9, min_labels=15,
    )
    calibration_rows_after = [r for r in journal.replay() if r["type"] == "calibration"]

    assert second == first
    assert len(calibration_rows_after) == len(calibration_rows_before)  # no second refit cycle


async def test_current_threshold_with_no_journal_only_reads_the_store(tmp_path):
    store = CalibrationStore(root=tmp_path / "store")
    store.set("safe|confidence", 0.77, engine="jev", default=0.5, n=25, precision=1.0)

    result = await current_threshold(
        journal=None, store=store, group="safe", scale="confidence", engine="jev", default_threshold=0.5,
    )

    assert result == 0.77


def _populate_journal_with_revision(journal: Journal, *, n: int, confidence: float, status: str, model_revision: str) -> None:
    ref = QuestionRef(qid="safe", group="safe", scale="confidence")
    for i in range(n):
        call_id = f"{model_revision}-{i}"
        answer = Answer.from_choice("safe", "yes", {"yes": confidence}, confidence=confidence)
        journal.record_decision(
            call_id=call_id, engine="jev", phase="decide", answers={"safe": answer},
            questions={"safe": ref}, model_revision=model_revision,
        )
        journal.record_outcome(call_id=call_id, gate=GateResult(GateVerdict.ACT, "ok", confidence))
        journal.record_verdict(call_id=call_id, verdict=Verdict(status=status), tests=("safe",))


async def test_current_threshold_pool_revisions_false_only_counts_the_given_revision(tmp_path):
    journal = Journal(root=tmp_path / "journal")
    store = CalibrationStore(root=tmp_path / "store")
    _populate_journal_with_revision(journal, n=15, confidence=0.95, status="verified", model_revision="jev-1")
    _populate_journal_with_revision(journal, n=15, confidence=0.95, status="verified", model_revision="jev-2")

    result = await current_threshold(
        journal=journal, store=store, group="safe", scale="confidence", engine="jev", model_revision="jev-1",
        default_threshold=0.0, min_precision=0.9, min_labels=20,
    )

    assert result == 0.0  # only 15 labels under jev-1, below min_labels=20 -- no fit


async def test_current_threshold_pool_revisions_true_counts_every_revision(tmp_path):
    journal = Journal(root=tmp_path / "journal")
    store = CalibrationStore(root=tmp_path / "store")
    _populate_journal_with_revision(journal, n=15, confidence=0.95, status="verified", model_revision="jev-1")
    _populate_journal_with_revision(journal, n=15, confidence=0.95, status="verified", model_revision="jev-2")

    result = await current_threshold(
        journal=journal, store=store, group="safe", scale="confidence", engine="jev", model_revision=None,
        default_threshold=0.0, min_precision=0.9, min_labels=20, pool_revisions=True,
    )

    assert result > 0.0  # pooled 30 labels across both revisions clears min_labels=20
