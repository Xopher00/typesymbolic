import pytest

from typesymbolic.domain import ActOutcome, ActStep, Verdict
from typesymbolic.gate import GateResult, GateVerdict
from typesymbolic.journal import Journal, JournalError
from typesymbolic.question import Answer, QuestionRef


def test_record_decision_journals_question_refs_and_scope(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_decision(
        call_id="c1", engine="jev", phase="decide",
        answers={"file.risk:a.py": Answer.from_score("file.risk", 1.5, {}, {"0": 0.1, "1": 0.2, "2": 0.7}, confidence=0.7)},
        questions={"file.risk:a.py": QuestionRef(qid="file.risk", subject="a.py", group="file.risk", scale="confidence")},
        scope={"run_id": "run-1"},
    )

    row = next(journal.replay())
    assert row["schema"] == 2
    assert row["scope"] == {"run_id": "run-1"}
    assert row["questions"]["file.risk:a.py"]["subject"] == "a.py"


def test_record_decision_and_outcome_join_by_call_id(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_decision(
        call_id="c1", engine="jev", phase="decide", model_revision="jev-1.13.0",
        answers={"escalate": Answer.from_noul("escalate", 0.91)},
    )
    journal.record_outcome(
        call_id="c1",
        gate=GateResult(GateVerdict.ACT, "confidence 0.91 >= threshold 0.8", 0.91, "c1"),
        outcome=ActOutcome(succeeded=True, detail={"secret": "never written"}),
    )

    rows = list(journal.replay())
    assert len(rows) == 2
    decision, outcome = rows
    assert decision["type"] == "decision"
    assert decision["call_id"] == "c1"
    assert decision["model_revision"] == "jev-1.13.0"
    assert decision["answers"]["escalate"]["noul"] == 0.91
    assert outcome["type"] == "outcome"
    assert outcome["call_id"] == "c1"
    assert outcome["gate_verdict"] == GateVerdict.ACT
    assert outcome["act_succeeded"] is True
    assert "verify_status" not in outcome


def test_record_verdict_is_its_own_row_and_may_be_deferred(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_decision(call_id="c1", engine="jev", phase="decide", answers={"escalate": Answer.from_noul("escalate", 0.91)})
    journal.record_outcome(call_id="c1", gate=GateResult(GateVerdict.ACT, "ok", 0.91))
    # No verdict yet -- Verify hasn't happened. A later, separate call records it.
    journal.record_verdict(call_id="c1", verdict=Verdict(status="verified"), tests=("escalate",))

    rows = list(journal.replay())
    assert [r["type"] for r in rows] == ["decision", "outcome", "verdict"]
    assert rows[2]["tests"] == ["escalate"]
    assert rows[2]["status"] == "verified"


def test_outcome_row_never_carries_act_outcome_detail(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_outcome(
        call_id="c2",
        gate=GateResult(GateVerdict.PUBLISH, "confidence 0.9 >= threshold 0.8", 0.9),
        outcome=ActOutcome(succeeded=True, detail={"raw_domain_payload": "should never be journaled"}),
    )

    row = next(journal.replay())
    assert "raw_domain_payload" not in str(row)
    assert "detail" not in row


def test_disabled_journal_writes_nothing(tmp_path):
    journal = Journal(root=tmp_path / "does-not-exist", enabled=False)
    journal.record_outcome(call_id="c3", gate=GateResult(GateVerdict.DENY, "deny_listed"))
    assert not (tmp_path / "does-not-exist").exists()
    assert list(journal.replay()) == []


def test_replay_skips_corrupt_lines(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_outcome(call_id="c4", gate=GateResult(GateVerdict.ACT, "ok", 0.9))
    with open(journal._path(), "a", encoding="utf-8") as handle:
        handle.write("not json at all\n")

    rows = list(journal.replay())
    assert len(rows) == 1
    assert rows[0]["call_id"] == "c4"


def test_record_calibration_row_shape(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_calibration(
        group="safe", scale="noul_p", engine="jev", model_revision="jev-1.13.0", current=0.8, proposed=0.85,
        applied=True, n=25, precision=0.96, human_signoff=False, note="threshold 0.85 clears precision 0.96",
    )

    row = next(journal.replay())
    assert row["type"] == "calibration"
    assert "call_id" not in row
    assert row["group"] == "safe"
    assert row["scale"] == "noul_p"
    assert row["current"] == 0.8
    assert row["proposed"] == 0.85
    assert row["applied"] is True


def test_calibration_row_interleaved_does_not_disturb_replay_order(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_decision(call_id="c1", engine="jev", phase="decide", answers={"safe": Answer.from_noul("safe", 0.9)})
    journal.record_calibration(
        group="safe", scale="noul_p", engine="jev", model_revision=None, current=0.8, proposed=0.8,
        applied=False, n=3, precision=None, human_signoff=False, note="n=3 < 20 -- no proposal",
    )
    journal.record_outcome(call_id="c1", gate=GateResult(GateVerdict.ACT, "ok", 0.9))

    rows = list(journal.replay())
    assert [r["type"] for r in rows] == ["decision", "calibration", "outcome"]


def test_record_snapshot_round_trips_an_opaque_payload(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_snapshot(run_id="run-1", tags={"repo": "acme"}, payload={"top_files": ["a.py", "b.py"], "sha": "abc123"})

    row = next(journal.replay())
    assert row["type"] == "snapshot"
    assert row["run_id"] == "run-1"
    assert row["tags"] == {"repo": "acme"}
    assert row["payload"] == {"top_files": ["a.py", "b.py"], "sha": "abc123"}


def test_snapshot_rows_never_enter_the_labeled_index(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_snapshot(run_id="run-1", payload={"anything": True})
    journal.flush()
    assert journal.labeled_pairs("anything", "confidence", engine="jev") == []


def _record_verified_choice(journal, call_id, *, engine="jev", model_revision=None):
    ref = QuestionRef(qid="safe", group="safe", scale="confidence")
    journal.record_decision(
        call_id=call_id, engine=engine, phase="decide", model_revision=model_revision,
        answers={"safe": Answer.from_choice("safe", "yes", {"yes": 0.95, "no": 0.05})},
        questions={"safe": ref},
    )
    journal.record_verdict(call_id=call_id, verdict=Verdict(status="verified"), tests=("safe",))


def test_index_is_rebuilt_from_disk_on_construction(tmp_path):
    first = Journal(root=tmp_path)
    for i in range(3):
        _record_verified_choice(first, f"c{i}", model_revision="rev1")
    first.flush()

    reopened = Journal(root=tmp_path)
    assert len(reopened.labeled_pairs("safe", "confidence", engine="jev", model_revision="rev1")) == 3


def test_a_deferred_verdict_written_by_another_journal_reaches_this_one_after_refresh(tmp_path):
    writer = Journal(root=tmp_path)
    ref = QuestionRef(qid="risk", group="risk", scale="confidence")
    writer.record_decision(
        call_id="c1", engine="jev", phase="decide",
        answers={"risk": Answer.from_choice("risk", "high", {"high": 0.9, "low": 0.1})},
        questions={"risk": ref},
    )
    writer.flush()

    reader = Journal(root=tmp_path)
    assert reader.labeled_pairs("risk", "confidence", engine="jev") == []

    # a second process (a later backtest run) writes the verdict later
    late = Journal(root=tmp_path)
    late.record_verdict(call_id="c1", verdict=Verdict(status="verified"), tests=("risk",))
    late.flush()

    assert reader.labeled_pairs("risk", "confidence", engine="jev") == []  # stale until refreshed
    reader.refresh()
    assert reader.labeled_pairs("risk", "confidence", engine="jev") == [(0.9, True)]


def test_a_later_verdict_supersedes_an_earlier_one_for_the_same_key(tmp_path):
    from datetime import UTC, datetime

    journal = Journal(root=tmp_path)
    ref = QuestionRef(qid="risk", group="risk", scale="confidence")
    journal.record_decision(
        call_id="c1", engine="jev", phase="decide",
        answers={"risk": Answer.from_choice("risk", "high", {"high": 0.9, "low": 0.1})},
        questions={"risk": ref},
    )
    journal.record_verdict(
        call_id="c1", verdict=Verdict(status="failed", observed_at=datetime(2026, 1, 1, tzinfo=UTC)), tests=("risk",),
    )
    journal.record_verdict(
        call_id="c1", verdict=Verdict(status="verified", observed_at=datetime(2026, 6, 1, tzinfo=UTC)), tests=("risk",),
    )
    journal.flush()

    assert journal.labeled_pairs("risk", "confidence", engine="jev") == [(0.9, True)]


def test_a_repeated_dedupe_key_is_ignored(tmp_path):
    journal = Journal(root=tmp_path)
    ref = QuestionRef(qid="risk", group="risk", scale="confidence")
    journal.record_decision(
        call_id="c1", engine="jev", phase="decide",
        answers={"risk": Answer.from_choice("risk", "high", {"high": 0.9, "low": 0.1})},
        questions={"risk": ref},
    )
    journal.record_verdict(call_id="c1", verdict=Verdict(status="verified", dedupe_key="run-a-vs-run-b"), tests=("risk",))
    journal.record_verdict(call_id="c1", verdict=Verdict(status="failed", dedupe_key="run-a-vs-run-b"), tests=("risk",))
    journal.flush()

    # the second write shares the first's dedupe_key -- it's ignored, not applied
    assert journal.labeled_pairs("risk", "confidence", engine="jev") == [(0.9, True)]


def test_two_qids_sharing_a_calib_group_pool_their_labels(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_decision(
        call_id="c1", engine="jev", phase="decide",
        answers={"fit_1": Answer.from_noul("fit_1", 0.8)},
        questions={"fit_1": QuestionRef(qid="fit_1", group="fit", scale="noul_p")},
    )
    journal.record_verdict(call_id="c1", verdict=Verdict(status="verified"), tests=("fit_1",))
    journal.record_decision(
        call_id="c2", engine="jev", phase="decide",
        answers={"fit_2": Answer.from_noul("fit_2", 0.3)},
        questions={"fit_2": QuestionRef(qid="fit_2", group="fit", scale="noul_p")},
    )
    journal.record_verdict(call_id="c2", verdict=Verdict(status="failed"), tests=("fit_2",))
    journal.flush()

    assert sorted(journal.labeled_pairs("fit", "noul_p", engine="jev")) == [(0.3, False), (0.8, True)]


def test_a_verdict_labels_only_the_keys_it_names(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_decision(
        call_id="c1", engine="jev", phase="decide",
        answers={
            "pick": Answer.from_choice("pick", "a", {"a": 0.9, "b": 0.1}),
            "side": Answer.from_noul("side", 0.7),
        },
        questions={
            "pick": QuestionRef(qid="pick", group="pick", scale="confidence"),
            "side": QuestionRef(qid="side", group="side", scale="noul_p"),
        },
    )
    journal.record_verdict(call_id="c1", verdict=Verdict(status="failed"), tests=("pick",))
    journal.flush()

    assert journal.labeled_pairs("pick", "confidence", engine="jev") == [(0.9, False)]
    assert journal.labeled_pairs("side", "noul_p", engine="jev") == []


def test_record_decision_offloads_a_large_state_field_and_replay_resolves_it(tmp_path):
    journal = Journal(root=tmp_path, blob_min_bytes=100)
    big_state = {"screen_xml": "x" * 3000}
    journal.record_decision(call_id="c1", engine="jev", phase="decide", answers={}, state=big_state)

    row = next(journal.replay())
    assert row["state"] == big_state  # replay resolves the blob transparently
    on_disk = journal._path().read_text()
    assert "x" * 3000 not in on_disk  # the raw payload never sits inline in the journal file


def test_record_decision_state_with_a_datetime_does_not_crash_the_writer(tmp_path):
    from datetime import UTC, datetime

    journal = Journal(root=tmp_path, background_writes=False)
    journal.record_decision(
        call_id="c1", engine="jev", phase="decide", answers={},
        state={"observed_at": datetime(2026, 1, 1, tzinfo=UTC), "padding": "x" * 3000},
    )
    row = next(journal.replay())
    assert row["state"]["padding"] == "x" * 3000


def test_record_decision_capture_fields_are_never_read_by_the_calibration_index(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_decision(
        call_id="c1", engine="jev", phase="decide",
        answers={"risk": Answer.from_choice("risk", "high", {"high": 0.9, "low": 0.1})},
        questions={"risk": QuestionRef(qid="risk", group="risk", scale="confidence")},
        state={"raw": "domain payload"}, extra={"trace_id": "abc"},
    )
    journal.record_verdict(call_id="c1", verdict=Verdict(status="verified"), tests=("risk",))
    journal.flush()

    assert journal.labeled_pairs("risk", "confidence", engine="jev") == [(0.9, True)]


def test_many_medium_extra_fields_exceeding_the_row_total_still_keep_the_row_skeleton(tmp_path):
    """record_outcome offloads each extra field on its own, never the row
    as a whole, so many small fields summing past the blob floor still
    leave type/ts/call_id in place and the row indexable."""
    journal = Journal(root=tmp_path, blob_min_bytes=2048)
    extra = {f"field_{i}": "y" * 500 for i in range(10)}  # ~5000 bytes combined; the row as a whole would clear 2048

    journal.record_outcome(call_id="c1", gate=GateResult(GateVerdict.ACT, "ok", 0.9), extra=extra)

    row = next(journal.replay())
    assert row["type"] == "outcome"
    assert row["call_id"] == "c1"
    assert row["gate_verdict"] == GateVerdict.ACT
    for key, value in extra.items():
        assert row[key] == value  # none individually cleared the floor -- none was offloaded


def test_record_outcome_extra_is_merged_flat_and_offloaded_per_field(tmp_path):
    journal = Journal(root=tmp_path, blob_min_bytes=100)
    journal.record_outcome(
        call_id="c1", gate=GateResult(GateVerdict.ACT, "ok", 0.9),
        extra={"probe_output": "y" * 3000, "elapsed_ms": 12},
    )

    row = next(journal.replay())
    assert row["elapsed_ms"] == 12
    assert row["probe_output"] == "y" * 3000
    assert "detail" not in row  # still no ActOutcome.detail leakage


def test_background_writes_false_is_durable_and_indexed_synchronously(tmp_path):
    journal = Journal(root=tmp_path, background_writes=False)
    journal.record_decision(
        call_id="c1", engine="jev", phase="decide",
        answers={"risk": Answer.from_choice("risk", "high", {"high": 0.9, "low": 0.1})},
        questions={"risk": QuestionRef(qid="risk", group="risk", scale="confidence")},
    )
    journal.record_verdict(call_id="c1", verdict=Verdict(status="verified"), tests=("risk",))

    # no flush() call -- background_writes=False means every record_* already landed
    assert journal.labeled_pairs("risk", "confidence", engine="jev") == [(0.9, True)]
    assert journal._path().exists()


def test_injected_clock_drives_the_ts_field(tmp_path):
    from datetime import UTC, datetime

    fixed = datetime(2026, 1, 1, tzinfo=UTC)
    journal = Journal(root=tmp_path, background_writes=False, clock=lambda: fixed)
    journal.record_outcome(call_id="c1", gate=GateResult(GateVerdict.ACT, "ok", 0.9))

    row = next(journal.replay())
    assert row["ts"] == fixed.isoformat(timespec="milliseconds")


def test_record_outcome_writes_steps_key_and_episode_id(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_outcome(
        call_id="c1", gate=GateResult(GateVerdict.ACT, "ok", 0.9), key="pick", episode_id="ep-1",
        outcome=ActOutcome(succeeded=True, steps=(ActStep(name="drop", succeeded=True), ActStep(name="rank"))),
    )

    row = next(journal.replay())
    assert row["key"] == "pick"
    assert row["episode_id"] == "ep-1"
    assert [s["name"] for s in row["steps"]] == ["drop", "rank"]


def test_record_outcome_defaults_key_from_outcome(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_outcome(call_id="c1", gate=GateResult(GateVerdict.ACT, "ok", 0.9), outcome=ActOutcome(succeeded=True, key="drop"))

    row = next(journal.replay())
    assert row["key"] == "drop"


def test_record_outcome_accepts_gate_none_for_code_policy_acts(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_outcome(call_id="c1", gate=None, outcome=ActOutcome(succeeded=True), key="docs_exclusion")

    row = next(journal.replay())
    assert row["gate_verdict"] is None
    assert row["gate_reason"] is None
    assert row["act_succeeded"] is True
    assert row["key"] == "docs_exclusion"


def test_record_outcome_extra_colliding_with_core_column_raises(tmp_path):
    journal = Journal(root=tmp_path)
    with pytest.raises(JournalError):
        journal.record_outcome(call_id="c1", gate=GateResult(GateVerdict.ACT, "ok", 0.9), extra={"key": "collides"})


def test_record_decision_error_row_allows_empty_answers(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_decision(call_id="c1", engine="jev", phase="decide", answers={}, error="timeout")

    row = next(journal.replay())
    assert row["answers"] == {}
    assert row["error"] == "timeout"


def test_record_decision_elapsed_ms_and_asked(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_decision(
        call_id="c1", engine="jev", phase="decide", answers={"escalate": Answer.from_noul("escalate", 0.9)},
        elapsed_ms=42.5, asked={"escalate": "does the device need escalation?"},
    )

    row = next(journal.replay())
    assert row["elapsed_ms"] == 42.5
    assert row["asked"] == {"escalate": "does the device need escalation?"}


def test_record_decision_offloads_a_large_asked_field(tmp_path):
    journal = Journal(root=tmp_path, blob_min_bytes=100)
    big_asked = {"q": "y" * 3000}
    journal.record_decision(call_id="c1", engine="jev", phase="decide", answers={}, asked=big_asked)

    row = next(journal.replay())
    assert row["asked"] == big_asked
    on_disk = journal._path().read_text()
    assert "y" * 3000 not in on_disk


def test_record_verdict_writes_calibrate_field(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_verdict(call_id="c1", verdict=Verdict(status="failed", calibrate=False), tests=("risk",))

    row = next(journal.replay())
    assert row["calibrate"] is False


def test_record_verdict_calibrate_defaults_true(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_verdict(call_id="c1", verdict=Verdict(status="verified"), tests=("risk",))

    row = next(journal.replay())
    assert row["calibrate"] is True


def test_outcomes_reader_filters_by_call_id_and_episode_id(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_decision(call_id="c1", engine="jev", phase="decide", answers={})
    journal.record_outcome(call_id="c1", gate=GateResult(GateVerdict.ACT, "ok", 0.9), episode_id="ep-1")
    journal.record_outcome(call_id="c2", gate=GateResult(GateVerdict.ACT, "ok", 0.9), episode_id="ep-2")

    assert [r["call_id"] for r in journal.outcomes()] == ["c1", "c2"]
    assert [r["call_id"] for r in journal.outcomes(call_id="c1")] == ["c1"]
    assert [r["call_id"] for r in journal.outcomes(episode_id="ep-2")] == ["c2"]
    assert journal.outcomes(call_id="c1", episode_id="ep-2") == []


def test_daily_rotation_splits_files_by_clock_and_reads_across_them(tmp_path):
    from datetime import UTC, datetime

    day1 = Journal(root=tmp_path, rotation="daily", background_writes=False, clock=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    day1.record_outcome(call_id="c1", gate=GateResult(GateVerdict.ACT, "ok", 0.9))

    day2 = Journal(root=tmp_path, rotation="daily", background_writes=False, clock=lambda: datetime(2026, 1, 2, tzinfo=UTC))
    day2.record_outcome(call_id="c2", gate=GateResult(GateVerdict.ACT, "ok", 0.9))

    assert sorted(p.name for p in tmp_path.glob("journal-*.jsonl")) == ["journal-20260101.jsonl", "journal-20260102.jsonl"]
    call_ids = [row["call_id"] for row in day2.replay()]
    assert call_ids == ["c1", "c2"]  # a fresh instance on day 2 still sees day 1's rows
