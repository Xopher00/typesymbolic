from typesymbolic.domain import ActOutcome, Verdict
from typesymbolic.gate import GateResult, GateVerdict
from typesymbolic.journal import Journal
from typesymbolic.question import Answer


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
        verdict=Verdict(status="verified"),
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
    assert outcome["verify_status"] == "verified"


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
        qid="safe", engine="jev", model_revision="jev-1.13.0", current=0.8, proposed=0.85,
        applied=True, n=25, precision=0.96, human_signoff=False, note="threshold 0.85 clears precision 0.96",
    )

    row = next(journal.replay())
    assert row["type"] == "calibration"
    assert "call_id" not in row
    assert row["qid"] == "safe"
    assert row["current"] == 0.8
    assert row["proposed"] == 0.85
    assert row["applied"] is True


def test_calibration_row_interleaved_does_not_disturb_replay_order(tmp_path):
    journal = Journal(root=tmp_path)
    journal.record_decision(call_id="c1", engine="jev", phase="decide", answers={"safe": Answer.from_noul("safe", 0.9)})
    journal.record_calibration(
        qid="safe", engine="jev", model_revision=None, current=0.8, proposed=0.8,
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
    assert journal.labeled_pairs("anything", engine="jev") == []


def test_index_is_rebuilt_from_disk_on_construction(tmp_path):
    first = Journal(root=tmp_path)
    for i in range(3):
        call_id = f"c{i}"
        first.record_decision(call_id=call_id, engine="jev", phase="decide", model_revision="rev1", answers={"safe": Answer.from_noul("safe", 0.95)})
        first.record_outcome(call_id=call_id, gate=GateResult(GateVerdict.ACT, "ok", 0.95), verdict=Verdict(status="verified"))
    first.flush()

    reopened = Journal(root=tmp_path)
    assert len(reopened.labeled_pairs("safe", engine="jev", model_revision="rev1")) == 3


def test_tighten_only_threshold_ignores_calibration_rows(tmp_path):
    from typesymbolic.calibrate import tighten_only_threshold

    journal = Journal(root=tmp_path)
    for i in range(25):
        call_id = f"c{i}"
        journal.record_decision(call_id=call_id, engine="jev", phase="decide", answers={"safe": Answer.from_noul("safe", 0.95)})
        journal.record_outcome(call_id=call_id, gate=GateResult(GateVerdict.ACT, "ok", 0.95), verdict=Verdict(status="verified"))
    journal.record_calibration(
        qid="safe", engine="jev", model_revision=None, current=0.8, proposed=0.9,
        applied=True, n=25, precision=1.0, human_signoff=False, note="applied",
    )

    proposal = tighten_only_threshold(list(journal.replay()), "safe", current=0.5, min_labels=20)
    assert proposal.n == 25
