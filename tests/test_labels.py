from datetime import UTC, datetime

from typesymbolic.labels import LabelIndex


def _decision(call_id, *, key="safe", group="safe", scale="confidence", engine="jev", model_revision=None, confidence=0.9):
    return {
        "type": "decision", "call_id": call_id, "engine": engine, "model_revision": model_revision,
        "questions": {key: {"qid": key, "group": group, "scale": scale}},
        "answers": {key: {"qid": key, "type": "choice", "choice": "yes", "confidence": confidence}},
    }


def _verdict(call_id, status, *, tests=("safe",), dedupe_key=None, observed_at=""):
    return {
        "type": "verdict", "call_id": call_id, "tests": list(tests), "status": status,
        "dedupe_key": dedupe_key, "observed_at": observed_at,
    }


def test_a_verdict_labels_only_the_keys_it_names():
    index = LabelIndex()
    index.add_row(_decision("c1", key="a", group="a"))
    index.add_row({**_decision("c1", key="b", group="b"), "call_id": "c1"})
    index.add_row(_verdict("c1", "failed", tests=("a",)))
    assert index.pairs("a", "confidence", engine="jev") == [(0.9, False)]
    assert index.pairs("b", "confidence", engine="jev") == []


def test_an_unconfirmed_verdict_sharing_a_dedupe_key_does_not_block_a_later_verified_one():
    """An unconfirmed verdict labels nothing, so its dedupe_key is not
    marked seen and does not block a later verified verdict sharing it."""
    index = LabelIndex()
    index.add_row(_decision("c1"))
    index.add_row(_verdict("c1", "unconfirmed", dedupe_key="k1"))
    index.add_row(_verdict("c1", "verified", dedupe_key="k1"))
    assert index.pairs("safe", "confidence", engine="jev") == [(0.9, True)]


def test_dedupe_key_is_not_consumed_until_a_label_is_actually_produced():
    """A verdict processed before its matching decision produces no label,
    so its dedupe_key is not marked seen and is retried on the next pass."""
    index = LabelIndex()
    index.add_row(_verdict("c1", "verified", dedupe_key="k1"))  # decision not seen yet
    assert index.pairs("safe", "confidence", engine="jev") == []
    index.add_row(_decision("c1"))
    index.add_row(_verdict("c1", "verified", dedupe_key="k1"))  # retried
    assert index.pairs("safe", "confidence", engine="jev") == [(0.9, True)]


def test_a_repeated_dedupe_key_after_a_successful_label_is_ignored():
    index = LabelIndex()
    index.add_row(_decision("c1"))
    index.add_row(_verdict("c1", "verified", dedupe_key="k1"))
    index.add_row(_verdict("c1", "failed", dedupe_key="k1"))
    assert index.pairs("safe", "confidence", engine="jev") == [(0.9, True)]


def test_observed_at_supersession_compares_datetimes_not_strings():
    """Supersession parses `observed_at` into a datetime before comparing.
    "2026-01-01T13:00+02:00" (== 11:00 UTC) sorts AFTER
    "2026-01-01T12:00:00.000+00:00" as a plain string (since "13" > "12"),
    but is the earlier observation and must not supersede it."""
    index = LabelIndex()
    index.add_row(_decision("c1"))
    index.add_row(_verdict("c1", "verified", observed_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC).isoformat(timespec="milliseconds")))
    index.add_row(_verdict("c1", "failed", observed_at="2026-01-01T13:00:00.000+02:00"))
    assert index.pairs("safe", "confidence", engine="jev") == [(0.9, True)]


def test_malformed_observed_at_sorts_as_oldest_and_is_superseded():
    index = LabelIndex()
    index.add_row(_decision("c1"))
    index.add_row(_verdict("c1", "failed", observed_at="not a timestamp"))
    index.add_row(_verdict("c1", "verified", observed_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(timespec="milliseconds")))
    assert index.pairs("safe", "confidence", engine="jev") == [(0.9, True)]


def test_query_filters_are_optional_unlike_pairs():
    index = LabelIndex()
    index.add_row(_decision("c1", engine="jev", model_revision="v1", confidence=0.9))
    index.add_row(_verdict("c1", "verified"))
    index.add_row(_decision("c2", engine="laya", model_revision="v2", confidence=0.5))
    index.add_row(_verdict("c2", "failed"))

    all_pairs, revisions = index.query("safe", "confidence")
    assert sorted(all_pairs) == [(0.5, False), (0.9, True)]
    assert revisions == {("jev", "v1"), ("laya", "v2")}

    jev_only, _ = index.query("safe", "confidence", engine="jev")
    assert jev_only == [(0.9, True)]


def test_pairs_treats_none_model_revision_as_an_exact_match_not_a_wildcard():
    index = LabelIndex()
    index.add_row(_decision("c1", engine="jev", model_revision="v1"))
    index.add_row(_verdict("c1", "verified"))
    assert index.pairs("safe", "confidence", engine="jev", model_revision=None) == []
    assert index.pairs("safe", "confidence", engine="jev", model_revision="v1") == [(0.9, True)]


def test_pairs_any_revision_pools_every_model_revision():
    index = LabelIndex()
    index.add_row(_decision("c1", engine="jev", model_revision="v1"))
    index.add_row(_verdict("c1", "verified"))
    index.add_row(_decision("c2", engine="jev", model_revision="v2", confidence=0.5))
    index.add_row(_verdict("c2", "failed"))
    assert index.pairs("safe", "confidence", engine="jev", model_revision="v1") == [(0.9, True)]
    assert sorted(index.pairs("safe", "confidence", engine="jev", any_revision=True)) == [(0.5, False), (0.9, True)]


def test_query_since_drops_labels_observed_before_the_cutoff():
    index = LabelIndex()
    index.add_row(_decision("c1"))
    index.add_row(_verdict("c1", "verified", observed_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(timespec="milliseconds")))
    index.add_row(_decision("c2", key="a", group="safe"))
    index.add_row(_verdict("c2", "failed", tests=("a",), observed_at=datetime(2026, 2, 1, tzinfo=UTC).isoformat(timespec="milliseconds")))

    all_pairs, _ = index.query("safe", "confidence")
    assert sorted(all_pairs) == [(0.9, False), (0.9, True)]

    recent_pairs, _ = index.query("safe", "confidence", since=datetime(2026, 1, 15, tzinfo=UTC))
    assert recent_pairs == [(0.9, False)]


def test_a_verdict_with_calibrate_false_is_recorded_but_not_labeled():
    index = LabelIndex()
    index.add_row(_decision("c1"))
    index.add_row({**_verdict("c1", "verified"), "calibrate": False})
    assert index.pairs("safe", "confidence", engine="jev") == []


def test_a_verdict_with_no_calibrate_key_defaults_to_true():
    index = LabelIndex()
    index.add_row(_decision("c1"))
    index.add_row(_verdict("c1", "verified"))
    assert "calibrate" not in _verdict("c1", "verified")
    assert index.pairs("safe", "confidence", engine="jev") == [(0.9, True)]
