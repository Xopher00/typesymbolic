import json

import pytest

from typesymbolic.calibration_store import CalibrationStore


def test_path_is_public_and_matches_where_set_writes(tmp_path):
    store = CalibrationStore(root=tmp_path)
    assert store.path == tmp_path / "calibration.json"
    written = store.set("safe", 0.9, engine="jev", default=0.8, n=25, precision=0.96)
    assert written == store.path
    assert store.path.exists()


def test_get_returns_default_on_unset_key(tmp_path):
    store = CalibrationStore(root=tmp_path)
    assert store.get("safe", engine="jev", default=0.8) == 0.8


def test_float_set_get_round_trip(tmp_path):
    store = CalibrationStore(root=tmp_path)
    store.set("safe", 0.9, engine="jev", default=0.8, n=25, precision=0.96)
    assert store.get("safe", engine="jev", default=0.8) == 0.9


def test_dict_set_get_round_trip(tmp_path):
    store = CalibrationStore(root=tmp_path)
    weights = {"W_CHURN": 0.4, "W_CENTRAL": 0.6}
    store.set("ranking", weights, engine="jev", default={"W_CHURN": 0.5, "W_CENTRAL": 0.5}, n=30, precision=None)
    assert store.get("ranking", engine="jev", default={"W_CHURN": 0.5, "W_CENTRAL": 0.5}) == weights


def test_keys_isolate_across_name_engine_model_revision(tmp_path):
    store = CalibrationStore(root=tmp_path)
    store.set("safe", 0.9, engine="jev", model_revision="v1", default=0.8, n=25, precision=1.0)
    assert store.get("safe", engine="jev", model_revision="v2", default=0.5) == 0.5
    assert store.get("safe", engine="laya", model_revision="v1", default=0.5) == 0.5
    assert store.get("other", engine="jev", model_revision="v1", default=0.5) == 0.5
    assert store.get("safe", engine="jev", model_revision="v1", default=0.5) == 0.9


def test_pooled_key_does_not_collide_with_a_concrete_one(tmp_path):
    store = CalibrationStore(root=tmp_path)
    store.set("safe", 0.9, engine="jev", model_revision=None, default=0.8, n=25, precision=1.0)
    assert store.get("safe", engine="jev", model_revision="v1", default=0.5) == 0.5
    assert store.get("safe", engine="jev", model_revision=None, default=0.5) == 0.9


def test_set_creates_root(tmp_path):
    root = tmp_path / "does-not-exist-yet"
    store = CalibrationStore(root=root)
    store.set("safe", 0.9, engine="jev", default=0.8, n=25, precision=1.0)
    assert root.is_dir()


def test_no_tmp_file_survives_a_write(tmp_path):
    store = CalibrationStore(root=tmp_path)
    store.set("safe", 0.9, engine="jev", default=0.8, n=25, precision=1.0)
    assert [p.name for p in tmp_path.iterdir()] == ["calibration.json"]


def test_missing_file_degrades_to_default(tmp_path):
    store = CalibrationStore(root=tmp_path)
    assert store.get("safe", engine="jev", default=0.8) == 0.8


def test_corrupt_file_degrades_to_default(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "calibration.json").write_text("not json at all")
    store = CalibrationStore(root=tmp_path)
    assert store.get("safe", engine="jev", default=0.8) == 0.8


def test_wrong_schema_degrades_to_default(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "calibration.json").write_text(json.dumps({"schema": 99, "entries": {}}))
    store = CalibrationStore(root=tmp_path)
    assert store.get("safe", engine="jev", default=0.8) == 0.8


def test_out_of_range_stored_value_degrades_to_default(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries = {CalibrationStore.key("safe", "jev", None): {"value": 1.5}}
    (tmp_path / "calibration.json").write_text(json.dumps({"schema": 1, "entries": entries}))
    store = CalibrationStore(root=tmp_path)
    assert store.get("safe", engine="jev", default=0.8) == 0.8


def test_shape_mismatch_degrades_to_default(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries = {CalibrationStore.key("safe", "jev", None): {"value": {"a": 1.0}}}
    (tmp_path / "calibration.json").write_text(json.dumps({"schema": 1, "entries": entries}))
    store = CalibrationStore(root=tmp_path)
    assert store.get("safe", engine="jev", default=0.8) == 0.8


def test_set_rejects_out_of_range_float(tmp_path):
    store = CalibrationStore(root=tmp_path)
    with pytest.raises(ValueError):
        store.set("safe", 1.5, engine="jev", default=0.8, n=25, precision=1.0)


def test_set_rejects_wrong_shape_value(tmp_path):
    store = CalibrationStore(root=tmp_path)
    with pytest.raises(ValueError):
        store.set("safe", "not a number", engine="jev", default=0.8, n=25, precision=1.0)


def test_all_returns_every_stored_value(tmp_path):
    store = CalibrationStore(root=tmp_path)
    store.set("a", 0.9, engine="jev", default=0.8, n=25, precision=1.0)
    store.set("b", 0.7, engine="jev", default=0.8, n=25, precision=1.0)
    assert len(store.all()) == 2


def test_set_does_not_lose_another_process_key_written_after_this_ones_first_load(tmp_path):
    """set() force-reloads from disk before merging, so a second store
    instance's write does not overwrite what the first instance wrote."""
    first = CalibrationStore(root=tmp_path)
    second = CalibrationStore(root=tmp_path)
    first.get("a", engine="jev", default=0.5)  # populates first's cache with an empty store
    second.set("a", 0.9, engine="jev", default=0.8, n=25, precision=1.0)  # written after first's cache was filled

    first.set("b", 0.7, engine="jev", default=0.8, n=25, precision=1.0)

    reread = CalibrationStore(root=tmp_path)
    assert reread.get("a", engine="jev", default=0.5) == 0.9
    assert reread.get("b", engine="jev", default=0.5) == 0.7
