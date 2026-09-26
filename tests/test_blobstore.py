from typesymbolic.blobstore import BLOB_KEY, BlobStore, offload, resolve


def test_store_and_load_round_trips_bytes(tmp_path):
    store = BlobStore(tmp_path)
    sha = store.store(b"hello world")
    assert store.load(sha) == b"hello world"


def test_store_is_content_addressed(tmp_path):
    store = BlobStore(tmp_path)
    a = store.store(b"same content")
    b = store.store(b"same content")
    assert a == b


def test_offload_leaves_small_values_untouched(tmp_path):
    store = BlobStore(tmp_path)
    value = {"small": "x"}
    assert offload(value, store=store, min_bytes=2048) == value


def test_offload_swaps_a_large_string_for_a_blob_ref(tmp_path):
    store = BlobStore(tmp_path)
    big = "x" * 3000
    result = offload({"screen": big}, store=store, min_bytes=2048)
    assert BLOB_KEY in result["screen"]
    assert result["screen"][BLOB_KEY]["bytes"] == 3000
    assert resolve(result, store=store) == {"screen": big}


def test_offload_disabled_at_zero_min_bytes(tmp_path):
    store = BlobStore(tmp_path)
    big = "x" * 3000
    result = offload({"screen": big}, store=store, min_bytes=0)
    assert result == {"screen": big}


def test_offload_leaves_several_medium_fields_alone_when_none_alone_clears_the_floor(tmp_path):
    store = BlobStore(tmp_path)
    medium_fields = {f"field_{i}": "y" * 100 for i in range(10)}  # ~1000 bytes combined, none alone over 2048
    result = offload(medium_fields, store=store, min_bytes=2048)
    assert result == medium_fields  # under the floor even summed; nothing offloaded


def test_offload_offloads_children_first_then_the_parent_if_still_large(tmp_path):
    store = BlobStore(tmp_path)
    value = {"a": "y" * 500, "b": "y" * 500, "c": "y" * 500, "d": "y" * 500, "e": "y" * 500}
    result = offload(value, store=store, min_bytes=100)
    # every child clears the low floor first; the parent, now a dict of refs, still clears it too
    assert BLOB_KEY in result
    resolved = resolve(result, store=store)
    assert resolved == value


def test_resolve_is_a_no_op_on_a_value_with_no_blob_refs(tmp_path):
    store = BlobStore(tmp_path)
    value = {"a": 1, "b": [1, 2, {"c": "d"}]}
    assert resolve(value, store=store) == value


def test_a_dict_literally_named_blob_key_with_non_dict_content_is_not_mistaken_for_a_ref(tmp_path):
    store = BlobStore(tmp_path)
    value = {BLOB_KEY: "not a ref dict"}
    assert resolve(value, store=store) == value


def test_offload_does_not_crash_on_a_datetime_value(tmp_path):
    """offload()'s json.dumps uses `default=str`, so a datetime anywhere in
    state=/extra= serializes instead of raising, keeping the journal's
    fail-open write guarantee. A small datetime leaf stays under the floor
    and is returned unchanged; a large sibling next to it is still blobbed
    correctly."""
    from datetime import UTC, datetime

    store = BlobStore(tmp_path)
    captured_at = datetime(2026, 1, 1, tzinfo=UTC)
    value = {"captured_at": captured_at, "padding": "y" * 3000}

    result = offload(value, store=store, min_bytes=2048)

    assert result["captured_at"] == captured_at  # too small to blob -- returned as-is
    assert BLOB_KEY in result["padding"]
    assert resolve(result, store=store)["padding"] == "y" * 3000
