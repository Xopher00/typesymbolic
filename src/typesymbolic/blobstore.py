"""Content-addressed blob storage for the journal's optional capture
fields (`state`/`extra`) -- a big value goes out of line so the journal
file stays small.

`offload()`/`resolve()` walk a value's dict/list structure children-first,
swapping/restoring anything at or above `min_bytes`. Callers offload one
named field at a time, never the row dict as a whole."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ._io import atomic_write

BLOB_KEY = "$blob"  # namespaced marker, so a domain field literally named "blob" can't collide


class BlobStore:
    """One content-addressed file per blob, at `root/<sha[:2]>/<sha>`."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, sha256: str) -> Path:
        return self.root / sha256[:2] / sha256

    def store(self, data: bytes) -> str:
        sha256 = hashlib.sha256(data).hexdigest()
        path = self._path(sha256)
        if not path.exists():
            atomic_write(path, data)
        return sha256

    def load(self, sha256: str) -> bytes:
        return self._path(sha256).read_bytes()


def _is_blob_ref(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {BLOB_KEY} and isinstance(value[BLOB_KEY], dict)


def offload(value: Any, *, store: BlobStore, min_bytes: int) -> Any:
    """`value`, with any dict/list/str at or above `min_bytes` (serialized)
    swapped for a `{BLOB_KEY: {sha256, bytes, encoding}}` ref, children
    first. `min_bytes` of 0 disables offloading."""
    if min_bytes <= 0:
        return value
    if isinstance(value, dict):
        value = {k: offload(v, store=store, min_bytes=min_bytes) for k, v in value.items()}
    elif isinstance(value, list):
        value = [offload(v, store=store, min_bytes=min_bytes) for v in value]
    if isinstance(value, str):
        data, encoding = value.encode("utf-8"), "utf-8"
    else:
        data, encoding = json.dumps(value, separators=(",", ":"), default=str).encode("utf-8"), "json"
    size = len(data)
    if size < min_bytes:
        return value
    sha256 = store.store(data)
    return {BLOB_KEY: {"sha256": sha256, "bytes": size, "encoding": encoding}}


def resolve(value: Any, *, store: BlobStore) -> Any:
    """Inverse of `offload()`, recursive: a blobbed parent's own content
    may carry further refs from its already-blobbed children."""
    if _is_blob_ref(value):
        ref = value[BLOB_KEY]
        data = store.load(ref["sha256"])
        loaded = json.loads(data.decode("utf-8")) if ref.get("encoding") == "json" else data.decode("utf-8")
        return resolve(loaded, store=store)
    if isinstance(value, dict):
        return {k: resolve(v, store=store) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, store=store) for v in value]
    return value
