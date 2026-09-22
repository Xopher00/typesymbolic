"""Calibrated values -- a threshold float or a weight dict -- stored beside
the journal under the same caller-configured `root`. Modeled on a sibling
project's override-file pattern: one JSON object, read once per process with
a cache, falls back to a caller's default on anything wrong with it (never
raises on read -- an optional layer degrades, it doesn't break a live
decision), written whole with provenance on every recalibration.

One store, two producers: `calibrate.tighten_only_threshold()`'s scalar
threshold and `calibrate.grid_search_weights()`'s weight dict are both plain
JSON values keyed the same way, so they share one file and one class instead
of two parallel persistence stories. Only the threshold path has a live
consumer today (`engine.resolve_one()`); the weights path is wired to the
same store and ready for whichever domain plugin's ranking logic needs it --
see `calibrate.py`'s `grid_search_weights()` docstring for why that isn't
built ahead of a real caller.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

CalibrationValue = float | dict[str, float]

FILENAME = "calibration.json"
SCHEMA = 1


class CalibrationStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._cache: dict[str, dict] | None = None

    @staticmethod
    def key(name: str, engine: str, model_revision: str | None) -> str:
        """Valid only for the (engine, model_revision) it was fit against —
        `calibrate.py` keys its pairs the same way; a pooled fit
        (`model_revision=None`) is stored and read back under the pooled key."""
        return f"{name}|{engine}|{model_revision or ''}"

    def _path(self) -> Path:
        return self.root / FILENAME

    def _load(self) -> dict[str, dict]:
        if self._cache is not None:
            return self._cache
        self._cache = {}
        path = self._path()
        if not path.exists():
            return self._cache
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._cache  # unreadable/corrupt: use defaults
        if isinstance(record, dict) and record.get("schema") == SCHEMA and isinstance(record.get("entries"), dict):
            self._cache = record["entries"]
        return self._cache

    def get(self, name: str, *, engine: str, model_revision: str | None = None, default: CalibrationValue) -> CalibrationValue:
        entry = self._load().get(self.key(name, engine, model_revision))
        if not entry:
            return default
        value = entry.get("value")
        if isinstance(default, dict):
            if not (isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values())):
                return default
            return {k: float(v) for k, v in value.items()}
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return default
        value = float(value)
        return value if 0.0 <= value <= 1.0 else default

    def set(
        self, name: str, value: CalibrationValue, *, engine: str, model_revision: str | None = None,
        default: CalibrationValue, n: int, precision: float | None = None, note: str = "",
    ) -> Path | None:
        """Write `value` as the new override. Raises `ValueError` for a
        caller-passed value of the wrong shape or an out-of-range float --
        that's a programming error, not malformed on-disk data. Never raises
        on the write itself: an unwritable home degrades to `None`, same as
        `Journal`'s write path."""
        if isinstance(value, dict):
            if not all(isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool) for k, v in value.items()):
                raise ValueError(f"weight dict {value!r} must map str -> float")
            stored: CalibrationValue = {k: float(v) for k, v in value.items()}
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"threshold {value} outside [0, 1]")
            stored = float(value)
        else:
            raise ValueError(f"value {value!r} must be a float or a dict[str, float]")  # noqa: TRY004

        entries = dict(self._load())
        entries[self.key(name, engine, model_revision)] = {
            "name": name, "engine": engine, "model_revision": model_revision,
            "value": stored, "default": default, "n": n, "precision": precision,
            "updated_at": datetime.now(UTC).isoformat(timespec="milliseconds"), "note": note,
        }
        record = {"schema": SCHEMA, "entries": entries}
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / f".{FILENAME}.{os.getpid()}.tmp"
            tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, self._path())
        except OSError:
            return None
        self._cache = entries  # refill so this process sees its own write immediately
        return self._path()

    def all(self) -> dict[str, dict]:
        """Every stored value, for a caller inspecting what calibration did."""
        return dict(self._load())
