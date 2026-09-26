"""Vocabulary protocol: a domain plugin implements this to own a frozen,
versioned set of question wordings instead of improvising instructions text
at call sites. `ask()` must raise on an unknown qid or a slot the wording
references but wasn't supplied — never fall back to improvised wording.

`FrozenVocabulary` is a concrete, standalone implementation: one version
string plus a plain `{qid: {"type", "instructions", "criteria"}}` mapping,
validated whole at construction so a malformed set fails at import time
rather than on whichever call site happens to hit the bad entry.
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from typing import Any, Protocol, get_args

from .errors import TypesymbolicError
from .question import Choice, Noul, NoulCriteria, Question, Scale, Score


class Vocabulary(Protocol):
    version: str

    def ask(self, qid: str, *, criteria: Mapping[str, str | None] | None = None, **slots: str) -> Question: ...


class VocabularyError(TypesymbolicError, RuntimeError):
    """An unknown question id, a slot the wording needs but wasn't supplied,
    or a malformed entry. Fail-closed: never fall back to improvised wording."""


QUESTION_TYPES = ("noul", "choice", "score")
SCALE_VALUES = get_args(Scale)
# A vocabulary entry with no explicit "scale" falls back to this by type.
DEFAULT_SCALE_BY_TYPE: dict[str, Scale] = {"noul": "noul_p", "choice": "confidence", "score": "confidence"}


def calibration_unit(vocab: Vocabulary, qid: str, question: Question) -> tuple[str, Scale]:
    """`(calib_group, scale)` for `qid`. Uses `vocab.unit(qid)` when
    defined (optional, not part of the `Vocabulary` protocol); else
    `(qid, <scale by question type>)`."""
    unit_fn = getattr(vocab, "unit", None)
    if callable(unit_fn):
        found = unit_fn(qid)
        if found is not None:
            return found
    return qid, DEFAULT_SCALE_BY_TYPE[question.type]


def unit_name(group: str, scale: str) -> str:
    """The lookup-key form of a calibration unit (`CalibrationStore`,
    journaled `calibration` rows)."""
    return f"{group}|{scale}"


def _required_slots(qid: str, template: str) -> frozenset[str]:
    """Named placeholders the wording needs, parsed once at construction so a
    missing one is a clear error instead of a format-time KeyError."""
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as error:
        raise VocabularyError(f"{qid}: malformed wording template: {error}") from error
    names = set()
    for _literal, field, _spec, _conv in parsed:
        if field is None:
            continue
        if field == "" or field.isdigit():
            raise VocabularyError(f"{qid}: positional slots are not supported; name every slot")
        names.add(field.split("!")[0].split(":")[0].split(".")[0].split("[")[0])
    return frozenset(names)


class FrozenVocabulary:
    """One frozen, versioned set of question wordings. Bump `version` on
    any wording/criteria change, or journal rows pool as one question.
    A Choice driven by `resolve_one()` has `criteria` overwritten by
    `propose()`; declaring `{}` there is normal."""

    def __init__(self, version: str, entries: Mapping[str, dict]) -> None:
        if not version:
            raise VocabularyError("a vocabulary needs a non-empty version string")
        if not entries:
            raise VocabularyError(f"vocabulary {version} is empty")
        self.version = version
        self.entries: dict[str, dict] = {qid: dict(entry) for qid, entry in entries.items()}
        self._required: dict[str, frozenset[str]] = {}
        for qid, entry in self.entries.items():
            self._validate(qid, entry)
            self._required[qid] = _required_slots(qid, entry["instructions"])

    def unit(self, qid: str) -> tuple[str, Scale] | None:
        """`(calib_group, scale)` declared on `qid`'s entry, or `None`."""
        entry = self.entries.get(qid)
        if entry is None or ("calib_group" not in entry and "scale" not in entry):
            return None
        group = entry.get("calib_group", qid)
        scale = entry.get("scale") or DEFAULT_SCALE_BY_TYPE[entry["type"]]
        return group, scale

    @staticmethod
    def _validate(qid: str, entry: dict) -> None:
        kind = entry.get("type")
        if kind not in QUESTION_TYPES:
            raise VocabularyError(f"{qid}: unknown question type {kind!r}; expected one of {QUESTION_TYPES}")
        instructions = entry.get("instructions")
        if not isinstance(instructions, str) or not instructions:
            raise VocabularyError(f"{qid}: missing instructions")
        calib_group = entry.get("calib_group")
        if calib_group is not None and not isinstance(calib_group, str):
            raise VocabularyError(f"{qid}: calib_group must be a string")
        scale = entry.get("scale")
        if scale is not None and scale not in SCALE_VALUES:
            raise VocabularyError(f"{qid}: scale must be one of {SCALE_VALUES}, got {scale!r}")
        criteria = entry.get("criteria")
        if kind == "noul":
            if criteria is not None and not (
                isinstance(criteria, dict)
                and set(criteria) <= {"true", "false"}
                and all(v is None or isinstance(v, str) for v in criteria.values())
            ):
                raise VocabularyError(f"{qid}: a noul entry's criteria must be a dict of 'true'/'false' -> label or None")
        elif kind == "choice":
            if criteria is not None and not (
                isinstance(criteria, dict) and all(isinstance(k, str) and (v is None or isinstance(v, str)) for k, v in criteria.items())
            ):
                raise VocabularyError(f"{qid}: a choice entry's criteria must be a dict of candidate id -> label or None")
        elif not (isinstance(criteria, list) and criteria and all(isinstance(level, str) for level in criteria)):
            raise VocabularyError(f"{qid}: a score entry's criteria must be a non-empty ordered list of level descriptions")

    def ask(self, qid: str, *, criteria: Mapping[str, str | None] | None = None, **slots: Any) -> Question:
        """`criteria`, when given, is live-enumerated data that overrides the
        entry's own criteria for this one call (a Choice's candidates, or a
        Noul's true/false wording); `None` falls back to the entry."""
        entry = self.entries.get(qid)
        if entry is None:
            known = ", ".join(sorted(self.entries))
            raise VocabularyError(f"unknown question id {qid!r} in vocabulary {self.version} (have: {known})")
        missing = sorted(self._required[qid] - slots.keys())
        if missing:
            raise VocabularyError(f"{qid}: missing slot(s) {missing} -- fail-closed, not improvised wording")
        instructions = entry["instructions"].format(**slots)
        kind = entry["type"]
        if kind == "noul":
            resolved = criteria if criteria is not None else entry.get("criteria")
            return Noul(instructions=instructions, criteria=NoulCriteria(**resolved) if resolved else None)
        if kind == "choice":
            resolved = criteria if criteria is not None else (entry.get("criteria") or {})
            return Choice(instructions=instructions, criteria=dict(resolved))
        return Score(instructions=instructions, criteria=entry["criteria"])
