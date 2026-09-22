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
from typing import Any, Protocol

from .question import Choice, Noul, Question, Score


class Vocabulary(Protocol):
    version: str

    def ask(self, qid: str, **slots: str) -> Question: ...


class VocabularyError(RuntimeError):
    """An unknown question id, a slot the wording needs but wasn't supplied,
    or a malformed entry. Fail-closed: never fall back to improvised wording."""


QUESTION_TYPES = ("noul", "choice", "score")


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
    """One frozen, versioned set of question wordings. Bump `version` on any
    change to an existing entry's wording or criteria — journal rows spanning
    the change are otherwise pooled as if one question.

    A Choice driven by `engine.resolve_one()` has its `criteria` overwritten
    with `propose()`'s live-enumerated candidates, so declaring `{}` there is
    normal; criteria is load-bearing for a Score's ordered rubric and for a
    Choice asked outside `resolve_one()`.
    """

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

    @staticmethod
    def _validate(qid: str, entry: dict) -> None:
        kind = entry.get("type")
        if kind not in QUESTION_TYPES:
            raise VocabularyError(f"{qid}: unknown question type {kind!r}; expected one of {QUESTION_TYPES}")
        instructions = entry.get("instructions")
        if not isinstance(instructions, str) or not instructions:
            raise VocabularyError(f"{qid}: missing instructions")
        criteria = entry.get("criteria")
        if kind == "noul":
            if criteria is not None:
                raise VocabularyError(f"{qid}: a noul entry carries no criteria")
        elif kind == "choice":
            if criteria is not None and not (
                isinstance(criteria, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in criteria.items())
            ):
                raise VocabularyError(f"{qid}: a choice entry's criteria must be a dict of candidate id -> label")
        elif not (isinstance(criteria, list) and criteria and all(isinstance(level, str) for level in criteria)):
            raise VocabularyError(f"{qid}: a score entry's criteria must be a non-empty ordered list of level descriptions")

    def ask(self, qid: str, **slots: Any) -> Question:
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
            return Noul(instructions=instructions)
        if kind == "choice":
            return Choice(instructions=instructions, criteria=entry.get("criteria") or {})
        return Score(instructions=instructions, criteria=entry["criteria"])
