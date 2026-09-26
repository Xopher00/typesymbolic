"""The decision/verdict label-join rule, shared by `Journal`'s live index
and `calibrate.py`'s offline sweep over raw rows.

A verdict labels only the answer keys named in its `tests`; only
`verified`/`failed` label anything. A later `observed_at` supersedes an
earlier one for the same `(call_id, key)`. A `dedupe_key` is registered
only once it has produced at least one label, so a verdict arriving before
its decision is retried on the next `add_row` pass rather than dropped.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import ValidationError

from .question import Answer, QuestionRef

DECISION = "decision"
VERDICT = "verdict"

_EPOCH = datetime.min.replace(tzinfo=UTC)

# (group, scale, engine, model_revision, value) for one journaled answer key.
_DecisionEntry = tuple[str, str, str, "str | None", float]
# (group, scale, engine, model_revision, value, verified, observed_at) for one label.
_Label = tuple[str, str, str, "str | None", float, bool, datetime]


def _parse_observed_at(value: str) -> datetime:
    """Best-effort ISO8601 parse; a missing or malformed timestamp sorts as
    the oldest possible observation."""
    if not value:
        return _EPOCH
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return _EPOCH
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class LabelIndex:
    """In-memory join of `decision` + `verdict` rows into (value, verified)
    pairs per calibration unit. Not thread-safe; callers with concurrent
    writers lock around it."""

    def __init__(self) -> None:
        self._decisions: dict[str, dict[str, _DecisionEntry]] = {}
        self._labels: dict[tuple[str, str], _Label] = {}
        self._dedupe_seen: set[str] = set()

    def add_row(self, row: dict, *, refs: dict[str, QuestionRef] | None = None) -> None:
        row_type = row.get("type")
        if row_type == DECISION:
            self._add_decision(row, refs)
        elif row_type == VERDICT:
            self._add_verdict(row)

    def _add_decision(self, row: dict, refs: dict[str, QuestionRef] | None = None) -> None:
        call_id, engine, model_revision = row.get("call_id"), row.get("engine"), row.get("model_revision")
        if not call_id:
            return
        answers = row.get("answers") or {}
        if refs is not None:
            for key, ref in refs.items():
                answer = answers.get(key)
                if answer is None:
                    continue
                try:
                    value = Answer.model_validate(answer).value(ref.scale)
                except (ValidationError, ValueError, KeyError):
                    continue
                if value is None:
                    continue
                self._decisions.setdefault(call_id, {})[key] = (ref.group, ref.scale, engine, model_revision, value)
            return
        questions = row.get("questions") or {}
        for key, ref in questions.items():
            answer = answers.get(key)
            if answer is None:
                continue
            try:
                value = Answer.model_validate(answer).value(ref["scale"])
            except (ValidationError, ValueError, KeyError):
                continue
            if value is None:
                continue
            self._decisions.setdefault(call_id, {})[key] = (ref["group"], ref["scale"], engine, model_revision, value)

    def _add_verdict(self, row: dict) -> None:
        call_id = row.get("call_id")
        if not call_id:
            return
        status = row.get("status")
        if status not in ("verified", "failed"):
            return
        if row.get("calibrate", True) is False:
            return
        dedupe_key = row.get("dedupe_key")
        if dedupe_key and dedupe_key in self._dedupe_seen:
            return
        verified = status == "verified"
        observed_at = _parse_observed_at(row.get("observed_at") or row.get("ts") or "")
        decision_keys = self._decisions.get(call_id, {})
        labeled_any = False
        for key in row.get("tests") or []:
            entry = decision_keys.get(key)
            if entry is None:
                continue
            group, scale, engine, model_revision, value = entry
            label_key = (call_id, key)
            prior = self._labels.get(label_key)
            if prior is not None and prior[6] >= observed_at:
                continue
            self._labels[label_key] = (group, scale, engine, model_revision, value, verified, observed_at)
            labeled_any = True
        if dedupe_key and labeled_any:
            self._dedupe_seen.add(dedupe_key)

    def pairs(
        self, group: str, scale: str, *, engine: str, model_revision: str | None = None, any_revision: bool = False,
    ) -> list[tuple[float, bool]]:
        """Pairs for exactly this `engine`. By default `model_revision=None`
        matches "no revision reported" exactly, not a wildcard;
        `any_revision=True` pools every revision (jevdevice's opt-in
        `pool_revisions` calibration)."""
        return [
            (value, verified)
            for g, s, e, mr, value, verified, _observed_at in self._labels.values()
            if g == group and s == scale and e == engine and (any_revision or mr == model_revision)
        ]

    def query(
        self, group: str, scale: str, *, engine: str | None = None, model_revision: str | None = None,
        since: datetime | None = None,
    ) -> tuple[list[tuple[float, bool]], set[tuple[str, str | None]]]:
        """`(pairs, revisions_seen)` for `(group, scale)`. Unlike `pairs()`,
        `engine`/`model_revision` are optional filters -- `None` means
        "don't filter", not "match None exactly". `since`, when given,
        drops any label whose verdict `observed_at` is earlier."""
        matches = [
            (value, verified, e, mr)
            for g, s, e, mr, value, verified, observed_at in self._labels.values()
            if g == group and s == scale and (since is None or observed_at >= since)
        ]
        revisions_seen = {(e, mr) for _value, _verified, e, mr in matches}
        pairs = [
            (value, verified) for value, verified, e, mr in matches
            if (engine is None or e == engine) and (model_revision is None or mr == model_revision)
        ]
        return pairs, revisions_seen
