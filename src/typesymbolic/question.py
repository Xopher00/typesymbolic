"""The three typed-judgment primitives a judge answers: it is never asked
for free text, only one of these closed shapes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NoulCriteria(BaseModel):
    """Outcome descriptions for a noul's two poles; `None` is the wire
    convention for an undescribed pole."""

    true: str | None = None
    false: str | None = None


class Noul(BaseModel):
    """Yes/no over the given state; the answer is a bare probability."""

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: NoulCriteria | None = None


class Choice(BaseModel):
    """Pick exactly one of a closed, code-enumerated set of candidates.
    `criteria` maps a candidate id to its label, never invented by the
    judge or caller; `None` for a candidate with no wording."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str | None]


class Score(BaseModel):
    """Place the state on a closed, ordered rubric. `criteria` is an ordered
    list of level descriptions (index 0 = lowest), not a map — the answer's
    `score` is a float index into this list."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


Question = Noul | Choice | Score

# Raw quantity a gate/calibration fit reads -- never synthesized across types.
Scale = Literal["noul_p", "noul_not_p", "confidence", "margin"]


class QuestionRef(BaseModel):
    """The frozen question a journaled answer key came from, which
    real-world item it asked about, and which calibration unit it pools
    into. `vocab.calibration_unit()` resolves `group`/`scale`."""

    qid: str
    vocab_version: str | None = None
    subject: str | None = None
    group: str
    scale: Scale


class Answer(BaseModel):
    """One judged answer, normalized across the three question types.
    `confidence` is the judge's own reported value for choice/score;
    `None` for a noul -- never synthesized. `value()` reads whatever raw
    quantity a gate actually wants.

    `abstained` is a typed signal that the judge declined, instead of a
    sentinel baked into `choice`; no wire format sets it natively, so a
    caller sets it after construction per its judge's own convention."""

    qid: str
    type: Literal["noul", "choice", "score"]
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    legend: dict[str, str] | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    abstained: bool = False

    def value(self, scale: Scale) -> float | None:
        """The raw quantity `scale` names, or `None` if this answer doesn't
        carry it -- callers must treat `None` as "nothing to gate on"."""
        if scale == "noul_p":
            return self.noul if self.type == "noul" else None
        if scale == "noul_not_p":
            return 1.0 - self.noul if self.type == "noul" and self.noul is not None else None
        if scale == "confidence":
            return self.confidence
        if scale == "margin":
            if not self.probabilities or len(self.probabilities) < 2:
                return None
            top1, top2 = sorted(self.probabilities.values(), reverse=True)[:2]
            return top1 - top2
        raise ValueError(f"unknown scale {scale!r}")

    @classmethod
    def from_noul(cls, qid: str, noul: float, *, abstained: bool = False) -> Answer:
        return cls(qid=qid, type="noul", noul=noul, abstained=abstained)

    @classmethod
    def from_choice(
        cls, qid: str, choice: str, probabilities: dict[str, float], confidence: float | None = None, *, abstained: bool = False,
    ) -> Answer:
        return cls(
            qid=qid,
            type="choice",
            choice=choice,
            probabilities=probabilities,
            confidence=confidence if confidence is not None else probabilities.get(choice, 0.0),
            abstained=abstained,
        )

    @classmethod
    def from_score(
        cls, qid: str, score: float, legend: dict[str, str], probabilities: dict[str, float], confidence: float,
        *, abstained: bool = False,
    ) -> Answer:
        return cls(
            qid=qid, type="score", score=score, legend=legend, probabilities=probabilities, confidence=confidence,
            abstained=abstained,
        )
