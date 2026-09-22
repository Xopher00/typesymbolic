"""The three typed-judgment primitives a judge answers: it is never asked
for free text, only one of these closed shapes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Noul(BaseModel):
    """Yes/no over the given state; the answer is a bare probability."""

    type: Literal["noul"] = "noul"
    instructions: str


class Choice(BaseModel):
    """Pick exactly one of a closed, code-enumerated set of candidates.

    `criteria` maps a candidate id to its label — ids must come from real,
    live-enumerated options (device candidates, file paths, ...), never
    invented by the judge or the caller.
    """

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]


class Score(BaseModel):
    """Place the state on a closed, ordered rubric. `criteria` is an ordered
    list of level descriptions (index 0 = lowest), not a map — the answer's
    `score` is a float index into this list."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


Question = Noul | Choice | Score


class Answer(BaseModel):
    """One judged answer, normalized across the three question types.

    `confidence` is always populated: a Noul answer carries no native
    confidence, so it degrades to the noul probability's distance from 0.5,
    doubled — this gives every question type one number to threshold on.

    `abstained` is an explicit signal that the judge declined rather than
    picking a real candidate — a typed field instead of a sentinel value
    baked into `choice` (e.g. a conventional "none_of_these" option) that
    every caller would otherwise have to string-match for. No wire format
    this package talks to reports it natively, so it defaults `False` and a
    caller sets it after construction for whichever abstention convention
    its judge actually uses.
    """

    qid: str
    type: Literal["noul", "choice", "score"]
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    legend: dict[str, str] | None = None
    probabilities: dict[str, float] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False

    @classmethod
    def from_noul(cls, qid: str, noul: float, *, abstained: bool = False) -> Answer:
        return cls(qid=qid, type="noul", noul=noul, confidence=abs(noul - 0.5) * 2, abstained=abstained)

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
