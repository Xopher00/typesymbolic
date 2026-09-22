"""Typed-judge boundary: a swappable backend behind one `ask_all()` call.
`JevEngine` is one implementation, not the contract — any engine that
answers Noul/Choice/Score with a genuinely calibrated `confidence` (not a
model's self-reported guess) can implement `JudgeEngine` and drop in, HTTP-
backed or fully local. `gate.py`'s thresholds and `calibrate.py`'s refit
only mean what they say against a confidence that was actually calibrated
for this question shape; an engine that fakes it breaks that silently, not
loudly, so this is a contract to honor deliberately, not a type checker
will catch.

`JevEngine` wraps the official `typesafe-sdk` client against the native
TypeSafe API — retries, response validation, and error handling are
maintained upstream instead of hand-rolled here. It journals nothing and
tracks no usage itself; those are separate concerns (`journal.py`,
`engine.py`) that compose around a plain `ask_all()` call rather than living
inside it. `AskResult.model_revision` is the concrete version that actually
answered (never a requested alias) — `journal.py` and `calibrate.py` key on
it alongside `name` because a threshold is only valid for the specific
engine+revision it was fit against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .question import Answer, Choice, Noul, Question, Score


class JudgeError(RuntimeError):
    """A judge call failed to produce usable answers."""


@dataclass
class AskResult:
    answers: dict[str, Answer]
    model_revision: str | None = None  # the concrete model version that answered, not a requested alias


class JudgeEngine(Protocol):
    name: str

    async def ask_all(self, state: dict, questions: dict[str, Question]) -> AskResult: ...


def _to_sdk_question(sdk, question: Question):
    if isinstance(question, Noul):
        return sdk.Noul(instructions=question.instructions)
    if isinstance(question, Choice):
        return sdk.Choice(instructions=question.instructions, criteria=question.criteria)
    if isinstance(question, Score):
        return sdk.Score(instructions=question.instructions, criteria=question.criteria)
    raise JudgeError(f"unknown question type: {type(question).__name__}")


def _from_sdk_answer(qid: str, answer) -> Answer:
    if answer.type == "noul":
        return Answer.from_noul(qid, answer.noul)
    if answer.type == "choice":
        return Answer.from_choice(qid, answer.choice, answer.probabilities, confidence=answer.confidence)
    if answer.type == "score":
        legend = {str(k): v for k, v in (answer.legend or {}).items()}
        probabilities = {str(k): v for k, v in answer.probabilities.items()}
        return Answer.from_score(qid, answer.score, legend, probabilities, answer.confidence)
    raise JudgeError(f"unknown answer type from Jev: {answer.type!r}")


class JevEngine:
    """Jev over the native TypeSafe API, via `typesafe-sdk`'s
    `AsyncTypeSafeClient`. One request per ask_all() call — independent
    questions over the same state are answered in parallel server-side, so
    callers should batch questions rather than call ask_all() repeatedly."""

    name = "jev"

    def __init__(
        self, api_key: str, model: str | None = None, *, timeout: float = 30.0, transport: object | None = None
    ) -> None:
        if not api_key:
            raise JudgeError("TYPESAFE_AI_API is required")
        import typesafe_sdk  # optional dependency (pyproject's "jev" extra)

        self._sdk = typesafe_sdk
        self._client = typesafe_sdk.AsyncTypeSafeClient(api_key=api_key, model=model, timeout=timeout, transport=transport)

    async def ask_all(self, state: dict, questions: dict[str, Question]) -> AskResult:
        if not questions:
            raise JudgeError("ask_all() requires at least one question")
        sdk_questions = {qid: _to_sdk_question(self._sdk, q) for qid, q in questions.items()}
        try:
            response = await self._client.system_one(state, sdk_questions)
        except self._sdk.TypeSafeError as error:
            raise JudgeError(f"Jev request failed ({type(error).__name__}: {error})") from error
        answers = {qid: _from_sdk_answer(qid, answer) for qid, answer in response.answers.items()}
        return AskResult(answers=answers, model_revision=response.model)


class ScriptedJudge:
    """Answers questions in the order ask_all() is called, one script entry
    per call — for offline tests, no network. A script entry is either a
    plain `{qid: Answer}` dict (wrapped with `model_revision="scripted"`) or
    an `AskResult`, for a test that needs a specific revision."""

    name = "scripted"

    def __init__(self, script: list[dict[str, Answer] | AskResult]) -> None:
        self._script = list(script)
        self.calls: list[tuple[dict, dict[str, Question]]] = []

    async def ask_all(self, state: dict, questions: dict[str, Question]) -> AskResult:
        self.calls.append((state, questions))
        if not self._script:
            raise JudgeError("ScriptedJudge ran out of scripted answers")
        entry = self._script.pop(0)
        return entry if isinstance(entry, AskResult) else AskResult(answers=entry, model_revision="scripted")
