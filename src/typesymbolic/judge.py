"""Typed-judge boundary: a swappable backend behind one `ask_all()` call.
`JevEngine` is one implementation, not the contract — any `JudgeEngine`
must answer with a genuinely calibrated `confidence`, not a self-reported
guess; `gate.py`/`calibrate.py` only mean what they say against that.

`JevEngine` wraps `typesafe-sdk` against the native TypeSafe API; it
journals nothing and tracks no usage itself, that's `journal.py`/`engine.py`.
`AskResult.model_revision` is the concrete version that answered (never a
requested alias) — `journal.py` and `calibrate.py` key on it alongside `name`.

A `JevEngine` holds one pooled HTTP client for its whole lifetime, and that
pool binds to whichever event loop first used it — `asyncio.run()` tears
its loop down on return, so calling `ask_all_sync()` twice against the same
`JevEngine` crashes the second call with "Event loop is closed", not just
leaks a connection. `ask_all_sync()` is for exactly one call; a synchronous
caller making more than one must use `SyncJevSession`, which keeps one loop
alive for the session instead of one per call.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, Self

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


def ask_all_sync(judge: JudgeEngine, state: dict, questions: dict[str, Question]) -> AskResult:
    """One `ask_all()` call from synchronous code, via `asyncio.run()` — for
    a caller whose own control flow isn't async and shouldn't have to
    become so just to reach `JudgeEngine`. Not for use inside a running
    event loop, and safe for exactly one call per `judge` — see the module
    docstring; `SyncJevSession` is the safe primitive for more than one."""
    return asyncio.run(judge.ask_all(state, questions))


class SyncJevSession:
    """Drives repeated `ask_all()` calls from synchronous code inside one
    event loop kept alive for the whole session — unlike calling
    `ask_all_sync()` more than once against the same `judge`, this never
    hands a pooled connection from one loop to another."""

    def __init__(self, judge: JudgeEngine) -> None:
        self._judge = judge
        self._loop = asyncio.new_event_loop()

    def ask_all(self, state: dict, questions: dict[str, Question]) -> AskResult:
        return self._loop.run_until_complete(self._judge.ask_all(state, questions))

    def close(self) -> None:
        aclose = getattr(self._judge, "aclose", None)
        if aclose is not None:
            self._loop.run_until_complete(aclose())
        self._loop.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


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
        self, api_key: str, model: str | None = None, *, timeout: float = 30.0,
        transport: object | None = None, retry: object | None = None,
    ) -> None:
        if not api_key:
            raise JudgeError("TYPESAFE_AI_API is required")
        import typesafe_sdk  # optional dependency (pyproject's "jev" extra)

        self._sdk = typesafe_sdk
        self._client = typesafe_sdk.AsyncTypeSafeClient(
            api_key=api_key, model=model, timeout=timeout, transport=transport, retry=retry,
        )

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

    async def aclose(self) -> None:
        """Release the pooled HTTP connection. Required before letting the
        event loop that made this engine's calls end — an unclosed pool
        left bound to a dead loop is exactly `ask_all_sync()`'s crash."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


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
