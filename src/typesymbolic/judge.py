"""Typed-judge boundary: a swappable backend behind one `ask_all()` call.
Any `JudgeEngine` must answer with a genuinely calibrated `confidence`,
not a self-reported guess -- `gate.py`/`calibrate.py` mean what they say
only against that. `AskResult.model_revision` is the concrete version
that answered, never a requested alias; `journal.py`/`calibrate.py` key
on it alongside `name`.

`JevEngine`'s pooled HTTP client binds to whichever event loop first used
it, so `ask_all_sync()` (via `asyncio.run()`) is safe for exactly one call
per instance -- a second call crashes with "Event loop is closed".
`SyncJevSession` keeps one loop alive for a synchronous caller making more
than one call.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol, Self

from .errors import TypesymbolicError
from .question import Answer, Choice, Noul, Question, Score


class JudgeError(TypesymbolicError, RuntimeError):
    """A judge call failed to produce usable answers."""


@dataclass
class AskResult:
    answers: dict[str, Answer]
    model_revision: str | None = None  # the concrete model version that answered, not a requested alias
    usage: dict[str, int] | None = None  # input_tokens, output_tokens
    elapsed_ms: float | None = None


class JudgeEngine(Protocol):
    name: str

    async def ask_all(self, state: dict, questions: dict[str, Question]) -> AskResult: ...


def ask_all_sync(judge: JudgeEngine, state: dict, questions: dict[str, Question]) -> AskResult:
    """One `ask_all()` call from sync code. Not for use inside a running
    event loop; safe for exactly one call per `judge` -- see module docstring."""
    return asyncio.run(judge.ask_all(state, questions))


class SyncJevSession:
    """Repeated `ask_all()` calls from sync code, one event loop for the
    whole session -- never hands a pooled connection between loops."""

    def __init__(self, judge: JudgeEngine) -> None:
        self._judge = judge
        self._loop = asyncio.new_event_loop()

    def ask_all(self, state: dict, questions: dict[str, Question]) -> AskResult:
        return self._loop.run_until_complete(self._judge.ask_all(state, questions))

    @property
    def judge(self) -> JudgeEngine:
        return self._judge

    def run(self, coro):
        """Run any coroutine on this session's loop, e.g. `engine.resolve_one()`
        against the same pooled connection."""
        return self._loop.run_until_complete(coro)

    def ask_batch(self, questions, facts, *, journal=None, phase=None, refs=None, scope=None):
        from .engine import ask_batch

        return self._loop.run_until_complete(
            ask_batch(judge=self._judge, questions=questions, facts=facts, journal=journal, phase=phase, refs=refs, scope=scope)
        )

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
        criteria = question.criteria.model_dump() if question.criteria is not None else None
        return sdk.Noul(instructions=question.instructions, criteria=criteria)
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


def _default_retry(sdk):
    """3 attempts on 429/529, deterministic exponential backoff (1s, 2s, no
    jitter) -- matches jevdevice's `JevClient` retry loop exactly."""
    return sdk.RetryPolicy(max_retries=2, http_statuses={429, 529}, backoff_initial=1.0, backoff_jitter=0.0)


class JevEngine:
    """Jev over the native TypeSafe API. One request per `ask_all()` call
    -- batch questions rather than calling repeatedly."""

    def __init__(
        self, api_key: str, model: str | None = None, *, name: str = "jev", timeout: float = 30.0,
        transport: object | None = None, retry: object | None = None,
    ) -> None:
        if not api_key:
            raise JudgeError("TYPESAFE_AI_API is required")
        import typesafe_sdk  # optional dependency (pyproject's "jev" extra)

        self.name = name
        self._sdk = typesafe_sdk
        self._client = typesafe_sdk.AsyncTypeSafeClient(
            api_key=api_key, model=model, timeout=timeout, transport=transport,
            retry=_default_retry(typesafe_sdk) if retry is None else retry,
        )

    async def ask_all(self, state: dict, questions: dict[str, Question]) -> AskResult:
        if not questions:
            raise JudgeError("ask_all() requires at least one question")
        sdk_questions = {qid: _to_sdk_question(self._sdk, q) for qid, q in questions.items()}
        started = time.perf_counter()
        try:
            response = await self._client.system_one(state, sdk_questions)
        except self._sdk.TypeSafeError as error:
            raise JudgeError(f"Jev request failed ({type(error).__name__}: {error})") from error
        elapsed_ms = (time.perf_counter() - started) * 1000
        answers = {qid: _from_sdk_answer(qid, answer) for qid, answer in response.answers.items()}
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        return AskResult(answers=answers, model_revision=response.model, usage=usage, elapsed_ms=elapsed_ms)

    async def aclose(self) -> None:
        """Release the pooled HTTP connection before its event loop ends."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


class ScriptedJudge:
    """Answers in call order from a script, for offline tests. An entry
    is a `{qid: Answer}` dict or an `AskResult`, for a fixed revision."""

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
