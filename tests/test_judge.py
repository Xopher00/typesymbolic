import asyncio

import httpx2
import pytest
import typesafe_sdk

from typesymbolic.judge import (
    AskResult,
    JevEngine,
    JudgeError,
    ScriptedJudge,
    SyncJevSession,
    ask_all_sync,
)
from typesymbolic.question import Answer, Choice, Noul, Score


async def test_scripted_judge_returns_script_entries_in_order():
    questions = {"escalate": Noul(instructions="should we escalate?")}
    script = [
        {"escalate": Answer.from_noul("escalate", 0.9)},
        {"escalate": Answer.from_noul("escalate", 0.1)},
    ]
    judge = ScriptedJudge(script)

    first = await judge.ask_all({"finding": "a"}, questions)
    second = await judge.ask_all({"finding": "b"}, questions)

    assert first.answers["escalate"].noul == 0.9
    assert first.model_revision == "scripted"
    assert second.answers["escalate"].noul == 0.1
    assert [state for state, _ in judge.calls] == [{"finding": "a"}, {"finding": "b"}]


async def test_scripted_judge_raises_when_script_exhausted():
    judge = ScriptedJudge([])
    with pytest.raises(JudgeError):
        await judge.ask_all({}, {"q": Noul(instructions="?")})


def test_jev_engine_requires_api_key():
    with pytest.raises(JudgeError):
        JevEngine(api_key="")


def test_jev_engine_forwards_retry_to_the_sdk_client(monkeypatch):
    captured = {}
    real_client = typesafe_sdk.AsyncTypeSafeClient

    def spy(**kwargs):
        captured.update(kwargs)
        return real_client(**kwargs)

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", spy)
    retry = typesafe_sdk.RetryPolicy(max_retries=5)

    JevEngine(api_key="fake-key", retry=retry)

    assert captured["retry"] is retry


async def test_jev_engine_requires_at_least_one_question():
    engine = JevEngine(api_key="fake-key")
    with pytest.raises(JudgeError):
        await engine.ask_all({}, {})


async def test_jev_engine_parses_all_three_answer_shapes():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "jev-latest",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "answers": {
                    "escalate": {"type": "noul", "noul": 0.91},
                    "category": {"type": "choice", "choice": "auth", "probabilities": {"auth": 0.95, "other": 0.05}, "confidence": 0.95},
                    "severity": {
                        "type": "score", "score": 1.98,
                        "legend": {0: "low", 1: "medium", 2: "high"},
                        "probabilities": {0: 0.02, 1: 0.1, 2: 0.88},
                        "confidence": 0.98,
                    },
                },
            },
        )

    engine = JevEngine(api_key="fake-key", transport=httpx2.MockTransport(handler))
    questions = {
        "escalate": Noul(instructions="escalate now?"),
        "category": Choice(instructions="pick a category", criteria={"auth": "auth issue", "other": "other"}),
        "severity": Score(instructions="rate severity", criteria=["low", "medium", "high"]),
    }

    result = await engine.ask_all({"finding": "..."}, questions)
    answers = result.answers

    assert result.model_revision == "jev-latest"
    assert answers["escalate"].noul == 0.91
    assert answers["escalate"].confidence == pytest.approx(abs(0.91 - 0.5) * 2)
    assert answers["category"].choice == "auth"
    assert answers["category"].confidence == 0.95
    assert answers["severity"].score == 1.98
    assert answers["severity"].confidence == 0.98
    assert answers["severity"].legend == {"0": "low", "1": "medium", "2": "high"}


async def test_jev_engine_raises_judge_error_on_api_failure():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"error": {"message": "boom"}})

    engine = JevEngine(api_key="fake-key", transport=httpx2.MockTransport(handler))
    with pytest.raises(JudgeError):
        await engine.ask_all({}, {"q": Noul(instructions="?")})


def test_ask_all_sync_runs_a_synchronous_judge_call():
    questions = {"escalate": Noul(instructions="should we escalate?")}
    judge = ScriptedJudge([{"escalate": Answer.from_noul("escalate", 0.9)}])

    result = ask_all_sync(judge, {"finding": "a"}, questions)

    assert result.answers["escalate"].noul == 0.9
    assert result.model_revision == "scripted"


class _LoopSpyJudge:
    """Records which event loop was running for each `ask_all()` call --
    proves `SyncJevSession` reuses one loop rather than a fresh one per
    call, the exact mismatch `ask_all_sync()` reused across calls has."""

    name = "loop-spy"

    def __init__(self) -> None:
        self.loops: list[object] = []

    async def ask_all(self, state: dict, questions) -> AskResult:
        self.loops.append(asyncio.get_running_loop())
        return AskResult(answers={}, model_revision="loop-spy")


def test_sync_jev_session_reuses_one_event_loop_across_calls():
    judge = _LoopSpyJudge()
    with SyncJevSession(judge) as session:
        session.ask_all({}, {"q": Noul(instructions="?")})
        session.ask_all({}, {"q": Noul(instructions="?")})

    assert len(judge.loops) == 2
    assert judge.loops[0] is judge.loops[1]


def test_sync_jev_session_closes_the_judge_if_it_has_aclose():
    class _ClosableJudge:
        name = "closable"
        closed = False

        async def ask_all(self, state: dict, questions) -> AskResult:
            return AskResult(answers={}, model_revision="closable")

        async def aclose(self) -> None:
            self.closed = True

    judge = _ClosableJudge()
    session = SyncJevSession(judge)
    session.ask_all({}, {"q": Noul(instructions="?")})
    session.close()

    assert judge.closed is True


def test_sync_jev_session_close_is_safe_without_an_aclose_method():
    judge = ScriptedJudge([{"q": Answer.from_noul("q", 0.5)}])
    session = SyncJevSession(judge)
    session.ask_all({}, {"q": Noul(instructions="?")})
    session.close()  # ScriptedJudge has no aclose() -- must not raise


async def test_jev_engine_aclose_closes_the_underlying_client():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"model": "jev-latest", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": {"q": {"type": "noul", "noul": 0.5}}})

    engine = JevEngine(api_key="fake-key", transport=httpx2.MockTransport(handler))
    await engine.ask_all({}, {"q": Noul(instructions="?")})
    assert engine._client._http_client.is_closed is False

    await engine.aclose()

    assert engine._client._http_client.is_closed is True


async def test_jev_engine_async_context_manager_closes_on_exit():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"model": "jev-latest", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": {"q": {"type": "noul", "noul": 0.5}}})

    async with JevEngine(api_key="fake-key", transport=httpx2.MockTransport(handler)) as engine:
        await engine.ask_all({}, {"q": Noul(instructions="?")})

    assert engine._client._http_client.is_closed is True


async def test_jev_engine_wraps_typesafe_error_as_cause():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"error": {"message": "bad key"}})

    engine = JevEngine(api_key="fake-key", transport=httpx2.MockTransport(handler))
    try:
        await engine.ask_all({}, {"q": Noul(instructions="?")})
        pytest.fail("expected JudgeError")
    except JudgeError as error:
        assert isinstance(error.__cause__, typesafe_sdk.TypeSafeError)
