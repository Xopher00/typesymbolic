import httpx2
import pytest
import typesafe_sdk

from typesymbolic.judge import JevEngine, JudgeError, ScriptedJudge
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


async def test_jev_engine_wraps_typesafe_error_as_cause():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"error": {"message": "bad key"}})

    engine = JevEngine(api_key="fake-key", transport=httpx2.MockTransport(handler))
    try:
        await engine.ask_all({}, {"q": Noul(instructions="?")})
        pytest.fail("expected JudgeError")
    except JudgeError as error:
        assert isinstance(error.__cause__, typesafe_sdk.TypeSafeError)
