"""JevEngine additions: name override, usage/elapsed_ms, retry, frozen wire shapes."""

import json

import httpx2
import pytest
import typesafe_sdk

from typesymbolic.judge import AskResult, JevEngine, JudgeError
from typesymbolic.question import Choice, Noul, Score


def _ok_response(**extra):
    return httpx2.Response(
        200,
        json={
            "model": "jev-latest",
            "usage": {"input_tokens": 7, "output_tokens": 3},
            "answers": {"q": {"type": "noul", "noul": 0.5}},
            **extra,
        },
    )


async def test_jev_engine_defaults_to_jev_name():
    engine = JevEngine(api_key="fake-key")
    assert engine.name == "jev"


async def test_jev_engine_accepts_a_name_override():
    engine = JevEngine(api_key="fake-key", name="laya")
    assert engine.name == "laya"


async def test_jev_engine_populates_usage_and_elapsed_ms():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _ok_response()

    engine = JevEngine(api_key="fake-key", transport=httpx2.MockTransport(handler))
    result = await engine.ask_all({}, {"q": Noul(instructions="?")})

    assert isinstance(result, AskResult)
    assert result.usage == {"input_tokens": 7, "output_tokens": 3}
    assert result.elapsed_ms is not None
    assert result.elapsed_ms >= 0


def test_default_retry_targets_429_and_529_with_three_attempts():
    from typesymbolic.judge import _default_retry

    retry = _default_retry(typesafe_sdk)
    assert retry.max_retries == 2
    assert retry.http_statuses == {429, 529}
    assert retry.backoff_jitter == 0.0


async def test_jev_engine_retries_429_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx2.Response(429, json={"error": {"message": "slow down"}})
        return _ok_response()

    retry = typesafe_sdk.RetryPolicy(max_retries=2, http_statuses={429, 529}, backoff_initial=0.0, backoff_jitter=0.0)
    engine = JevEngine(api_key="fake-key", transport=httpx2.MockTransport(handler), retry=retry)

    result = await engine.ask_all({}, {"q": Noul(instructions="?")})

    assert calls["n"] == 2
    assert result.model_revision == "jev-latest"


async def test_jev_engine_does_not_retry_a_non_retryable_status():
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        return httpx2.Response(400, json={"error": {"message": "bad request"}})

    retry = typesafe_sdk.RetryPolicy(max_retries=2, http_statuses={429, 529}, backoff_initial=0.0, backoff_jitter=0.0)
    engine = JevEngine(api_key="fake-key", transport=httpx2.MockTransport(handler), retry=retry)

    with pytest.raises(JudgeError):
        await engine.ask_all({}, {"q": Noul(instructions="?")})

    assert calls["n"] == 1


async def test_wire_body_is_exactly_model_state_questions_for_noul():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return _ok_response()

    engine = JevEngine(api_key="fake-key", model="jev-1.13.0", transport=httpx2.MockTransport(handler))
    await engine.ask_all({"finding": "a"}, {"q": Noul(instructions="is this safe?")})

    assert captured["body"] == {
        "model": "jev-1.13.0",
        "state": {"finding": "a"},
        "questions": {"q": {"type": "noul", "instructions": "is this safe?"}},
    }


async def test_wire_body_keeps_undescribed_choice_labels_as_null():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return _ok_response()

    engine = JevEngine(api_key="fake-key", model="jev-1.13.0", transport=httpx2.MockTransport(handler))
    question = Choice(instructions="pick one", criteria={"pkg.a": None, "pkg.b": "described"})
    await engine.ask_all({}, {"q": question})

    assert captured["body"]["questions"]["q"] == {
        "type": "choice",
        "instructions": "pick one",
        "criteria": {"pkg.a": None, "pkg.b": "described"},
    }


async def test_wire_body_score_shape_is_the_ordered_criteria_list():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return _ok_response()

    engine = JevEngine(api_key="fake-key", model="jev-1.13.0", transport=httpx2.MockTransport(handler))
    question = Score(instructions="rate it", criteria=["low", "high"])
    await engine.ask_all({}, {"q": question})

    assert captured["body"]["questions"]["q"] == {
        "type": "score",
        "instructions": "rate it",
        "criteria": ["low", "high"],
    }
