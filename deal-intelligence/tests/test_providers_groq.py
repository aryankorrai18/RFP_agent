"""GroqLLM against a fake HTTP transport: request shape, retries and error mapping, no network."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from pydantic import BaseModel

import deal_intelligence.providers.groq as groq_module
from deal_intelligence import config
from deal_intelligence.config import Settings
from deal_intelligence.providers import model_choice
from deal_intelligence.providers.base import LLMError
from deal_intelligence.providers.errors import reason_of
from deal_intelligence.providers.groq import GroqLLM

SETTINGS = replace(Settings(), provider="groq", model="llama-3.3-70b-versatile")


class Risk(BaseModel):
    label: str
    severity: int


class Verdict(BaseModel):
    answer: str
    risks: list[Risk]


VERDICT_JSON = json.dumps({"answer": "Ready to close.", "risks": [{"label": "pricing", "severity": 2}]})


def ok(content, model="llama-3.3-70b-versatile"):
    return httpx.Response(200, json={
        "model": model, "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 12},
    })


def failure(status, code="", message="", headers=None):
    return httpx.Response(status, json={"error": {"code": code, "message": message}}, headers=headers)


def llm_with(*responses) -> tuple[GroqLLM, list[httpx.Request]]:
    queue, seen = list(responses), []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    client = httpx.AsyncClient(base_url=groq_module.BASE_URL, transport=httpx.MockTransport(handler))
    return GroqLLM(SETTINGS, client), seen


def call(llm: GroqLLM, **overrides):
    kwargs = dict(purpose="brief", output_format=Verdict, system="Be brief.", user="Summarise the deal.")
    return asyncio.run(llm.structured(**{**kwargs, **overrides}))


@pytest.fixture(autouse=True)
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(groq_module.asyncio, "sleep", no_wait)


def test_sends_json_mode_with_the_schema_and_reads_usage():
    llm, seen = llm_with(ok(VERDICT_JSON))
    result = call(llm, max_tokens=1234)
    assert result.output.risks[0].label == "pricing" and result.model == "llama-3.3-70b-versatile"
    assert result.usage.input_tokens == 50 and result.usage.output_tokens == 12
    body = json.loads(seen[0].content)
    assert body["response_format"] == {"type": "json_object"} and body["model"] == "llama-3.3-70b-versatile"
    assert body["max_completion_tokens"] == 1234
    system, user = body["messages"]
    assert system["role"] == "system" and system["content"].startswith("Be brief.")
    assert "JSON Schema" in system["content"] and "risks" in system["content"]
    assert user == {"role": "user", "content": "Summarise the deal."}
    assert seen[0].headers["authorization"] == "Bearer test-key"


def test_temperature_defaults_low_and_is_passed_through_when_given():
    llm, seen = llm_with(ok(VERDICT_JSON))
    call(llm)
    assert json.loads(seen[0].content)["temperature"] == 0.2

    llm, seen = llm_with(ok(VERDICT_JSON))
    call(llm, temperature=0)
    assert json.loads(seen[0].content)["temperature"] == 0


def test_effort_is_accepted_and_not_sent():
    llm, seen = llm_with(ok(VERDICT_JSON))
    call(llm, effort="high")
    assert "effort" not in json.loads(seen[0].content)


def test_malformed_output_is_retried_once_then_reported():
    llm, seen = llm_with(ok("not json"), ok(VERDICT_JSON))
    assert call(llm).output.answer
    assert len(seen) == 2
    llm, _ = llm_with(ok("nope"), ok("still nope"))
    with pytest.raises(LLMError) as caught:
        call(llm)
    assert caught.value.kind == "malformed" and reason_of(caught.value.message) == "malformed"


def test_empty_and_wrong_shape_are_malformed():
    llm, seen = llm_with(ok(""), ok('{"answer": 1}'))
    with pytest.raises(LLMError) as caught:
        call(llm)
    assert caught.value.kind == "malformed" and len(seen) == 2


def test_groq_rejecting_the_json_is_retried_like_malformed_output():
    llm, seen = llm_with(failure(400, "json_validate_failed", "bad json"), ok(VERDICT_JSON))
    assert call(llm).output.answer
    assert len(seen) == 2
    llm, _ = llm_with(failure(400, "json_validate_failed", "bad json"), failure(400, "json_validate_failed", "bad json"))
    with pytest.raises(LLMError) as caught:
        call(llm)
    assert caught.value.kind == "malformed"


def test_rate_limit_waits_and_retries_then_explains(monkeypatch):
    waits: list[float] = []

    async def record_wait(seconds):
        waits.append(seconds)

    monkeypatch.setattr(groq_module.asyncio, "sleep", record_wait)
    llm, seen = llm_with(failure(429, "rate_limit_exceeded", "slow down", {"retry-after": "3"}), ok(VERDICT_JSON))
    assert call(llm).output.answer
    assert len(seen) == 2 and waits == [3.0]  # waits as long as Groq asked
    llm, seen = llm_with(*[failure(429, "rate_limit_exceeded", "TPM limit") for _ in range(groq_module.RATE_LIMIT_ATTEMPTS)])
    with pytest.raises(LLMError) as caught:
        call(llm)
    assert caught.value.kind == "api_error" and "DEAL_CONCURRENCY" in caught.value.message
    assert caught.value.reason == "rate_limited" and reason_of(caught.value.message) == "rate_limited"
    assert len(seen) == groq_module.RATE_LIMIT_ATTEMPTS


@pytest.mark.parametrize("response, kind, reason", [
    (failure(401, "invalid_api_key", "Invalid API Key"), "auth", "auth"),
    (failure(403, "", "forbidden"), "auth", "auth"),
    (failure(404, "model_not_found", "no such model"), "bad_request", "model_unavailable"),
    (failure(400, "model_decommissioned", "retired"), "bad_request", "model_unavailable"),
    (failure(413, "request_too_large", "too big"), "bad_request", "too_large"),
    (failure(400, "context_length_exceeded", "Please reduce the length of the messages"), "bad_request", "too_large"),
    (failure(503, "", "overloaded"), "api_error", "provider_error"),
    (failure(400, "invalid_request_error", "something else"), "bad_request", "unknown"),
])
def test_errors_map_to_reasons_that_survive_being_stored(response, kind, reason):
    llm, _ = llm_with(response)
    with pytest.raises(LLMError) as caught:
        call(llm)
    assert (caught.value.kind, caught.value.reason) == (kind, reason)
    assert reason_of(caught.value.message) == reason  # stored errors keep only the message


def test_a_daily_limit_is_reported_at_once_not_retried():
    daily = "Rate limit reached for model `llama-3.3-70b-versatile` on tokens per day (TPD): Limit 100000, Used 99990"
    llm, seen = llm_with(failure(429, "rate_limit_exceeded", daily))
    with pytest.raises(LLMError) as caught:
        call(llm)
    assert caught.value.reason == "quota_exhausted" and reason_of(caught.value.message) == "quota_exhausted"
    assert len(seen) == 1  # waiting seconds can't clear a daily limit


def test_connection_errors_are_mapped():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.AsyncClient(base_url=groq_module.BASE_URL, transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError) as caught:
        call(GroqLLM(SETTINGS, client))
    assert caught.value.kind == "api_error" and caught.value.reason == reason_of(caught.value.message) == "unreachable"


def test_missing_key_is_reported_before_any_request(monkeypatch):
    llm, seen = llm_with()
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(LLMError) as caught:
        call(llm)
    assert caught.value.kind == "auth" and "GROQ_API_KEY" in caught.value.message
    assert reason_of(caught.value.message) == "auth" and not seen


def test_provider_resolution_and_defaults(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DEAL_LLM_PROVIDER", "DEAL_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert config.resolve_provider(None) == "groq"  # only Groq's key is set
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert config.resolve_provider(None) == "gemini"  # Gemini is preferred over Groq
    assert config.DEFAULT_MODEL["groq"] == "llama-3.3-70b-versatile" and config.DEFAULT_CONCURRENCY["groq"] == 2


def test_model_list_is_sorted_and_slashed_ids_are_valid(monkeypatch):
    async def fake_list():
        return [("openai/gpt-oss-120b", 131072), ("llama-3.3-70b-versatile", 131072)]

    monkeypatch.setattr(groq_module, "list_groq_models", fake_list)
    model_choice._cache.clear()
    options, error = asyncio.run(model_choice.list_models("groq"))
    assert error is None and [o.id for o in options] == ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
    assert model_choice.MODEL_NAME.match("openai/gpt-oss-120b")
    assert model_choice.MODEL_NAME.match("meta-llama/llama-4-scout-17b-16e-instruct")
    model_choice._cache.clear()


def test_list_groq_models_filters_non_text_and_inactive(monkeypatch):
    payload = {"data": [
        {"id": "llama-3.3-70b-versatile", "context_window": 131072, "active": True},
        {"id": "whisper-large-v3", "context_window": 448},
        {"id": "meta-llama/llama-guard-4-12b", "context_window": 131072},
        {"id": "old-model", "context_window": 8192, "active": False},
    ]}
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        return real_client(*args, **{**kwargs, "transport": httpx.MockTransport(lambda request: httpx.Response(200, json=payload))})

    monkeypatch.setattr(groq_module.httpx, "AsyncClient", factory)
    assert asyncio.run(groq_module.list_groq_models()) == [("llama-3.3-70b-versatile", 131072)]
