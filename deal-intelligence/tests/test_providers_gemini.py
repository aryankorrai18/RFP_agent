"""GeminiLLM against a fake google-genai client: request shape and error mapping, no network."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types
from pydantic import BaseModel

from deal_intelligence.config import Settings
from deal_intelligence.providers.base import LLMError
from deal_intelligence.providers.errors import reason_of
from deal_intelligence.providers.gemini import GeminiLLM, json_schema_for

SETTINGS = replace(Settings(), provider="gemini", model="gemini-3.5-flash-lite")


class Risk(BaseModel):
    label: str
    severity: int


class Verdict(BaseModel):
    answer: str
    risks: list[Risk]


VERDICT_JSON = json.dumps({"answer": "Ready to close.", "risks": [{"label": "pricing", "severity": 2}]})


def response(text, finish=types.FinishReason.STOP, block_reason=None, model_version="gemini-3.5-flash-lite"):
    usage = SimpleNamespace(prompt_token_count=40, candidates_token_count=10, thoughts_token_count=5, cached_content_token_count=None)
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish)],
        prompt_feedback=SimpleNamespace(block_reason=block_reason) if block_reason else None,
        usage_metadata=usage,
        model_version=model_version,
    )


class FakeModels:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def llm_with(*outcomes) -> tuple[GeminiLLM, FakeModels]:
    models = FakeModels(*outcomes)
    return GeminiLLM(SETTINGS, client=SimpleNamespace(aio=SimpleNamespace(models=models))), models


def call(llm: GeminiLLM, **overrides):
    kwargs = dict(purpose="brief", output_format=Verdict, system="Be brief.", user="Summarise the deal.")
    return asyncio.run(llm.structured(**{**kwargs, **overrides}))


def client_error(code, message, status="INVALID_ARGUMENT"):
    return errors.ClientError(code, {"error": {"code": code, "message": message, "status": status}})


def test_request_shape_and_result():
    llm, models = llm_with(response(VERDICT_JSON))
    result = call(llm, effort="low", max_tokens=1234)

    sent = models.calls[0]
    config = sent["config"]
    assert sent["model"] == "gemini-3.5-flash-lite"
    assert sent["contents"] == ["Summarise the deal."]
    assert config.system_instruction == "Be brief."
    assert config.response_mime_type == "application/json"
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert config.max_output_tokens == 1234
    assert "$ref" not in json.dumps(config.response_json_schema)
    assert result.output.risks[0].label == "pricing"
    assert result.model == "gemini-3.5-flash-lite"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (40, 15)


@pytest.mark.parametrize(("effort", "level"), [
    ("medium", types.ThinkingLevel.MEDIUM), ("high", types.ThinkingLevel.HIGH),
    ("xhigh", types.ThinkingLevel.HIGH), ("max", types.ThinkingLevel.HIGH),
])
def test_effort_maps_to_thinking_level(effort, level):
    llm, models = llm_with(response(VERDICT_JSON))
    call(llm, effort=effort)
    assert models.calls[0]["config"].thinking_config.thinking_level == level


def test_defaults_are_medium_thinking_and_4000_tokens():
    llm, models = llm_with(response(VERDICT_JSON))
    call(llm)
    config = models.calls[0]["config"]
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM and config.max_output_tokens == 4000


def test_temperature_is_unset_by_default_and_passed_through_when_given():
    llm, models = llm_with(response(VERDICT_JSON))
    call(llm)
    assert models.calls[0]["config"].temperature is None  # Gemini keeps its own sampling

    llm, models = llm_with(response(VERDICT_JSON))
    call(llm, temperature=0)
    assert models.calls[0]["config"].temperature == 0  # the memory comparison pins it down


def test_schema_has_no_refs():
    schema = json_schema_for(Verdict)
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text
    assert schema["properties"]["risks"]["items"]["properties"]["severity"]["type"] == "integer"


def test_malformed_json_is_retried_once():
    llm, models = llm_with(response("{truncated"), response(VERDICT_JSON))
    assert call(llm).output.answer
    assert len(models.calls) == 2


def test_wrong_shape_is_malformed():
    llm, models = llm_with(response('{"answer": 1}'), response('{"nope": true}'))
    with pytest.raises(LLMError) as error:
        call(llm)
    assert error.value.kind == "malformed" and len(models.calls) == 2


def test_malformed_twice_is_an_error():
    llm, _ = llm_with(response("{bad"), response(None, finish=types.FinishReason.MAX_TOKENS))
    with pytest.raises(LLMError) as error:
        call(llm)
    assert error.value.kind == "malformed" and error.value.reason == "malformed"
    assert "MAX_TOKENS" in error.value.message
    assert reason_of(error.value.message) == "malformed"


@pytest.mark.parametrize("finish", [types.FinishReason.SAFETY, types.FinishReason.PROHIBITED_CONTENT])
def test_safety_finish_is_a_refusal_and_not_retried(finish):
    llm, models = llm_with(response(None, finish=finish))
    with pytest.raises(LLMError) as error:
        call(llm)
    assert error.value.kind == "refused" and reason_of(error.value.message) == "refused"
    assert len(models.calls) == 1


def test_blocked_prompt_is_a_refusal():
    llm, _ = llm_with(response(None, block_reason=types.BlockedReason.SAFETY))
    with pytest.raises(LLMError) as error:
        call(llm)
    assert error.value.kind == "refused" and reason_of(error.value.message) == "refused"


def quota_error(quota_id, message="You exceeded your current quota, please check your plan and billing details."):
    """A 429 shaped like Google's: the quota that was hit is in the details, not the message."""
    return errors.ClientError(429, {"error": {
        "code": 429, "message": message, "status": "RESOURCE_EXHAUSTED",
        "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                     "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                     "quotaId": quota_id}]},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}],
    }})


@pytest.mark.parametrize(
    ("exception", "kind", "reason", "hint"),
    [
        (client_error(400, "API key not valid. Please pass a valid API key."), "auth", "auth", "API key"),
        (client_error(401, "Unauthenticated", "UNAUTHENTICATED"), "auth", "auth", ""),
        (client_error(403, "Permission denied", "PERMISSION_DENIED"), "auth", "auth", ""),
        (client_error(429, "Resource has been exhausted", "RESOURCE_EXHAUSTED"), "api_error", "rate_limited", "DEAL_CONCURRENCY"),
        (quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"), "api_error", "rate_limited", "DEAL_CONCURRENCY"),
        (quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier"), "api_error", "quota_exhausted", "gemini-3.5-flash-lite"),
        (quota_error("GenerateContentInputTokensPerModelPerMinute-FreeTier",
                     "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0"), "api_error", "quota_exhausted", ""),
        (client_error(404, "models/nope is not found", "NOT_FOUND"), "bad_request", "model_unavailable", "DEAL_MODEL"),
        (client_error(400, "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576)."),
         "bad_request", "too_large", ""),
        (client_error(400, "Invalid thinking level", "INVALID_ARGUMENT"), "bad_request", "unknown", ""),
        (errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}), "api_error", "provider_error", ""),
        (httpx.ConnectError("connection refused"), "api_error", "unreachable", "reach"),
    ],
)
def test_errors_are_mapped(exception, kind, reason, hint):
    llm, _ = llm_with(exception)
    with pytest.raises(LLMError) as error:
        call(llm)
    assert (error.value.kind, error.value.reason) == (kind, reason)
    assert hint in error.value.message
    assert reason_of(error.value.message) == reason  # stored errors keep only the message


def test_missing_key_is_an_auth_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError) as error:
        call(GeminiLLM(SETTINGS))
    assert error.value.kind == "auth" and "GEMINI_API_KEY" in error.value.message
    assert reason_of(error.value.message) == "auth"
