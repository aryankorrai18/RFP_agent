"""ClaudeLLM against a fake SDK client: request shape and error mapping, no network."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import BaseModel, ValidationError

from deal_intelligence.config import Settings
from deal_intelligence.providers.base import LLMError
from deal_intelligence.providers.claude import FALLBACK_BETA, ClaudeLLM
from deal_intelligence.providers.errors import reason_of

SETTINGS = replace(Settings(), provider="anthropic", model="claude-opus-5")


class Verdict(BaseModel):
    answer: str
    score: int


def response(parsed, stop_reason="end_turn", model="claude-opus-5", stop_details=None):
    usage = SimpleNamespace(input_tokens=12, output_tokens=7, cache_read_input_tokens=None, cache_creation_input_tokens=3)
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason, model=model, usage=usage, stop_details=stop_details)


def validation_error() -> ValidationError:
    try:
        Verdict.model_validate_json("{truncated")
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


class FakeMessages:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def llm_with(*outcomes) -> tuple[ClaudeLLM, FakeMessages]:
    messages = FakeMessages(*outcomes)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return ClaudeLLM(SETTINGS, client=client), messages


def call(llm: ClaudeLLM, **overrides):
    kwargs = dict(purpose="brief", output_format=Verdict, system="Be brief.", user="Summarise the deal.")
    return asyncio.run(llm.structured(**{**kwargs, **overrides}))


def http_error(cls, status, message="nope"):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(message=message, response=httpx2.Response(status, request=request), body=None)


def test_request_shape_and_result():
    parsed = Verdict(answer="ok", score=3)
    llm, messages = llm_with(response(parsed))
    result = call(llm, effort="high", max_tokens=1234)

    sent = messages.calls[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["output_format"] is Verdict
    assert sent["output_config"] == {"effort": "high"}
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["betas"] == [FALLBACK_BETA]
    assert sent["fallbacks"] == "default"
    assert sent["max_tokens"] == 1234
    assert sent["system"] == "Be brief."
    assert sent["messages"] == [{"role": "user", "content": "Summarise the deal."}]
    assert result.output is parsed and result.model == "claude-opus-5"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (12, 7)
    assert (result.usage.cache_read_input_tokens, result.usage.cache_creation_input_tokens) == (0, 3)


def test_defaults_are_medium_effort_and_4000_tokens():
    llm, messages = llm_with(response(Verdict(answer="ok", score=1)))
    call(llm)
    assert messages.calls[0]["output_config"] == {"effort": "medium"} and messages.calls[0]["max_tokens"] == 4000


def test_temperature_is_accepted_and_ignored():
    llm, messages = llm_with(response(Verdict(answer="ok", score=1)))
    call(llm, temperature=0)
    assert "temperature" not in messages.calls[0]


def test_client_is_created_lazily():
    llm = ClaudeLLM(SETTINGS)
    assert llm._client is None and llm.model == "claude-opus-5"


def test_malformed_output_is_retried_once_then_succeeds():
    parsed = Verdict(answer="ok", score=2)
    llm, messages = llm_with(validation_error(), response(parsed))
    assert call(llm).output is parsed
    assert len(messages.calls) == 2


def test_malformed_output_twice_is_an_error():
    llm, messages = llm_with(validation_error(), response(None))
    with pytest.raises(LLMError) as error:
        call(llm)
    assert error.value.kind == "malformed" and error.value.reason == "malformed"
    assert reason_of(error.value.message) == "malformed"
    assert len(messages.calls) == 2


def test_refusal_is_not_retried():
    details = SimpleNamespace(category="cyber", explanation="...")
    llm, messages = llm_with(response(None, stop_reason="refusal", stop_details=details))
    with pytest.raises(LLMError) as error:
        call(llm)
    assert error.value.kind == "refused" and "cyber" in error.value.message
    assert reason_of(error.value.message) == "refused"
    assert len(messages.calls) == 1


def test_fallback_model_is_reported():
    llm, _ = llm_with(response(Verdict(answer="ok", score=1), model="claude-opus-4-8"))
    assert call(llm).model == "claude-opus-4-8"


@pytest.mark.parametrize(
    ("exception", "kind", "reason"),
    [
        (http_error(anthropic.AuthenticationError, 401), "auth", "auth"),
        (http_error(anthropic.PermissionDeniedError, 403), "auth", "auth"),
        (http_error(anthropic.BadRequestError, 400), "bad_request", "unknown"),
        (http_error(anthropic.BadRequestError, 400, "Your credit balance is too low to access the Anthropic API."),
         "api_error", "quota_exhausted"),
        (http_error(anthropic.BadRequestError, 400, "prompt is too long: 250000 tokens > 200000 maximum"),
         "bad_request", "too_large"),
        (http_error(anthropic.NotFoundError, 404), "bad_request", "model_unavailable"),
        (http_error(anthropic.RateLimitError, 429), "api_error", "rate_limited"),
        (http_error(anthropic.RateLimitError, 429, "You have hit your daily limit"), "api_error", "quota_exhausted"),
        (http_error(anthropic.InternalServerError, 500), "api_error", "provider_error"),
        (http_error(anthropic.OverloadedError, 529), "api_error", "provider_error"),
        (TypeError("Could not resolve authentication method. Expected one of api_key..."), "auth", "auth"),
    ],
)
def test_sdk_errors_are_mapped(exception, kind, reason):
    llm, _ = llm_with(exception)
    with pytest.raises(LLMError) as error:
        call(llm)
    assert (error.value.kind, error.value.reason) == (kind, reason)
    assert reason_of(error.value.message) == reason  # stored errors keep only the message


def test_connection_errors_are_mapped():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    llm, _ = llm_with(anthropic.APIConnectionError(request=request))
    with pytest.raises(LLMError) as error:
        call(llm)
    assert error.value.kind == "api_error" and error.value.reason == reason_of(error.value.message) == "unreachable"


def test_unrelated_type_errors_are_not_swallowed():
    llm, _ = llm_with(TypeError("unexpected keyword argument 'foo'"))
    with pytest.raises(TypeError):
        call(llm)
