"""GroqLLM against a fake HTTP transport: request shape, retries and error mapping, no network."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

import backend.llm_groq as groq_module
from backend import config, model_choice
from backend.config import Settings
from backend.llm import LLMError
from backend.llm_groq import GroqLLM
from backend.parser import ParsedDocument
from backend.provider_errors import reason_of
from backend.schemas import Fact, Requirement

SETTINGS = replace(Settings(), provider="groq", model="llama-3.3-70b-versatile")
DOC = ParsedDocument(filename="rfp.txt", kind="text", text="3.1 Do you support SSO?")
FACTS = [Fact(id="FACT-001", topic="SSO", statement="We support SAML 2.0 SSO.")]
REQ = Requirement(id="REQ-001", section="Security", question="Do you support SSO?", mandatory=True, word_limit=50, reference="3.1")

EXTRACTION_JSON = json.dumps({"requirements": [
    {"section": "Security", "question": "Do you support SSO?", "mandatory": True, "word_limit": 50, "reference": "3.1"}
]})
DRAFT_JSON = json.dumps({
    "answer": "We support SAML 2.0 SSO.",
    "claims": [{"text": "SAML 2.0 SSO", "source_ids": ["FACT-001"]}],
    "unsupported_claims": [], "needs_sme": False, "sme_question": None,
})


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


@pytest.fixture(autouse=True)
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(groq_module.asyncio, "sleep", no_wait)


def test_extraction_sends_json_mode_with_the_schema_and_reads_usage():
    llm, seen = llm_with(ok(EXTRACTION_JSON))
    result = asyncio.run(llm.extract_requirements(DOC))
    assert result.output.requirements[0].question == "Do you support SSO?"
    assert result.usage.input_tokens == 50 and result.usage.output_tokens == 12
    body = json.loads(seen[0].content)
    assert body["response_format"] == {"type": "json_object"} and body["model"] == "llama-3.3-70b-versatile"
    assert "JSON Schema" in body["messages"][0]["content"] and "requirements" in body["messages"][0]["content"]
    assert seen[0].headers["authorization"] == "Bearer test-key"


def test_draft_answer_is_validated_against_the_schema():
    llm, _ = llm_with(ok(DRAFT_JSON))
    result = asyncio.run(llm.draft_answer("Larkspur", FACTS, REQ))
    assert result.output.claims[0].source_ids == ["FACT-001"]


def test_malformed_output_is_retried_once_then_reported():
    llm, seen = llm_with(ok("not json"), ok(DRAFT_JSON))
    assert asyncio.run(llm.draft_answer("Larkspur", FACTS, REQ)).output.answer
    assert len(seen) == 2
    llm, _ = llm_with(ok("nope"), ok("still nope"))
    with pytest.raises(LLMError) as caught:
        asyncio.run(llm.draft_answer("Larkspur", FACTS, REQ))
    assert caught.value.kind == "malformed"


def test_groq_rejecting_the_json_is_retried_like_malformed_output():
    llm, seen = llm_with(failure(400, "json_validate_failed", "bad json"), ok(DRAFT_JSON))
    assert asyncio.run(llm.draft_answer("Larkspur", FACTS, REQ)).output.answer
    assert len(seen) == 2


def test_rate_limit_waits_and_retries_then_explains():
    llm, seen = llm_with(failure(429, "rate_limit_exceeded", "slow down", {"retry-after": "3"}), ok(DRAFT_JSON))
    assert asyncio.run(llm.draft_answer("Larkspur", FACTS, REQ)).output.answer
    assert len(seen) == 2
    llm, seen = llm_with(*[failure(429, "rate_limit_exceeded", "TPM limit") for _ in range(groq_module.RATE_LIMIT_ATTEMPTS)])
    with pytest.raises(LLMError) as caught:
        asyncio.run(llm.draft_answer("Larkspur", FACTS, REQ))
    assert caught.value.kind == "api_error" and "RFP_DRAFT_CONCURRENCY" in caught.value.message
    assert caught.value.reason == "rate_limited" and reason_of(caught.value.message) == "rate_limited"
    assert len(seen) == groq_module.RATE_LIMIT_ATTEMPTS


@pytest.mark.parametrize("response, kind, reason", [
    (failure(401, "invalid_api_key", "Invalid API Key"), "auth", "auth"),
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
        asyncio.run(llm.draft_answer("Larkspur", FACTS, REQ))
    assert (caught.value.kind, caught.value.reason) == (kind, reason)
    assert reason_of(caught.value.message) == reason  # a draft stores only the message


def test_a_daily_limit_is_reported_at_once_not_retried():
    daily = "Rate limit reached for model `llama-3.3-70b-versatile` on tokens per day (TPD): Limit 100000, Used 99990"
    llm, seen = llm_with(failure(429, "rate_limit_exceeded", daily))
    with pytest.raises(LLMError) as caught:
        asyncio.run(llm.draft_answer("Larkspur", FACTS, REQ))
    assert caught.value.reason == "quota_exhausted" and reason_of(caught.value.message) == "quota_exhausted"
    assert len(seen) == 1  # waiting seconds can't clear a daily limit, so no retries were spent


def test_missing_key_and_scanned_pdf_are_reported_before_any_request(monkeypatch):
    llm, seen = llm_with()
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(LLMError) as caught:
        asyncio.run(llm.draft_answer("Larkspur", FACTS, REQ))
    assert caught.value.kind == "auth" and "GROQ_API_KEY" in caught.value.message
    assert reason_of(caught.value.message) == "auth"
    scanned = ParsedDocument(filename="scan.pdf", kind="pdf", text="", pdf_bytes=b"%PDF")
    with pytest.raises(LLMError) as caught:
        asyncio.run(llm.extract_requirements(scanned))
    assert "scanned PDFs" in caught.value.message and not seen
    assert caught.value.reason == reason_of(caught.value.message) == "unsupported_input"


def test_provider_resolution_and_defaults(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY", "RFP_LLM_PROVIDER", "RFP_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert config.resolve_provider(None) == "groq"  # only Groq's key is set
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert config.resolve_provider(None) == "gemini"  # Gemini is preferred over Groq
    assert config.resolve_provider("groq") == "groq"
    assert config.DEFAULT_MODEL["groq"] == "llama-3.3-70b-versatile" and config.DEFAULT_CONCURRENCY["groq"] == 2
    monkeypatch.setenv("RFP_LLM_PROVIDER", "groq")
    settings = Settings.from_env()
    assert settings.provider == "groq" and settings.model == "llama-3.3-70b-versatile" and settings.draft_concurrency == 2
    with pytest.raises(ValueError, match="groq"):
        config.resolve_provider("openai")


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
