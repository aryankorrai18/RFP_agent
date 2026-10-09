"""ClaudeLLM against a fake SDK client: request shape and error mapping, no network."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import ValidationError

from rfp_assistant.config import Settings
from rfp_assistant.providers.base import FALLBACK_BETA, ClaudeLLM, LLMError
from rfp_assistant.parsing.parser import ParsedDocument
from rfp_assistant.providers.errors import reason_of
from rfp_assistant.schemas import DraftResult, ExtractionResult, Fact, Requirement

DOC = ParsedDocument(filename="rfp.txt", kind="text", text="3.1 Do you support SSO?")
FACTS = [Fact(id="FACT-001", topic="SSO", statement="We support SAML 2.0 SSO.")]
REQ = Requirement(id="REQ-001", section="Security", question="Do you support SSO?", mandatory=True, word_limit=50, reference="3.1")


def response(parsed, stop_reason="end_turn", model="claude-opus-5", stop_details=None):
    usage = SimpleNamespace(input_tokens=12, output_tokens=7, cache_read_input_tokens=None, cache_creation_input_tokens=3)
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason, model=model, usage=usage, stop_details=stop_details)


def validation_error() -> ValidationError:
    try:
        ExtractionResult.model_validate_json("{truncated")
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
    return ClaudeLLM(Settings(), client=client), messages


def http_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(message="nope", response=httpx2.Response(status, request=request), body=None)


def test_extraction_request_shape():
    parsed = ExtractionResult(requirements=[])
    llm, messages = llm_with(response(parsed))
    result = asyncio.run(llm.extract_requirements(DOC))

    call = messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_format"] is ExtractionResult
    assert call["output_config"] == {"effort": "low"}
    assert call["thinking"] == {"type": "adaptive"}
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert "<rfp_document>" in call["messages"][0]["content"][0]["text"]
    assert result.output is parsed
    assert (result.usage.input_tokens, result.usage.cache_read_input_tokens, result.usage.cache_creation_input_tokens) == (12, 0, 3)


def test_scanned_pdf_is_sent_as_a_document_block():
    llm, messages = llm_with(response(ExtractionResult(requirements=[])))
    asyncio.run(llm.extract_requirements(ParsedDocument("scan.pdf", "pdf", "", pdf_bytes=b"%PDF-1.4")))
    block = messages.calls[0]["messages"][0]["content"][0]
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"


def test_draft_request_uses_cached_fact_sheet_and_draft_effort():
    llm, messages = llm_with(response(DraftResult(answer="", claims=[], unsupported_claims=[], needs_sme=True, sme_question="?")))
    asyncio.run(llm.draft_answer("Test Co", FACTS, REQ))
    call = messages.calls[0]
    assert call["output_format"] is DraftResult
    assert call["output_config"] == {"effort": "medium"}
    assert call["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert "[FACT-001]" in call["system"][1]["text"]
    assert "Word limit: 50 words" in call["messages"][0]["content"]


def test_malformed_output_is_retried_once_then_succeeds():
    parsed = ExtractionResult(requirements=[])
    llm, messages = llm_with(validation_error(), response(parsed))
    assert asyncio.run(llm.extract_requirements(DOC)).output is parsed
    assert len(messages.calls) == 2


def test_malformed_output_twice_is_an_error():
    llm, _ = llm_with(validation_error(), response(None))
    with pytest.raises(LLMError) as error:
        asyncio.run(llm.extract_requirements(DOC))
    assert error.value.kind == "malformed"


def test_refusal_is_not_retried():
    details = SimpleNamespace(category="cyber", explanation="...")
    llm, messages = llm_with(response(None, stop_reason="refusal", stop_details=details))
    with pytest.raises(LLMError) as error:
        asyncio.run(llm.draft_answer("Test Co", FACTS, REQ))
    assert error.value.kind == "refused"
    assert "cyber" in error.value.message
    assert len(messages.calls) == 1


def test_fallback_model_is_reported():
    parsed = ExtractionResult(requirements=[])
    llm, _ = llm_with(response(parsed, model="claude-opus-4-8"))
    assert asyncio.run(llm.extract_requirements(DOC)).model == "claude-opus-4-8"


def http_error_with(cls, status, message):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(message=message, response=httpx2.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("exception", "kind", "reason"),
    [
        (http_error(anthropic.AuthenticationError, 401), "auth", "auth"),
        (http_error(anthropic.PermissionDeniedError, 403), "auth", "auth"),
        (http_error(anthropic.BadRequestError, 400), "bad_request", "unknown"),
        (http_error_with(anthropic.BadRequestError, 400, "Your credit balance is too low to access the Anthropic API."),
         "api_error", "quota_exhausted"),
        (http_error_with(anthropic.BadRequestError, 400, "prompt is too long: 250000 tokens > 200000 maximum"),
         "bad_request", "too_large"),
        (http_error(anthropic.NotFoundError, 404), "bad_request", "model_unavailable"),
        (http_error(anthropic.RateLimitError, 429), "api_error", "rate_limited"),
        (http_error(anthropic.InternalServerError, 500), "api_error", "provider_error"),
        (http_error(anthropic.OverloadedError, 529), "api_error", "provider_error"),
        (TypeError("Could not resolve authentication method. Expected one of api_key..."), "auth", "auth"),
    ],
)
def test_sdk_errors_are_mapped(exception, kind, reason):
    llm, _ = llm_with(exception)
    with pytest.raises(LLMError) as error:
        asyncio.run(llm.extract_requirements(DOC))
    assert (error.value.kind, error.value.reason) == (kind, reason)
    assert reason_of(error.value.message) == reason  # a draft or project stores only the message


def test_connection_errors_are_mapped():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    llm, _ = llm_with(anthropic.APIConnectionError(request=request))
    with pytest.raises(LLMError) as error:
        asyncio.run(llm.extract_requirements(DOC))
    assert error.value.kind == "api_error" and error.value.reason == reason_of(error.value.message) == "unreachable"


def test_unrelated_type_errors_are_not_swallowed():
    llm, _ = llm_with(TypeError("unexpected keyword argument 'foo'"))
    with pytest.raises(TypeError):
        asyncio.run(llm.extract_requirements(DOC))


def test_pair_extraction_request_shape():
    from rfp_assistant.schemas import PairsResult

    llm, messages = llm_with(response(PairsResult(pairs=[])))
    asyncio.run(llm.extract_pairs(DOC))
    call = messages.calls[0]
    assert call["output_format"] is PairsResult
    assert "<proposal_document>" in call["messages"][0]["content"][0]["text"]
    assert "copied verbatim" in call["system"] or "verbatim" in call["system"]


def test_draft_with_past_answers_uses_v1_rules():
    from rfp_assistant.schemas import PastAnswer

    llm, messages = llm_with(response(DraftResult(answer="", claims=[], unsupported_claims=[], needs_sme=True, sme_question="?")))
    asyncio.run(llm.draft_answer("Test Co", FACTS, REQ, [PastAnswer(id="ANS-0001", question="q", answer="a")], "Shorter"))
    call = messages.calls[0]
    assert "<past_answers>" in call["system"][0]["text"]
    assert "[ANS-0001]" in call["messages"][0]["content"]
    assert "Shorter" in call["messages"][0]["content"]
