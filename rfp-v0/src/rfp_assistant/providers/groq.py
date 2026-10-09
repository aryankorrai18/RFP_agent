"""The AI calls on Groq's OpenAI-compatible API (plain httpx, no extra SDK).

Same contract as ClaudeLLM and GeminiLLM: returns LLMResult, raises LLMError. Differences:
- Structured output uses JSON mode. The JSON Schema goes in the system prompt and the reply is
  validated here with Pydantic, so it works on every Groq text model.
- 429 (rate limit) is retried after the wait Groq asks for; the free tier has tight per-minute
  token limits, so a long RFP can still hit them.
- Groq can't read scanned PDFs, so those are reported instead of sent.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, NoReturn, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from . import prompts
from ..config import PROVIDER_KEY_VARS, Settings
from .base import JUDGE_MAX_TOKENS, MALFORMED_ATTEMPTS, LLMError, LLMResult, TokenUsage
from .gemini import json_schema_for
from .errors import PHRASE, quota_is_hard
from ..parsing.parser import ParsedDocument
from ..schemas import DraftResult, ExtractionResult, Fact, JudgeResult, PairsResult, PastAnswer, Requirement

T = TypeVar("T", bound=BaseModel)

BASE_URL = "https://api.groq.com/openai/v1"
# Groq's per-minute limits count the output you ask for, so ask for less than Gemini.
EXTRACTION_MAX_TOKENS = 8000
PAIRS_MAX_TOKENS = 8000
DRAFT_MAX_TOKENS = 4000
RATE_LIMIT_ATTEMPTS = 5
MAX_WAIT_SECONDS = 60.0
REQUEST_TIMEOUT = 120.0
NON_TEXT_TAGS = ("whisper", "tts", "guard", "orpheus", "embed")


def api_key() -> str | None:
    for name in PROVIDER_KEY_VARS["groq"]:
        if os.environ.get(name):
            return os.environ[name]
    return None


class GroqLLM:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.model = settings.model
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=REQUEST_TIMEOUT)
        return self._client

    async def extract_requirements(self, document: ParsedDocument) -> LLMResult[ExtractionResult]:
        _text_only(document)
        return await self._generate(
            purpose="requirement extraction", output_format=ExtractionResult, system=prompts.EXTRACTION_SYSTEM,
            user=prompts.extraction_text(document), max_tokens=EXTRACTION_MAX_TOKENS,
        )

    async def extract_pairs(self, document: ParsedDocument) -> LLMResult[PairsResult]:
        _text_only(document)
        return await self._generate(
            purpose="pair extraction", output_format=PairsResult, system=prompts.PAIRS_SYSTEM,
            user=prompts.pairs_text(document), max_tokens=PAIRS_MAX_TOKENS,
        )

    async def draft_answer(
        self,
        company: str,
        facts: list[Fact],
        requirement: Requirement,
        past_answers: list[PastAnswer] | None = None,
        instructions: str | None = None,
        *,
        temperature: float | None = None,
    ) -> LLMResult[DraftResult]:
        return await self._generate(
            purpose=f"drafting {requirement.id}", output_format=DraftResult,
            system=prompts.drafting_system_text(company, facts),
            user=prompts.drafting_user_message(requirement, past_answers, instructions),
            max_tokens=DRAFT_MAX_TOKENS,
            temperature=0.2 if temperature is None else temperature,
        )

    async def judge(self, system: str, message: str) -> LLMResult[JudgeResult]:
        return await self._generate(
            purpose="judging a draft pair", output_format=JudgeResult, system=system, user=message,
            max_tokens=JUDGE_MAX_TOKENS, temperature=0,
        )

    async def _generate(
        self, *, purpose: str, output_format: type[T], system: str, user: str, max_tokens: int, temperature: float = 0.2
    ) -> LLMResult[T]:
        schema = json.dumps(json_schema_for(output_format))
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"{system}\n\nReply with one JSON object that matches this JSON "
                                              f"Schema, and nothing else:\n{schema}"},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        last_error: LLMError | None = None
        for _ in range(MALFORMED_ATTEMPTS):
            data = await self._post(purpose, body)
            if data is None:  # Groq itself rejected the model's JSON: retry like any malformed output
                last_error = LLMError(
                    "malformed", f"{purpose}: {PHRASE['malformed']}: the model did not produce valid JSON", reason="malformed"
                )
                continue
            choice = (data.get("choices") or [{}])[0]
            finish = choice.get("finish_reason")
            text = (choice.get("message") or {}).get("content") or ""
            if not text.strip():
                last_error = LLMError(
                    "malformed", f"{purpose}: {PHRASE['malformed']}: empty response (finish reason: {finish})", reason="malformed"
                )
                continue
            try:
                parsed = output_format.model_validate_json(text)
            except ValidationError:
                last_error = LLMError(
                    "malformed", f"{purpose}: {PHRASE['malformed']}: output did not match the expected schema (finish reason: {finish})",
                    reason="malformed",
                )
                continue
            usage = data.get("usage") or {}
            return LLMResult(
                output=parsed,
                model=data.get("model") or self.model,
                usage=TokenUsage(
                    input_tokens=usage.get("prompt_tokens", 0) or 0,
                    output_tokens=usage.get("completion_tokens", 0) or 0,
                ),
            )
        assert last_error is not None
        raise last_error

    async def _post(self, purpose: str, body: dict[str, Any]) -> dict[str, Any] | None:
        """The JSON reply, or None when Groq rejected the model's JSON (the caller retries)."""
        key = api_key()
        if not key:
            raise LLMError("auth", f"Groq {PHRASE['auth']}: no key found. Set GROQ_API_KEY in .env (see README).", reason="auth")
        for attempt in range(RATE_LIMIT_ATTEMPTS):
            try:
                response = await self.client.post("/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})
            except httpx.HTTPError as exc:
                raise LLMError(
                    "api_error", f"{purpose}: {PHRASE['unreachable']} the Groq API ({type(exc).__name__})", reason="unreachable"
                ) from exc
            # A daily limit won't clear by waiting a few seconds, so only per-minute limits are retried.
            if response.status_code == 429 and attempt < RATE_LIMIT_ATTEMPTS - 1 and not quota_is_hard(_error_code(response)[1]):
                await asyncio.sleep(_wait_seconds(response, attempt))
                continue
            if response.status_code == 200:
                return response.json()
            if _error_code(response)[0] == "json_validate_failed":
                return None
            _raise_for(purpose, response)
        raise AssertionError("unreachable")


def _text_only(document: ParsedDocument) -> None:
    if document.pdf_bytes is not None:
        raise LLMError(
            "bad_request",
            f"Groq {PHRASE['unsupported_input']}: Groq can't read scanned PDFs. Use a text-based file, "
            "or switch to a Gemini or Claude model for this one.",
            reason="unsupported_input",
        )


def _wait_seconds(response: httpx.Response, attempt: int) -> float:
    try:
        return min(MAX_WAIT_SECONDS, max(1.0, float(response.headers.get("retry-after", ""))))
    except ValueError:
        return min(MAX_WAIT_SECONDS, 2.0 * 2**attempt)


def _error_code(response: httpx.Response) -> tuple[str, str]:
    try:
        error = response.json().get("error") or {}
    except ValueError:
        return "", response.text[:200]
    return str(error.get("code") or ""), str(error.get("message") or "")[:300]


def _raise_for(purpose: str, response: httpx.Response) -> NoReturn:
    code, message = _error_code(response)
    status = response.status_code
    lowered = message.lower()
    if status in (401, 403) or code == "invalid_api_key":
        raise LLMError("auth", f"{purpose}: Groq {PHRASE['auth']} ({message})", reason="auth")
    if status == 429:
        if quota_is_hard(message):
            raise LLMError("api_error", f"{purpose}: Groq {PHRASE['quota_exhausted']} for this model ({message})", reason="quota_exhausted")
        raise LLMError(
            "api_error",
            f"{purpose}: Groq {PHRASE['rate_limited']}, even after waiting ({message}). Free-tier token limits are low: "
            "set RFP_DRAFT_CONCURRENCY=1, pick a model with a higher limit, or wait a minute.",
            reason="rate_limited",
        )
    if status == 413 or code == "context_length_exceeded" or "reduce the length" in lowered:
        raise LLMError(
            "bad_request",
            f"{purpose}: Groq {PHRASE['too_large']}: larger than this model or plan allows ({message}). Pick a model with a larger limit.",
            reason="too_large",
        )
    if status == 404 or code in ("model_not_found", "model_decommissioned"):
        raise LLMError(
            "bad_request", f"{purpose}: Groq {PHRASE['model_unavailable']}; pick another with the model button ({message})",
            reason="model_unavailable",
        )
    if status >= 500:
        raise LLMError("api_error", f"{purpose}: Groq {PHRASE['provider_error']} {status} ({message})", reason="provider_error")
    raise LLMError("bad_request", f"{purpose}: Groq rejected the request ({status}: {message})", reason="unknown")


async def list_groq_models() -> list[tuple[str, int | None]]:
    """(model id, context window) for the text models this key can use. Metadata only, no tokens."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=20.0) as client:
        response = await client.get("/models", headers={"Authorization": f"Bearer {api_key()}"})
        response.raise_for_status()
    return [
        (m["id"], m.get("context_window"))
        for m in response.json().get("data", [])
        if m.get("active", True) and not any(tag in m["id"] for tag in NON_TEXT_TAGS)
    ]
