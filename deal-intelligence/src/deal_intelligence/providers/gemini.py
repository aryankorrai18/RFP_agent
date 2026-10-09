"""Google Gemini adapter for the structured-output call.

Same contract as ClaudeLLM in claude.py: returns LLMResult, raises LLMError. Differences:
- Structured output uses `response_json_schema` and is validated here with Pydantic.
- Effort maps to Gemini's `thinking_level` (low / medium / high).
- 429 and 5xx responses are retried with exponential backoff inside the SDK, because the free
  tier's per-minute limits are easy to hit when several briefs run at once.
- There is no server-side refusal fallback; a safety block is reported as `refused`.
"""

from __future__ import annotations

import copy
import json
from typing import Any, TypeVar

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from ..config import Settings, has_key
from .base import MALFORMED_ATTEMPTS, LLMError, LLMResult, TokenUsage
from .errors import PHRASE, quota_is_hard

T = TypeVar("T", bound=BaseModel)

THINKING_LEVEL = {
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
    "xhigh": types.ThinkingLevel.HIGH,
    "max": types.ThinkingLevel.HIGH,
}

RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=6, initial_delay=2.0, max_delay=60.0, http_status_codes=[429, 500, 502, 503, 504]
)

# Finish reasons that mean the model declined or was blocked, not that output was malformed.
NO_KEY = f"Gemini {PHRASE['auth']}: no key found. Set GEMINI_API_KEY in .env (see README)."

REFUSAL_FINISH_REASONS = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION"}


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic's JSON schema with every $ref inlined, so no $defs lookups are needed."""
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = resolve(copy.deepcopy(definitions[node["$ref"].rsplit("/", 1)[-1]]))
                siblings = {k: resolve(v) for k, v in node.items() if k != "$ref"}
                return {**target, **siblings}
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


class GeminiLLM:
    def __init__(self, settings: Settings, client: genai.Client | None = None):
        self.settings = settings
        self.model = settings.model
        self._client = client

    @property
    def client(self) -> genai.Client:
        # Created lazily so the app can start (and serve the page) without a key.
        if self._client is None:
            # Check first: a keyless genai.Client() fails half-built and logs a spurious
            # AttributeError when it is garbage-collected.
            if not has_key("gemini"):
                raise LLMError("auth", NO_KEY, reason="auth")
            try:
                self._client = genai.Client(http_options=types.HttpOptions(retry_options=RETRY_OPTIONS))
            except ValueError as exc:
                raise LLMError("auth", NO_KEY, reason="auth") from exc
        return self._client

    async def structured(
        self,
        *,
        purpose: str,
        output_format: type[T],
        system: str,
        user: str,
        effort: str = "medium",
        max_tokens: int = 4000,
        temperature: float | None = None,
    ) -> LLMResult[T]:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=json_schema_for(output_format),
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL.get(effort, types.ThinkingLevel.MEDIUM)),
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        last_error: LLMError | None = None
        for _ in range(MALFORMED_ATTEMPTS):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model, contents=[user], config=config
                )
            except errors.ClientError as exc:
                raise _client_error(purpose, self.model, exc) from exc
            except errors.APIError as exc:
                raise LLMError(
                    "api_error", f"{purpose}: Gemini {PHRASE['provider_error']} {exc.code} ({exc.message})",
                    reason="provider_error",
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMError(
                    "api_error", f"{purpose}: {PHRASE['unreachable']} the Gemini API ({type(exc).__name__})",
                    reason="unreachable",
                ) from exc

            feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(feedback, "block_reason", None) if feedback else None
            if block_reason:
                raise LLMError(
                    "refused", f"{purpose}: Gemini {PHRASE['refused']}: it blocked the request ({_name(block_reason)})",
                    reason="refused",
                )

            candidates = getattr(response, "candidates", None) or []
            finish = _name(getattr(candidates[0], "finish_reason", None)) if candidates else None
            if finish in REFUSAL_FINISH_REASONS:
                raise LLMError("refused", f"{purpose}: Gemini {PHRASE['refused']} (finish reason: {finish})", reason="refused")

            text = response.text
            if not text:
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
            return LLMResult(
                output=parsed,
                model=getattr(response, "model_version", None) or self.model,
                usage=_usage(getattr(response, "usage_metadata", None)),
            )

        assert last_error is not None
        raise last_error


def _client_error(purpose: str, model: str, exc: errors.ClientError) -> LLMError:
    message = exc.message or ""
    lowered = message.lower()
    # Google puts the quota that was hit (e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier)
    # in the response details, not always in the message.
    details = json.dumps(exc.details, default=str) if exc.details else ""
    if exc.code in (401, 403) or "api key" in lowered:
        return LLMError("auth", f"{purpose}: Gemini {PHRASE['auth']} ({message})", reason="auth")
    if exc.code == 429:
        if quota_is_hard(message, details):
            return LLMError(
                "api_error", f"{purpose}: Gemini {PHRASE['quota_exhausted']} for {model} ({message})", reason="quota_exhausted"
            )
        return LLMError(
            "api_error",
            f"{purpose}: Gemini {PHRASE['rate_limited']}, even after retries ({message}). "
            "On the free tier, lower DEAL_CONCURRENCY or wait a minute.",
            reason="rate_limited",
        )
    if exc.code == 404:
        return LLMError(
            "bad_request",
            f"{purpose}: Gemini {PHRASE['model_unavailable']}: {model}; pick another with the model button or DEAL_MODEL ({message})",
            reason="model_unavailable",
        )
    if exc.code == 413 or ("token" in lowered and "exceeds" in lowered):
        return LLMError("bad_request", f"{purpose}: Gemini {PHRASE['too_large']} for {model} ({message})", reason="too_large")
    return LLMError("bad_request", f"{purpose}: Gemini rejected the request ({exc.code}: {message})", reason="unknown")


def _name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "name", None) or getattr(value, "value", None) or value)


def _usage(metadata: Any) -> TokenUsage:
    if metadata is None:
        return TokenUsage()
    output = (getattr(metadata, "candidates_token_count", 0) or 0) + (getattr(metadata, "thoughts_token_count", 0) or 0)
    return TokenUsage(
        input_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
        output_tokens=output,
        cache_read_input_tokens=getattr(metadata, "cached_content_token_count", 0) or 0,
    )
