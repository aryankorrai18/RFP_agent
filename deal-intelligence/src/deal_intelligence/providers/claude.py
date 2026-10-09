"""The Anthropic implementation of the structured-output call.

Uses the SDK's structured-output helper (`beta.messages.parse`) with:
- adaptive thinking, and effort set per call;
- the server-side refusal fallback (`fallbacks="default"`), which re-runs a declined
  request on a fallback model inside the same call.
"""

from __future__ import annotations

from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from ..config import Settings
from .base import MALFORMED_ATTEMPTS, LLMError, LLMResult, TokenUsage
from .errors import PHRASE, quota_is_hard

FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T", bound=BaseModel)


class ClaudeLLM:
    def __init__(self, settings: Settings, client: anthropic.AsyncAnthropic | None = None):
        self.settings = settings
        self.model = settings.model
        self._client = client

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        # Created lazily so the app can start (and serve the page) without credentials.
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(max_retries=3)
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
        temperature: float | None = None,  # ignored: extended thinking requires temperature=1
    ) -> LLMResult[T]:
        last_error: LLMError | None = None
        for _ in range(MALFORMED_ATTEMPTS):
            try:
                response = await self.client.beta.messages.parse(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    output_format=output_format,
                    output_config={"effort": effort},
                    thinking={"type": "adaptive"},
                    betas=[FALLBACK_BETA],
                    fallbacks="default",
                )
            except ValidationError:
                # The SDK validates the JSON inside parse(); a truncated (max_tokens) or
                # malformed output lands here.
                last_error = LLMError(
                    "malformed", f"{purpose}: {PHRASE['malformed']}: output did not match the expected schema",
                    reason="malformed",
                )
                continue
            except anthropic.APIStatusError as exc:
                raise _status_error(purpose, self.model, exc) from exc
            except anthropic.APIConnectionError as exc:
                raise LLMError(
                    "api_error", f"{purpose}: {PHRASE['unreachable']} the Anthropic API ({type(exc).__name__})",
                    reason="unreachable",
                ) from exc
            except TypeError as exc:
                if "authentication" in str(exc).lower():
                    raise LLMError(
                        "auth", f"Anthropic {PHRASE['auth']}: no credentials found. Set ANTHROPIC_API_KEY (see README).",
                        reason="auth",
                    ) from exc
                raise

            usage = _usage(response.usage)
            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                category = getattr(details, "category", None) if details else None
                raise LLMError(
                    "refused", f"{purpose}: the model {PHRASE['refused']} (category: {category or 'unknown'})",
                    reason="refused",
                )

            parsed = response.parsed_output
            if parsed is None:
                last_error = LLMError(
                    "malformed", f"{purpose}: {PHRASE['malformed']}: no structured output (stop_reason: {response.stop_reason})",
                    reason="malformed",
                )
                continue
            return LLMResult(output=parsed, model=response.model, usage=usage)

        assert last_error is not None
        raise last_error


def _status_error(purpose: str, model: str, exc: anthropic.APIStatusError) -> LLMError:
    status, detail = exc.status_code, exc.message
    lowered = (detail or "").lower()
    if status in (401, 403):
        return LLMError("auth", f"{purpose}: Anthropic {PHRASE['auth']} ({detail})", reason="auth")
    if status == 429 or "credit balance" in lowered:
        if quota_is_hard(detail):
            return LLMError("api_error", f"{purpose}: Anthropic {PHRASE['quota_exhausted']} ({detail})", reason="quota_exhausted")
        return LLMError("api_error", f"{purpose}: Anthropic {PHRASE['rate_limited']}, even after retries ({detail})", reason="rate_limited")
    if status == 404:
        return LLMError("bad_request", f"{purpose}: Anthropic {PHRASE['model_unavailable']}: {model} ({detail})", reason="model_unavailable")
    if status == 413 or "prompt is too long" in lowered:
        return LLMError("bad_request", f"{purpose}: Anthropic {PHRASE['too_large']} for {model} ({detail})", reason="too_large")
    if status >= 500:
        return LLMError("api_error", f"{purpose}: Anthropic {PHRASE['provider_error']} {status} ({detail})", reason="provider_error")
    return LLMError("bad_request", f"{purpose}: Anthropic rejected the request ({status}: {detail})", reason="unknown")


def _usage(usage: object) -> TokenUsage:
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
