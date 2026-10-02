"""Provider-independent LLM interface plus the Anthropic implementation.

Both calls use the Anthropic SDK's structured-output helper (`beta.messages.parse`) with:
- adaptive thinking, and effort set per call type;
- the server-side refusal fallback (`fallbacks="default"`), which re-runs a declined
  request on a fallback model inside the same call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from . import prompts
from ..config import Settings
from ..parsing.parser import ParsedDocument
from .errors import PHRASE, provider_health, quota_is_hard, reason_of
from ..schemas import DraftResult, ExtractionResult, Fact, JudgeResult, PairsResult, PastAnswer, Requirement

FALLBACK_BETA = "server-side-fallback-2026-07-01"
EXTRACTION_MAX_TOKENS = 16000
PAIRS_MAX_TOKENS = 16000  # answers are copied verbatim, so this output is the longest
DRAFT_MAX_TOKENS = 8000
MALFORMED_ATTEMPTS = 2  # one retry when the output doesn't match the schema
JUDGE_MAX_TOKENS = 1500  # a verdict is a winner, a short reason and six scores

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)


class LLMError(Exception):
    """A failed AI call. kind is one of: auth, refused, malformed, bad_request, api_error.
    reason is the finer cause the person sees (provider_errors.REASONS): quota used up, rate
    limited, model not available, and so on. Adapters set it; otherwise it is read from the message."""

    def __init__(self, kind: str, message: str, reason: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.reason = reason or reason_of(message) or "unknown"


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens


@dataclass
class LLMResult(Generic[T]):
    output: T
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)


class LLM(Protocol):
    model: str

    async def extract_requirements(self, document: ParsedDocument) -> LLMResult[ExtractionResult]: ...

    async def extract_pairs(self, document: ParsedDocument) -> LLMResult[PairsResult]: ...

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
        """Draft from official facts and retrieved approved answers. Instructions come from the
        reviewer (Regenerate).
        temperature is a best-effort request for the memory comparison (Step 4): it pins down
        the model's own sampling so a difference between two drafts is attributable to what
        memory offered, not to chance. Claude ignores it: extended thinking pins temperature=1."""
        ...

    async def judge(self, system: str, message: str) -> LLMResult[JudgeResult]:
        """V4 pairwise judge: which of two blinded drafts is better (rfp_assistant/api/v1/judge.py
        builds the prompt). Temperature 0 where the provider allows it."""
        ...


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

    async def extract_requirements(self, document: ParsedDocument) -> LLMResult[ExtractionResult]:
        return await self._parse(
            purpose="requirement extraction",
            output_format=ExtractionResult,
            system=prompts.EXTRACTION_SYSTEM,
            content=prompts.extraction_content(document),
            effort=self.settings.extraction_effort,
            max_tokens=EXTRACTION_MAX_TOKENS,
        )

    async def extract_pairs(self, document: ParsedDocument) -> LLMResult[PairsResult]:
        return await self._parse(
            purpose="pair extraction",
            output_format=PairsResult,
            system=prompts.PAIRS_SYSTEM,
            content=prompts.pairs_content(document),
            effort=self.settings.extraction_effort,
            max_tokens=PAIRS_MAX_TOKENS,
        )

    async def draft_answer(
        self,
        company: str,
        facts: list[Fact],
        requirement: Requirement,
        past_answers: list[PastAnswer] | None = None,
        instructions: str | None = None,
        *,
        temperature: float | None = None,  # ignored: extended thinking requires temperature=1
    ) -> LLMResult[DraftResult]:
        return await self._parse(
            purpose=f"drafting {requirement.id}",
            output_format=DraftResult,
            system=prompts.drafting_system(company, facts),
            content=prompts.drafting_user_message(requirement, past_answers, instructions),
            effort=self.settings.draft_effort,
            max_tokens=DRAFT_MAX_TOKENS,
        )

    async def judge(self, system: str, message: str) -> LLMResult[JudgeResult]:
        return await self._parse(
            purpose="judging a draft pair", output_format=JudgeResult, system=system, content=message,
            effort="low", max_tokens=JUDGE_MAX_TOKENS,
        )

    async def _parse(
        self,
        *,
        purpose: str,
        output_format: type[T],
        system: str | list[dict],
        content: str | list[dict],
        effort: str,
        max_tokens: int,
    ) -> LLMResult[T]:
        last_error: LLMError | None = None
        for _ in range(MALFORMED_ATTEMPTS):
            try:
                response = await self.client.beta.messages.parse(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": content}],
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


class MonitoredLLM:
    """Any provider's LLM, with each call's outcome recorded in provider_health so the status bar
    can say the model is failing, and why, as soon as a call fails (and clear it after a success)."""

    def __init__(self, inner: LLM, provider: str):
        self.inner = inner
        self.provider = provider

    @property
    def model(self) -> str:
        return self.inner.model

    async def _watch(self, call):  # noqa: ANN001, ANN202
        try:
            result = await call
        except LLMError as exc:
            provider_health.failure(self.provider, self.model, exc.reason, exc.message)
            log.warning("%s %s call failed (%s): %s", self.provider, self.model, exc.reason, exc.message)
            raise
        provider_health.success(self.provider, self.model)
        return result

    async def extract_requirements(self, document: ParsedDocument) -> LLMResult[ExtractionResult]:
        return await self._watch(self.inner.extract_requirements(document))

    async def extract_pairs(self, document: ParsedDocument) -> LLMResult[PairsResult]:
        return await self._watch(self.inner.extract_pairs(document))

    async def draft_answer(self, *args, **kwargs) -> LLMResult[DraftResult]:  # noqa: ANN002, ANN003
        return await self._watch(self.inner.draft_answer(*args, **kwargs))

    async def judge(self, system: str, message: str) -> LLMResult[JudgeResult]:
        return await self._watch(self.inner.judge(system, message))


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
