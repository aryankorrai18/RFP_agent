"""Provider-independent LLM interface: one structured-output call.

Every domain task (extracting a deal's signals, writing a brief) is a system prompt, a user message
and a pydantic class. Adapters (claude.py, gemini.py, groq.py) implement `structured`; nothing here
or in them knows about deals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from .errors import provider_health, reason_of

MALFORMED_ATTEMPTS = 2  # one retry when the output doesn't match the schema

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)


class LLMError(Exception):
    """A failed AI call. kind is one of: auth, refused, malformed, bad_request, api_error.
    reason is the finer cause the person sees (errors.REASONS): quota used up, rate limited, model
    not available, and so on. Adapters set it; otherwise it is read from the message."""

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
        """Run one call and return the validated `output_format` instance. Raises LLMError.

        temperature is a best-effort request (the memory comparison pins it to 0 so a difference
        between two briefs is attributable to what memory offered, not to chance). Claude ignores it:
        extended thinking pins temperature=1."""
        ...


class MonitoredLLM:
    """Any provider's LLM, with each call's outcome recorded in provider_health so the status bar
    can say the model is failing, and why, as soon as a call fails (and clear it after a success)."""

    def __init__(self, inner: LLM, provider: str):
        self.inner = inner
        self.provider = provider

    @property
    def model(self) -> str:
        return self.inner.model

    async def structured(self, **kwargs) -> LLMResult:  # noqa: ANN003
        try:
            result = await self.inner.structured(**kwargs)
        except LLMError as exc:
            provider_health.failure(self.provider, self.model, exc.reason, exc.message)
            log.warning("%s %s call failed (%s): %s", self.provider, self.model, exc.reason, exc.message)
            raise
        provider_health.success(self.provider, self.model)
        return result
