"""The hub's own language model: it only reads messages (see planner.py). The agents keep their own models and
keys for their own work, so nothing the hub's model says is ever shown as data about a deal or the library.

One provider today (Gemini, the free-tier default); `HubLLM` is the seam for another. The key comes from
agent-hub/.env (or the environment) and is re-read on every call, so pasting it in takes effect without a restart."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import httpx
from dotenv import dotenv_values
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT / ".env"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """A failed model call. `reason` is one of: no_key, auth, quota, rate_limited, model_unavailable, unreachable,
    refused, malformed, unknown; `message` is safe to show."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason, self.message = reason, message


@dataclass
class LLMResult:
    output: BaseModel
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class HubLLM(Protocol):
    model: str

    async def structured(
        self, *, system: str, user: str, output_format: type[T], max_tokens: int = 700, temperature: float = 0.0,
    ) -> LLMResult: ...


def apply_env_file() -> None:
    """Re-read agent-hub/.env. Only non-empty values are applied, so a blank line never erases a key set elsewhere."""
    for key, value in dotenv_values(ENV_FILE).items():
        if value:
            os.environ[key] = value


def has_key() -> bool:
    apply_env_file()
    return any(os.environ.get(name) for name in KEY_VARS)


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic's JSON schema with every $ref inlined."""
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = resolve(copy.deepcopy(definitions[node["$ref"].rsplit("/", 1)[-1]]))
                return {**target, **{k: resolve(v) for k, v in node.items() if k != "$ref"}}
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


class GeminiLLM:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("HUB_MODEL", "").strip() or DEFAULT_MODEL
        self._client: Any = None
        self._key: str | None = None

    def _client_for_key(self) -> Any:
        from google import genai
        from google.genai import types

        key = next((os.environ[n] for n in KEY_VARS if os.environ.get(n)), None)
        if not key:
            raise LLMError("no_key", "The hub has no model key. Put GEMINI_API_KEY in agent-hub/.env.")
        if self._client is None or key != self._key:
            self._client, self._key = genai.Client(
                api_key=key, http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(
                    attempts=3, initial_delay=1.0, max_delay=8.0, http_status_codes=[429, 500, 502, 503, 504]))), key
        return self._client

    async def structured(
        self, *, system: str, user: str, output_format: type[T], max_tokens: int = 700, temperature: float = 0.0,
    ) -> LLMResult:
        from google.genai import errors, types

        apply_env_file()
        client = self._client_for_key()
        config = types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json",
            response_json_schema=json_schema_for(output_format),
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            max_output_tokens=max_tokens, temperature=temperature,
        )
        for _attempt in range(2):
            try:
                response = await client.aio.models.generate_content(model=self.model, contents=[user], config=config)
            except errors.ClientError as exc:
                raise _client_error(self.model, exc) from exc
            except errors.APIError as exc:
                raise LLMError("unknown", f"The language model had a problem ({exc.code}).") from exc
            except httpx.HTTPError as exc:
                raise LLMError("unreachable", f"I couldn't reach the language model ({type(exc).__name__}).") from exc
            text = getattr(response, "text", None)
            if not text:
                continue
            try:
                meta = getattr(response, "usage_metadata", None)
                tokens_in = int(getattr(meta, "prompt_token_count", 0) or 0)
                tokens_out = int(getattr(meta, "candidates_token_count", 0) or 0) + int(getattr(meta, "thoughts_token_count", 0) or 0)
                return LLMResult(output_format.model_validate_json(text), self.model, tokens_in, tokens_out)
            except ValidationError:
                continue
        raise LLMError("malformed", "The language model returned something I couldn't use.")


def _client_error(model: str, exc: Any) -> LLMError:
    message = (exc.message or "").lower()
    if exc.code in (401, 403) or "api key" in message:
        return LLMError("auth", "The hub's model key was rejected. Check GEMINI_API_KEY in agent-hub/.env.")
    if exc.code == 429:
        hard = "quota" in message and ("day" in message or "exceeded" in message)
        return LLMError("quota" if hard else "rate_limited",
                        "The hub's model quota is used up." if hard else "The hub's model is rate limited right now.")
    if exc.code == 404:
        return LLMError("model_unavailable", f"The model {model} isn't available to this key. Set HUB_MODEL in agent-hub/.env.")
    return LLMError("unknown", f"The language model rejected the request ({exc.code}).")


_cached: GeminiLLM | None = None


def get_llm() -> HubLLM | None:
    """The hub's model, or None when there is no key (the hub then falls back to its basic phrase matching)."""
    global _cached
    if not has_key() or os.environ.get("HUB_PLANNER", "on").strip().lower() in ("off", "0", "false"):
        return None
    model = os.environ.get("HUB_MODEL", "").strip() or DEFAULT_MODEL
    if _cached is None or _cached.model != model:
        _cached = GeminiLLM(model)
    return _cached
