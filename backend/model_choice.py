"""The model chosen in the UI. It is stored per provider in a small JSON file and overrides
RFP_MODEL from .env until it is reset. The choice applies to every later model call (V0 quick
draft, V1 extraction and drafting, background jobs); calls already running keep their model."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .config import DEFAULT_MODEL, ROOT, Settings, has_key

CHOICE_FILE_ENV = "RFP_MODEL_CHOICE_FILE"
MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9._/\-]{1,80}$")
LIST_CACHE_SECONDS = 600
# Listed as supporting generateContent, but they produce audio or images or drive robots/computers,
# so they can't draft RFP answers.
NON_TEXT_TAGS = ("tts", "image", "transcribe", "computer-use", "robotics", "customtools")

# Anthropic has no free listing call here; offer the current Claude models.
ANTHROPIC_MODELS = ("claude-opus-5", "claude-opus-5-5", "claude-sonnet-5", "claude-haiku-4-5-20251001")


def choice_file() -> Path:
    raw = os.environ.get(CHOICE_FILE_ENV)
    if not raw:
        return ROOT / "data" / "model_choice.json"
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _read_all() -> dict[str, str]:
    try:
        data = json.loads(choice_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and MODEL_NAME.match(v)}


def read_choice(provider: str) -> str | None:
    return _read_all().get(provider)


def write_choice(provider: str, model: str | None) -> None:
    """Save (or with None, clear) the provider's chosen model."""
    choices = _read_all()
    if model is None:
        choices.pop(provider, None)
    else:
        choices[provider] = model
    path = choice_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(choices, indent=2), encoding="utf-8")
    tmp.replace(path)


def apply_choice(settings: Settings) -> Settings:
    chosen = read_choice(settings.provider)
    return replace(settings, model=chosen) if chosen else settings


def base_model(provider: str) -> str:
    """The model without a UI choice: RFP_MODEL from .env, else the provider default."""
    return os.environ.get("RFP_MODEL", "").strip() or DEFAULT_MODEL[provider]


def refresh(settings: Settings) -> Settings:
    """Re-apply the (possibly just changed) choice on top of the .env/default model."""
    return apply_choice(replace(settings, model=base_model(settings.provider)))


def model_source(settings: Settings) -> str:
    """Where the active model name comes from: "ui", "env" or "default"."""
    if read_choice(settings.provider):
        return "ui"
    return "env" if os.environ.get("RFP_MODEL", "").strip() else "default"


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    input_token_limit: int | None = None


_cache: dict[str, tuple[float, list[ModelOption]]] = {}


async def list_models(provider: str) -> tuple[list[ModelOption], str | None]:
    """Models the provider offers for text generation, plus an error message if listing failed.
    Listing Gemini models is a metadata call and uses no generation tokens; results are cached."""
    if provider == "anthropic":
        return [ModelOption(m, m) for m in ANTHROPIC_MODELS], None
    cached = _cache.get(provider)
    if cached and time.monotonic() - cached[0] < LIST_CACHE_SECONDS:
        return cached[1], None
    if provider == "groq":
        return await _list_groq()
    if not has_key("gemini"):
        return [], "No Gemini key found, so the model list can't be loaded."
    try:
        from google import genai

        client = genai.Client()
        options: list[ModelOption] = []
        pager = await client.aio.models.list(config={"page_size": 100})
        async for model in pager:
            name = (model.name or "").removeprefix("models/")
            actions = model.supported_actions or []
            if (
                name.startswith("gemini") and "generateContent" in actions and MODEL_NAME.match(name)
                and not any(tag in name for tag in NON_TEXT_TAGS)
            ):
                options.append(ModelOption(name, model.display_name or name, model.input_token_limit))
    except Exception as exc:  # listing must never break the page
        return [], f"Couldn't load the model list from Google: {type(exc).__name__}."
    options.sort(key=lambda o: o.id)
    _cache[provider] = (time.monotonic(), options)
    return options, None


async def _list_groq() -> tuple[list[ModelOption], str | None]:
    from .llm_groq import list_groq_models

    if not has_key("groq"):
        return [], "No Groq key found, so the model list can't be loaded."
    try:
        found = await list_groq_models()
    except Exception as exc:  # listing must never break the page
        return [], f"Couldn't load the model list from Groq: {type(exc).__name__}."
    options = sorted((ModelOption(i, i, limit) for i, limit in found if MODEL_NAME.match(i)), key=lambda o: o.id)
    _cache["groq"] = (time.monotonic(), options)
    return options, None


def default_model(provider: str) -> str:
    return DEFAULT_MODEL[provider]
