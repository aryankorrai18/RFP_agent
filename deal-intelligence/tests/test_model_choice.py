"""Choosing the model: stored per provider, overrides DEAL_MODEL, resettable; model lists cached. Offline."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from deal_intelligence.config import DEFAULT_MODEL, Settings
from deal_intelligence.providers import model_choice
from deal_intelligence.providers.model_choice import (
    ANTHROPIC_MODELS, ModelOption, apply_choice, base_model, default_model, list_models, model_source, read_choice,
    refresh, write_choice,
)

BASE = replace(Settings(), provider="gemini", model="gemini-3.5-flash-lite")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DEAL_MODEL_CHOICE_FILE", str(tmp_path / "model_choice.json"))
    monkeypatch.setenv("DEAL_MODEL", "gemini-3.5-flash-lite")
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    model_choice._cache.clear()
    yield tmp_path
    model_choice._cache.clear()


def test_choice_overrides_env_and_resets(isolated):
    assert apply_choice(BASE).model == "gemini-3.5-flash-lite"
    assert model_source(BASE) == "env"
    write_choice("gemini", "gemini-3.8-flash")
    assert apply_choice(BASE).model == "gemini-3.8-flash"
    assert model_source(BASE) == "ui"
    assert apply_choice(replace(BASE, provider="anthropic", model="claude-opus-5")).model == "claude-opus-5"
    write_choice("gemini", None)
    assert apply_choice(BASE).model == "gemini-3.5-flash-lite"
    assert read_choice("gemini") is None


def test_choice_file_defaults_next_to_the_data_dir(monkeypatch):
    monkeypatch.delenv("DEAL_MODEL_CHOICE_FILE", raising=False)
    assert model_choice.choice_file() == model_choice.ROOT / "data" / "model_choice.json"
    monkeypatch.setenv("DEAL_MODEL_CHOICE_FILE", "elsewhere/choice.json")
    assert model_choice.choice_file() == model_choice.ROOT / "elsewhere" / "choice.json"


def test_apply_choice_keeps_the_rest_of_the_settings(isolated):
    write_choice("gemini", "gemini-3.8-flash")
    applied = apply_choice(replace(BASE, concurrency=3, brief_effort="high"))
    assert (applied.model, applied.concurrency, applied.brief_effort, applied.provider) == ("gemini-3.8-flash", 3, "high", "gemini")


def test_choices_are_stored_per_provider_and_survive_each_other(isolated):
    write_choice("gemini", "gemini-3.8-flash")
    write_choice("groq", "openai/gpt-oss-120b")
    write_choice("gemini", None)
    assert read_choice("gemini") is None and read_choice("groq") == "openai/gpt-oss-120b"
    assert json.loads(model_choice.choice_file().read_text(encoding="utf-8")) == {"groq": "openai/gpt-oss-120b"}


def test_write_creates_missing_directories_and_leaves_no_temp_file(isolated, monkeypatch):
    path = isolated / "nested" / "dir" / "model_choice.json"
    monkeypatch.setenv("DEAL_MODEL_CHOICE_FILE", str(path))
    write_choice("gemini", "gemini-3.8-flash")
    assert path.exists() and not path.with_suffix(".tmp").exists()


def test_corrupt_or_invalid_choice_file_is_ignored(isolated):
    path = model_choice.choice_file()
    path.write_text("{not json", encoding="utf-8")
    assert apply_choice(BASE).model == "gemini-3.5-flash-lite"
    path.write_text(json.dumps({"gemini": "../../etc", "groq": 5, "anthropic": "claude-opus-5"}), encoding="utf-8")
    assert read_choice("gemini") is None and read_choice("groq") is None and read_choice("anthropic") == "claude-opus-5"


def test_base_model_and_source_follow_env_then_default(isolated, monkeypatch):
    assert base_model("gemini") == "gemini-3.5-flash-lite"
    monkeypatch.delenv("DEAL_MODEL")
    assert base_model("groq") == DEFAULT_MODEL["groq"] == default_model("groq")
    assert model_source(replace(BASE, provider="groq")) == "default"


def test_refresh_reapplies_a_just_changed_choice_over_the_base_model(isolated):
    stale = replace(BASE, model="gemini-3.8-flash")
    assert refresh(stale).model == "gemini-3.5-flash-lite"  # choice was reset: back to the env model
    write_choice("gemini", "gemini-3.9-pro")
    assert refresh(stale).model == "gemini-3.9-pro"


def test_anthropic_models_are_a_fixed_list(isolated):
    options, error = asyncio.run(list_models("anthropic"))
    assert error is None and [o.id for o in options] == list(ANTHROPIC_MODELS)
    assert all(model_choice.MODEL_NAME.match(o.id) for o in options)


def test_listing_without_a_key_explains_instead_of_failing(isolated):
    options, error = asyncio.run(list_models("gemini"))
    assert options == [] and "No Gemini key" in error
    options, error = asyncio.run(list_models("groq"))
    assert options == [] and "No Groq key" in error


class FakePager:
    def __init__(self, models):
        self.models = models

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for model in self.models:
            yield model


def gemini_model(name, actions=("generateContent",), display=None, limit=1_000_000):
    return SimpleNamespace(name=f"models/{name}", supported_actions=list(actions), display_name=display, input_token_limit=limit)


def test_gemini_list_filters_to_text_models_sorts_and_caches(isolated, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls: list[dict] = []
    found = [
        gemini_model("gemini-3.8-flash", display="Gemini 3.8 Flash"),
        gemini_model("gemini-3.5-flash-lite"),
        gemini_model("gemini-3.5-flash-tts"),
        gemini_model("gemini-embedding-001", actions=("embedContent",)),
        gemini_model("gemini-robotics-er-1.5"),
        gemini_model("imagen-4"),
    ]

    class FakeClient:
        def __init__(self):
            async def list_(config):
                calls.append(config)
                return FakePager(found)

            self.aio = SimpleNamespace(models=SimpleNamespace(list=list_))

    import google.genai

    monkeypatch.setattr(google.genai, "Client", FakeClient)
    options, error = asyncio.run(list_models("gemini"))
    assert error is None
    assert options == [ModelOption("gemini-3.5-flash-lite", "gemini-3.5-flash-lite", 1_000_000),
                       ModelOption("gemini-3.8-flash", "Gemini 3.8 Flash", 1_000_000)]
    again, _ = asyncio.run(list_models("gemini"))
    assert again == options and len(calls) == 1  # second call is served from the cache


def test_cache_expires(isolated, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    calls = {"n": 0}

    async def fake_list():
        calls["n"] += 1
        return [("llama-3.3-70b-versatile", 131072)]

    import deal_intelligence.providers.groq as groq_module

    monkeypatch.setattr(groq_module, "list_groq_models", fake_list)
    now = {"t": 1000.0}
    monkeypatch.setattr(model_choice.time, "monotonic", lambda: now["t"])
    asyncio.run(list_models("groq"))
    now["t"] += model_choice.LIST_CACHE_SECONDS - 1
    asyncio.run(list_models("groq"))
    assert calls["n"] == 1
    now["t"] += 2
    asyncio.run(list_models("groq"))
    assert calls["n"] == 2


def test_listing_failures_are_reported_not_raised_and_not_cached(isolated, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    async def failing():
        raise ConnectionError("down")

    import deal_intelligence.providers.groq as groq_module

    monkeypatch.setattr(groq_module, "list_groq_models", failing)
    options, error = asyncio.run(list_models("groq"))
    assert options == [] and "Couldn't load the model list from Groq: ConnectionError" in error
    assert "groq" not in model_choice._cache

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    import google.genai

    def broken_client():
        raise RuntimeError("boom")

    monkeypatch.setattr(google.genai, "Client", broken_client)
    options, error = asyncio.run(list_models("gemini"))
    assert options == [] and "Couldn't load the model list from Google: RuntimeError" in error
