from __future__ import annotations

import pytest

from backend.config import Settings, resolve_provider

KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY", "RFP_LLM_PROVIDER",
        "RFP_MODEL", "RFP_DRAFT_CONCURRENCY")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def test_auto_picks_gemini_when_only_a_gemini_key_exists(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    settings = Settings.from_env()
    assert (settings.provider, settings.model, settings.draft_concurrency) == ("gemini", "gemini-3.5-flash-lite", 2)


def test_auto_prefers_anthropic_when_both_keys_exist(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    settings = Settings.from_env()
    assert (settings.provider, settings.model, settings.draft_concurrency) == ("anthropic", "claude-opus-5", 8)


def test_auto_without_keys_defaults_to_anthropic():
    assert resolve_provider(None) == "anthropic"


def test_explicit_provider_and_overrides_win(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("RFP_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("RFP_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("RFP_DRAFT_CONCURRENCY", "1")
    settings = Settings.from_env()
    assert (settings.provider, settings.model, settings.draft_concurrency) == ("gemini", "gemini-3.8-flash", 1)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        resolve_provider("openai")


def test_env_file_is_reread_and_blank_values_never_erase_keys(tmp_path, monkeypatch):
    from backend import main

    env_file = tmp_path / ".env"
    monkeypatch.setattr(main, "ENV_FILE", env_file)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")

    env_file.write_text("GEMINI_API_KEY=\nANTHROPIC_API_KEY=\n", encoding="utf-8")
    main.apply_env_file()
    assert "GEMINI_API_KEY" not in __import__("os").environ
    assert __import__("os").environ["ANTHROPIC_API_KEY"] == "from-shell"

    env_file.write_text("GEMINI_API_KEY=pasted-later\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    try:
        assert main.get_settings().provider == "gemini"
    finally:
        # apply_env_file wrote os.environ directly; remove it so no later test sees the key.
        __import__("os").environ.pop("GEMINI_API_KEY", None)
