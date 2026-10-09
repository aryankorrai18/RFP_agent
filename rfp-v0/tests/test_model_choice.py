"""Choosing the model in the UI: stored per provider, overrides RFP_MODEL, resettable. Offline."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import rfp_assistant.main as main
from rfp_assistant.providers import model_choice
from rfp_assistant.config import Settings
from rfp_assistant.providers.model_choice import ModelOption, apply_choice, model_source, write_choice

BASE = replace(Settings(), provider="gemini", model="gemini-3.5-flash-lite")
OPTIONS = [ModelOption("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite"), ModelOption("gemini-3.8-flash", "Gemini 3.8 Flash")]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("RFP_MODEL_CHOICE_FILE", str(tmp_path / "model_choice.json"))
    monkeypatch.setenv("RFP_MODEL", "gemini-3.5-flash-lite")
    return tmp_path


@pytest.fixture
def client(isolated, monkeypatch):
    async def fake_list(provider):
        return OPTIONS, None

    monkeypatch.setattr(main, "list_models", fake_list)
    main.app.dependency_overrides[main.get_settings] = lambda: apply_choice(BASE)
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_choice_overrides_env_and_resets(isolated):
    assert apply_choice(BASE).model == "gemini-3.5-flash-lite"
    assert model_source(BASE) == "env"
    write_choice("gemini", "gemini-3.8-flash")
    assert apply_choice(BASE).model == "gemini-3.8-flash"
    assert model_source(BASE) == "ui"
    assert apply_choice(replace(BASE, provider="anthropic", model="claude-opus-5")).model == "claude-opus-5"
    write_choice("gemini", None)
    assert apply_choice(BASE).model == "gemini-3.5-flash-lite"


def test_corrupt_choice_file_is_ignored(isolated):
    model_choice.choice_file().write_text("{not json", encoding="utf-8")
    assert apply_choice(BASE).model == "gemini-3.5-flash-lite"


def test_api_lists_sets_and_resets(client):
    view = client.get("/v1/models").json()
    assert view["current"] == "gemini-3.5-flash-lite" and view["source"] == "env"
    assert [o["id"] for o in view["options"]] == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

    chosen = client.put("/v1/models", json={"model": "gemini-3.8-flash"}).json()
    assert chosen["current"] == "gemini-3.8-flash" and chosen["source"] == "ui"
    assert client.get("/health").json()["model"] == "gemini-3.8-flash"

    reset = client.put("/v1/models", json={"model": None}).json()
    assert reset["current"] == "gemini-3.5-flash-lite" and reset["source"] == "env"


def test_api_rejects_unknown_and_malformed_models(client):
    unknown = client.put("/v1/models", json={"model": "gemini-9-ultra"})
    assert unknown.status_code == 422 and unknown.json()["error"]["code"] == "unknown_model"
    bad = client.put("/v1/models", json={"model": "../../etc"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_model"
    assert client.get("/v1/models").json()["source"] == "env"


def test_api_refuses_to_guess_when_the_list_is_unavailable(isolated, monkeypatch):
    async def failing_list(provider):
        return [], "Couldn't load the model list from Google: ConnectError."

    monkeypatch.setattr(main, "list_models", failing_list)
    main.app.dependency_overrides[main.get_settings] = lambda: apply_choice(BASE)
    try:
        response = TestClient(main.app).put("/v1/models", json={"model": "gemini-3.8-flash"})
    finally:
        main.app.dependency_overrides.clear()
    assert response.status_code == 503 and response.json()["error"]["code"] == "model_list_unavailable"
