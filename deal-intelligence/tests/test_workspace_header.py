"""A caller can name its workspace per request (X-Workspace) without switching the one the app shows."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deal_intelligence import workspaces
from deal_intelligence.config import Settings
from deal_intelligence.main import app

from .builders import make_context
from .fake_llm import FakeLLM


@pytest.fixture
def two_workspaces(tmp_path):
    settings = Settings()
    workspaces.ensure_registry(settings)  # "main" is active
    other = workspaces.create("Other team", "company", settings)
    main_ctx = make_context(tmp_path / "main", llm=FakeLLM())
    other_ctx = make_context(tmp_path / "other", llm=FakeLLM())
    app.state.v1_factory = lambda: main_ctx
    app.state.workspace_factory = lambda wid: {other.id: other_ctx}[wid]
    try:
        with TestClient(app) as client:
            yield client, other.id, main_ctx, other_ctx
    finally:
        app.state.v1_factory = None
        app.state.workspace_factory = None


def add_deal(client: TestClient, name: str, **headers: str) -> int:
    response = client.post("/v1/deals", data={"name": name, "account": f"{name} Inc"}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def names(client: TestClient, **headers: str) -> list[str]:
    return [d["name"] for d in client.get("/v1/deals", headers=headers).json()["deals"]]


def test_a_named_workspace_has_its_own_deals(two_workspaces):
    client, other, _main, _other = two_workspaces
    add_deal(client, "Alpha")
    add_deal(client, "Beta", **{"X-Workspace": other})
    assert names(client) == ["Alpha"]
    assert names(client, **{"X-Workspace": other}) == ["Beta"]


def test_naming_the_active_workspace_is_the_same_as_naming_none(two_workspaces):
    client, _other, _main, _ = two_workspaces
    add_deal(client, "Alpha")
    assert names(client, **{"X-Workspace": "main"}) == ["Alpha"]


def test_the_active_workspace_is_not_switched(two_workspaces):
    client, other, _main, _ = two_workspaces
    names(client, **{"X-Workspace": other})
    assert client.get("/v1/workspaces").json()["active"] == "main"


def test_an_unknown_workspace_is_a_404(two_workspaces):
    client, *_ = two_workspaces
    response = client.get("/v1/deals", headers={"X-Workspace": "nope"})
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"


def test_a_blank_header_means_the_active_workspace(two_workspaces):
    client, *_ = two_workspaces
    add_deal(client, "Alpha")
    assert names(client, **{"X-Workspace": "  "}) == ["Alpha"]


def test_the_context_for_a_named_workspace_is_kept_and_closed_at_shutdown(two_workspaces):
    client, other, _main, other_ctx = two_workspaces
    names(client, **{"X-Workspace": other})
    names(client, **{"X-Workspace": other})
    assert list(app.state.pool) == [other] and other_ctx.workspace_id == other
    client.__exit__(None, None, None)
    assert app.state.pool == {}


def test_the_service_token_guards_the_api_only_when_it_is_set(two_workspaces, monkeypatch):
    client, *_ = two_workspaces
    assert client.get("/v1/deals").status_code == 200  # unset: nothing changes
    monkeypatch.setenv("DEAL_SERVICE_TOKEN", "s3cret-service-token")
    refused = client.get("/v1/deals")
    assert refused.status_code == 401 and refused.json()["error"]["code"] == "service_token"
    assert client.get("/v1/deals", headers={"X-Service-Token": "wrong"}).status_code == 401
    assert client.get("/v1/deals", headers={"X-Service-Token": "s3cret-service-token"}).status_code == 200
    assert client.get("/health").status_code == 200 and client.get("/").status_code == 200
