"""A caller can name its workspace per request (X-Workspace) without switching the one the app shows."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from rfp_assistant import workspaces
from rfp_assistant.config import ROOT, Settings
from rfp_assistant.main import app
from tests.conftest import docx_bytes
from tests.v1_fakes import FakeV1LLM, make_context, req


@pytest.fixture
def two_workspaces(tmp_path):
    settings = replace(Settings(), db_path=tmp_path / "main" / "rfp.db", uploads_dir=tmp_path / "main" / "uploads",
                       fact_sheet_path=ROOT / "data" / "fact_sheet.json")
    workspaces.ensure_registry(settings)  # "main" is active
    other = workspaces.create("Other company", "company", settings)
    main_ctx = make_context(tmp_path / "m", FakeV1LLM(requirements=[req("Describe your security program.")]))
    other_ctx = make_context(tmp_path / "o", FakeV1LLM(requirements=[req("Describe your security program.")]))
    app.state.v1_factory = lambda: main_ctx
    app.state.workspace_factory = lambda wid: {other.id: other_ctx}[wid]
    try:
        with TestClient(app) as client:
            yield client, other.id, other_ctx
    finally:
        app.state.v1_factory = None
        app.state.workspace_factory = None


def project_names(client: TestClient, **headers: str) -> list[str]:
    return [p["name"] for p in client.get("/v1/projects", headers=headers).json()]


def test_the_workspace_summary_describes_the_workspace_that_was_named(two_workspaces):
    client, other, _ctx = two_workspaces
    mine = client.get("/v1/workspace").json()
    theirs = client.get("/v1/workspace", headers={"X-Workspace": other}).json()
    assert mine["id"] == "main" and theirs["id"] == other and theirs["name"] == "Other company"


def test_the_active_workspace_is_not_switched(two_workspaces):
    client, other, _ctx = two_workspaces
    client.get("/v1/projects", headers={"X-Workspace": other})
    assert client.get("/v1/workspaces").json()["active"] == "main"


def test_each_workspace_keeps_its_own_projects(two_workspaces):
    client, other, _ctx = two_workspaces
    assert project_names(client) == [] and project_names(client, **{"X-Workspace": other}) == []
    client.post("/v1/projects", data={"name": "Only in other", "client": "X"},
                files={"file": ("rfp.docx", docx_bytes(["Describe your security program."]),
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                headers={"X-Workspace": other})
    assert project_names(client, **{"X-Workspace": other}) == ["Only in other"]
    assert project_names(client) == []


def test_unknown_workspace_is_a_404_and_blank_means_active(two_workspaces):
    client, *_ = two_workspaces
    response = client.get("/v1/projects", headers={"X-Workspace": "nope"})
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"
    assert client.get("/v1/projects", headers={"X-Workspace": " "}).status_code == 200


def test_pooled_contexts_are_closed_at_shutdown(two_workspaces):
    client, other, ctx = two_workspaces
    client.get("/v1/projects", headers={"X-Workspace": other})
    assert list(app.state.pool) == [other] and ctx.workspace_id == other
    client.__exit__(None, None, None)
    assert app.state.pool == {}


def test_the_service_token_guards_the_api_only_when_it_is_set(two_workspaces, monkeypatch):
    client, *_ = two_workspaces
    assert client.get("/v1/projects").status_code == 200  # unset: nothing changes
    monkeypatch.setenv("RFP_SERVICE_TOKEN", "s3cret-service-token")
    refused = client.get("/v1/projects")
    assert refused.status_code == 401 and refused.json()["error"]["code"] == "service_token"
    assert client.get("/v1/projects", headers={"X-Service-Token": "wrong"}).status_code == 401
    assert client.get("/v1/projects", headers={"X-Service-Token": "s3cret-service-token"}).status_code == 200
    assert client.get("/health").status_code == 200 and client.get("/").status_code == 200
