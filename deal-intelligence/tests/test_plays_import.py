"""A company's own play catalogue, saved through POST /v1/plays (no model call)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deal_intelligence.main import app
from tests.builders import make_context


@pytest.fixture
def client(tmp_path):  # noqa: ANN001, ANN201
    ctx = make_context(tmp_path)
    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.state.v1_factory = None


def test_a_fresh_workspace_has_no_plays_and_new_ones_get_codes(client):
    assert client.get("/v1/plays").json()["plays"] == []
    r = client.post("/v1/plays", json={"plays": [
        {"name": "Executive sponsor call", "description": "Bring our VP to meet theirs.", "category": "stakeholder", "addresses": ["timeline"]},
        {"name": "Security review pack", "description": "Send the SOC 2 report and answers.", "category": "security", "addresses": ["security review", "SSO"]},
    ]}).json()
    assert r["added"] == ["PLAY-01", "PLAY-02"] and r["updated"] == [] and r["ignored_objections"] == []
    pack = next(p for p in r["plays"] if p["code"] == "PLAY-02")
    assert pack["addresses"] == ["security_review", "sso"] and pack["category"] == "security"


def test_the_same_name_updates_instead_of_duplicating_and_codes_are_kept(client):
    client.post("/v1/plays", json={"plays": [{"name": "Pilot", "description": "Two-week pilot."}]})
    r = client.post("/v1/plays", json={"plays": [{"name": "pilot", "description": "Four-week paid pilot.", "addresses": ["pricing"]}]}).json()
    assert r["added"] == [] and r["updated"] == ["PLAY-01"]
    assert len(r["plays"]) == 1 and r["plays"][0]["description"] == "Four-week paid pilot." and r["plays"][0]["addresses"] == ["pricing"]


def test_unknown_objections_are_reported_not_stored_and_unknown_categories_become_process(client):
    r = client.post("/v1/plays", json={"plays": [{"name": "Discount", "category": "money", "addresses": ["pricing", "budget freeze"]}]}).json()
    play = r["plays"][0]
    assert play["addresses"] == ["pricing"] and play["category"] == "process" and r["ignored_objections"] == ["budget_freeze"]
    assert play["description"] == "Discount"  # a play needs a description; its name stands in


def test_an_empty_or_oversized_list_is_refused(client):
    assert client.post("/v1/plays", json={"plays": []}).status_code == 422
    assert client.post("/v1/plays", json={"plays": [{"name": f"P{i}"} for i in range(101)]}).status_code == 422
