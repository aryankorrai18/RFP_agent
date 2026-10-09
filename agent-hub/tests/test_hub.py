"""The router and the hub API. No network: agent health checks are replaced by a fixed answer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent_hub import main
from agent_hub.router import load_registry, route, score

AGENTS = load_registry()


def best(message: str) -> str | None:
    result = route(message, AGENTS)
    return result.best.agent["id"] if result.best else None


def test_registry_is_consistent():
    ids = [a["id"] for a in AGENTS]
    assert len(ids) == len(set(ids)) and len({a["number"] for a in AGENTS}) == len(AGENTS) == 15
    for agent in AGENTS:
        assert agent["status"] in ("live", "planned") and agent["keywords"] and agent["examples"]
        if agent["status"] == "live":
            assert agent["url"].startswith("http://127.0.0.1:") and agent["open_path"].startswith("/")


@pytest.mark.parametrize("message, expected", [
    ("Brief me on the Cedarline deal", "deals"),
    ("why did we lose similar deals last quarter?", "deals"),
    ("the champion went quiet and the competitor is circling", "deals"),
    ("Answer this security questionnaire for a bank", "rfp"),
    ("draft our response to a new RFP", "rfp"),
    ("Which proposals have we won?", "rfp"),
    ("What should we post on LinkedIn this week", "social"),
    ("checkout outage in production, any postmortem?", "incident"),
    ("Prep me for my meeting with Dana", "meeting"),
])
def test_questions_reach_the_right_agent(message, expected):
    assert best(message) == expected


def test_phrases_beat_single_words_and_unrelated_text_matches_nothing():
    assert score(AGENTS[0], "call prep for the deal").score > score(AGENTS[0], "the deal").score
    assert route("what is the capital of France", AGENTS).kind == "none"
    assert route("hello", AGENTS).kind == "greeting" and route("What can you do?", AGENTS).kind == "greeting"


def test_a_live_agent_wins_a_tie_over_a_planned_one():
    tie = {"a": 1}
    live = {"id": "x", "number": 99, "name": "Zed", "status": "live", "keywords": ["frobnicate"], "examples": ["x"]}
    planned = {"id": "y", "number": 1, "name": "Why", "status": "planned", "keywords": ["frobnicate"], "examples": ["y"]}
    assert route("please frobnicate", [planned, live]).best.agent["id"] == "x" and tie


@pytest.fixture
def client(monkeypatch):
    states = {a["id"]: ("down" if a["status"] == "live" else "planned") for a in AGENTS}

    async def fake_statuses() -> dict[str, str]:
        return states

    monkeypatch.setattr(main, "statuses", fake_statuses)
    with TestClient(main.app) as c:
        c.states = states
        yield c


def test_agents_endpoint_lists_all_with_status(client):
    agents = client.get("/api/agents").json()["agents"]
    assert len(agents) == 15
    deals = next(a for a in agents if a["id"] == "deals")
    assert deals["status"] == "down" and deals["open_url"] == "http://127.0.0.1:8002/#/deals" and deals["start_hint"]
    assert next(a for a in agents if a["id"] == "meeting")["open_url"] is None


def test_chat_points_to_a_running_agent_and_explains_a_stopped_one(client):
    stopped = client.post("/api/chat", json={"message": "Brief me on a stalled deal"}).json()
    assert stopped["agent"]["id"] == "deals" and "isn't running" in stopped["reply"] and "run.ps1" in stopped["reply"]
    client.states["deals"] = "up"
    running = client.post("/api/chat", json={"message": "Brief me on a stalled deal"}).json()
    assert running["agent"]["status"] == "up" and running["agent"]["open_url"].endswith("/#/deals")
    assert running["method"] == "keywords" and "deal" in running["matched"]


def test_chat_is_honest_about_planned_agents_and_unknown_questions(client):
    planned = client.post("/api/chat", json={"message": "what should I post on linkedin"}).json()
    assert planned["agent"]["id"] == "social" and "isn't built yet" in planned["reply"]
    unknown = client.post("/api/chat", json={"message": "what is the capital of France"}).json()
    assert unknown["agent"] is None and "not sure" in unknown["reply"] and unknown["alternatives"]
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 2001}).status_code == 422


def test_page_and_health(client):
    assert client.get("/health").json() == {"status": "ok", "agents": 15, "live": 2}
    page = client.get("/")
    assert page.status_code == 200 and "Agent Hub" in page.text
    assert page.headers["x-frame-options"] == "DENY"
