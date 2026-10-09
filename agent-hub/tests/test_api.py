"""The conversation API the chat page talks to (the engine runs against the fake agents)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from agent_hub import main
from agent_hub.engine import Engine
from agent_hub.store import MAX_UPLOAD_BYTES

from .helpers import RFP_DOC, action, actions_of, cards, links, texts


@pytest.fixture
def api(engine, monkeypatch):
    main.app.state.engine = engine
    with TestClient(main.app) as client:
        client.engine = engine
        yield client
    main.app.state.engine = None


def new(api) -> str:
    return api.post("/api/conversations").json()["id"]


def wait_idle(api, conv: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        page = api.get(f"/api/conversations/{conv}/events").json()
        if not page["busy"]:
            return page
        time.sleep(0.01)
    raise AssertionError("conversation stayed busy")


def send(api, conv: str, text: str = "", files=()) -> dict:
    data = {"text": text} if text else {}
    multipart = [("files", (name, content, "application/octet-stream")) for name, content in files]
    return api.post(f"/api/conversations/{conv}/messages", data=data, files=multipart or None)


def push(api, conv: str, action_id: str) -> None:
    assert api.post(f"/api/conversations/{conv}/actions", json={"action_id": action_id}).json() == {"accepted": True}


def test_a_new_conversation_starts_with_a_welcome_and_the_documented_shape(api):
    conv = new(api)
    body = api.get(f"/api/conversations/{conv}").json()
    assert set(body) == {"id", "title", "state", "events", "next", "busy", "workspaces", "company", "router_calls"} and body["id"] == conv
    assert body["title"] is None  # untitled until the first message
    assert body["busy"] is False and body["next"] == 1
    first = body["events"][0]
    assert first["n"] == 1 and first["role"] == "assistant" and first["kind"] == "text"
    assert "I'm the hub" in first["text"] and first["actions"][0]["id"].startswith("say:")
    assert first["created_at"]
    assert api.get(f"/api/conversations/{conv}").headers["cache-control"] == "no-store"


def test_unknown_conversations_are_404_on_every_route(api):
    for method, path, kwargs in [
        ("get", "/api/conversations/nope", {}), ("get", "/api/conversations/nope/events", {}),
        ("post", "/api/conversations/nope/messages", {"data": {"text": "hi"}}),
        ("post", "/api/conversations/nope/actions", {"json": {"action_id": "x"}}),
        ("get", "/api/conversations/nope/downloads/abc", {}),
    ]:
        assert getattr(api, method)(path, **kwargs).status_code == 404, path


def test_message_validation(api):
    conv = new(api)
    assert send(api, conv).status_code == 422  # neither text nor files
    assert api.post(f"/api/conversations/{conv}/messages", data={"text": "   "}).status_code == 422
    assert api.post(f"/api/conversations/{conv}/actions", json={}).status_code == 422
    assert api.post(f"/api/conversations/{conv}/actions", json={"action_id": ""}).status_code == 422
    assert api.get(f"/api/conversations/{conv}/events?after=-1").status_code == 422
    big = send(api, conv, "here", [("huge.pdf", b"x" * (MAX_UPLOAD_BYTES + 1))])
    assert big.status_code == 413 and "20 MB" in big.json()["detail"]
    assert api.get(f"/api/conversations/{conv}").json()["next"] == 1  # nothing was stored by the rejected requests


def test_the_user_event_is_stored_at_once_and_events_are_ordered(api):
    conv = new(api)
    accepted = send(api, conv, "Brief me on Cedarline").json()
    assert accepted == {"accepted": True, "n": 2}
    page = wait_idle(api, conv)
    ns = [e["n"] for e in page["events"]]
    assert ns == sorted(ns) and ns == list(range(1, len(ns) + 1))
    assert page["events"][1]["role"] == "user" and page["events"][1]["text"] == "Brief me on Cedarline"
    assert page["events"][2]["role"] == "assistant" and "D-001 Cedarline Renewal" in page["events"][2]["text"]
    assert page["next"] == ns[-1]


def test_after_returns_only_newer_events(api):
    conv = new(api)
    send(api, conv, "Brief me on Cedarline")
    full = wait_idle(api, conv)["events"]
    tail = api.get(f"/api/conversations/{conv}/events?after=2").json()
    assert [e["n"] for e in tail["events"]] == [e["n"] for e in full if e["n"] > 2]
    empty = api.get(f"/api/conversations/{conv}/events?after={full[-1]['n']}").json()
    assert empty["events"] == [] and empty["next"] == full[-1]["n"] and empty["busy"] is False
    assert api.get(f"/api/conversations/{conv}/events?after=999").json()["next"] == 999


def test_busy_is_true_while_a_job_runs_and_events_stream_in(api, fakes):
    fakes.deal.job_polls = 1000
    conv = new(api)
    send(api, conv, "Brief me on Cedarline")
    ask = wait_idle(api, conv)["events"][-1]
    push(api, conv, action([ask], "Yes")["id"])
    for _ in range(500):
        page = api.get(f"/api/conversations/{conv}/events").json()
        if any(e["kind"] == "progress" for e in page["events"]):
            break
        time.sleep(0.01)
    assert page["busy"] is True and any(e["kind"] == "progress" for e in page["events"])
    fakes.deal.job_polls = 1
    done = wait_idle(api, conv)
    assert done["busy"] is False and cards(done["events"])[-2]["title"].startswith("Brief: D-001")
    assert [e for e in done["events"] if e["role"] == "user"][-1]["text"] == "Yes, go ahead"


def test_extension_rules_are_friendly_not_a_500(api):
    conv = new(api)
    reply = send(api, conv, "New deal Tidewater at Tidewater Freight", [("tool.exe", b"MZ"), ("thread.eml", b"From: a\n\nhello")])
    assert reply.status_code == 200
    page = wait_idle(api, conv)
    user = [e for e in page["events"] if e["role"] == "user"][0]
    assert [f["name"] for f in user["files"]] == ["thread.eml", "tool.exe"]
    assert "I can't read .exe files" in texts(page["events"]) and "Created D-007" in texts(page["events"])
    empty = send(api, conv, "", [("blank.txt", b"")])
    assert empty.status_code == 200 and "blank.txt is empty" in texts(wait_idle(api, conv)["events"])


def test_a_whole_rfp_over_http_ends_in_a_proxied_download(api, fakes):
    conv = new(api)
    send(api, conv, "", [RFP_DOC])
    ask = wait_idle(api, conv)["events"][-1]
    push(api, conv, action([ask], "Yes")["id"])
    extract = wait_idle(api, conv)["events"]
    push(api, conv, actions_of(extract[-1:])[0]["id"])
    drafted = wait_idle(api, conv)["events"]
    push(api, conv, action(drafted[-1:], "Accept all")["id"])
    final = wait_idle(api, conv)["events"]
    url = [link for link in links(final) if "/downloads/" in link["url"]][0]["url"]

    response = api.get(url)
    assert response.status_code == 200 and response.content == b"FAKE-DOCX-FILE"
    assert response.headers["content-disposition"] == 'attachment; filename="Acme Security Questionnaire - response.docx"'
    assert response.headers["content-type"].startswith("application/fake-docx")
    assert api.get(f"/api/conversations/{conv}/downloads/not-a-token").status_code == 404
    other = new(api)
    assert api.get(f"/api/conversations/{other}/downloads/{url.rsplit('/', 1)[1]}").status_code == 404  # tokens are per chat
    fakes.rfp.down = True
    assert api.get(url).status_code == 503


def test_a_download_the_app_refuses_is_reported_not_crashed(api, fakes):
    conv = new(api)
    st = api.engine.load(conv)
    token = api.engine.new_download(st, 99, "docx")
    api.engine.save(conv, st)
    assert api.get(f"/api/conversations/{conv}/downloads/{token}").status_code == 502  # the app says: no such project


def test_the_old_routes_still_work(api):
    assert api.get("/health").json()["agents"] == 15
    assert api.post("/api/chat", json={"message": "Brief me on a stalled deal"}).json()["method"] == "keywords"
    assert api.get("/").status_code == 200


def test_the_hub_builds_and_closes_its_own_engine_when_none_is_given(monkeypatch, tmp_path):
    main.app.state.engine = None
    with TestClient(main.app) as client:
        built = main.app.state.engine
        assert isinstance(built, Engine) and built.store.data_dir == tmp_path / "default-data"
        conv = client.post("/api/conversations").json()["id"]
        assert client.get(f"/api/conversations/{conv}").status_code == 200
    assert main.app.state.engine is None
    assert (tmp_path / "default-data" / "hub.db").exists()


def test_the_page_is_told_which_workspaces_the_chat_uses(api):
    conv = new(api)
    assert api.get(f"/api/conversations/{conv}/events").json()["workspaces"] == {}
    send(api, conv, "use the Brightwater workspace")
    page = wait_idle(api, conv)
    assert page["workspaces"] == {"deals": "Brightwater Team"}
    assert api.get(f"/api/conversations/{conv}").json()["workspaces"] == {"deals": "Brightwater Team"}
