"""Chat history in the sidebar: a person's own chats, latest first, titled by their first message, renamed and deleted
by them only. Nothing here costs a model call."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from agent_hub import main
from agent_hub.store import Store

from .test_auth_api import make_user, second_browser, sign_in, web  # noqa: F401 - the signed-in fixture

pytestmark = pytest.mark.anyio


@pytest.fixture
def api(engine, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    main.app.state.engine = engine
    with TestClient(main.app) as client:
        client.engine = engine
        yield client
    main.app.state.engine = None


def chat(client, text: str | None = None) -> str:
    conv = client.post("/api/conversations").json()["id"]
    if text:
        assert client.post(f"/api/conversations/{conv}/messages", data={"text": text}).status_code == 200
    return conv


def listed(client, **params) -> dict:
    out = client.get("/api/conversations", params=params)
    assert out.status_code == 200
    return out.json()


def test_only_chats_with_a_message_are_listed_titled_by_it_latest_first(api):
    chat(api)  # only the welcome: not history yet
    first = chat(api, "How many deals do I have?")
    second = chat(api, "Why do we lose deals in banking and what should we change before the next renewal season starts?")
    found = listed(api)
    assert [c["id"] for c in found["conversations"]] == [second, first] and found["next"] is None
    assert found["conversations"][1]["title"] == "How many deals do I have?"
    long_title = found["conversations"][0]["title"]
    assert len(long_title) <= 60 and long_title.endswith("…")
    assert all(c["last_activity"] and c["created_at"] and isinstance(c["preview"], str) for c in found["conversations"])


def test_pages_follow_the_cursor_and_search_matches_titles(api):
    ids = [chat(api, f"Question number {i} about pricing" if i % 2 else f"Question number {i} about SSO") for i in range(5)]
    page = listed(api, limit=2)
    assert [c["id"] for c in page["conversations"]] == ids[::-1][:2] and page["next"]
    rest = listed(api, limit=10, before=page["next"])
    assert [c["id"] for c in rest["conversations"]] == ids[::-1][2:] and rest["next"] is None
    assert {c["id"] for c in listed(api, q="sso")["conversations"]} == {ids[0], ids[2], ids[4]}


def test_the_cursor_is_exact_when_chats_share_a_timestamp_and_a_chat_returns_its_title(api):
    ids = [chat(api, f"Same moment {i}") for i in range(4)]
    db = api.engine.store._db()
    db.execute("UPDATE conversations SET last_activity = '2026-10-09T10:00:00.000+00:00'")
    db.commit()
    seen, cursor = [], None
    while True:
        page = listed(api, limit=1, **({"before": cursor} if cursor else {}))
        seen += [c["id"] for c in page["conversations"]]
        cursor = page["next"]
        if not cursor:
            break
    assert sorted(seen) == sorted(ids) and len(seen) == 4
    assert api.get(f"/api/conversations/{ids[0]}").json()["title"] == "Same moment 0"


def test_rename_and_delete(api):
    conv = chat(api, "Brief me on Juniper")
    assert api.patch(f"/api/conversations/{conv}", json={"title": "  Juniper   prep "}).json() == {"id": conv, "title": "Juniper prep"}
    assert listed(api)["conversations"][0]["title"] == "Juniper prep"
    assert api.patch(f"/api/conversations/{conv}", json={"title": "   "}).status_code == 422
    assert api.patch(f"/api/conversations/{conv}", json={"title": "x" * 81}).status_code == 422
    assert api.delete(f"/api/conversations/{conv}").json() == {"ok": True}
    assert api.get(f"/api/conversations/{conv}").status_code == 404
    assert listed(api)["conversations"] == []
    assert api.post(f"/api/conversations/{conv}/messages", data={"text": "hello?"}).status_code == 404


def test_a_person_sees_and_changes_only_their_own_chats(web):  # noqa: F811
    make_user(web)
    make_user(web, "sam@globex.com", "Globex")
    assert sign_in(web).status_code == 200
    mine = chat(web, "Dana's question")
    other = second_browser(web)
    assert sign_in(other, "sam@globex.com").status_code == 200
    theirs = chat(other, "Sam's question")
    assert [c["id"] for c in listed(web)["conversations"]] == [mine]
    assert [c["id"] for c in listed(other)["conversations"]] == [theirs]
    assert other.patch(f"/api/conversations/{mine}", json={"title": "mine now"}).status_code == 404
    assert other.delete(f"/api/conversations/{mine}").status_code == 404
    assert listed(web)["conversations"][0]["title"] == "Dana's question"
    signed_out = second_browser(web)
    assert signed_out.get("/api/conversations").status_code == 401


def test_chats_made_before_the_history_list_get_a_title_and_time(tmp_path):
    store = Store(tmp_path / "old")
    conv = store.create_conversation()
    store.add_event(conv, "assistant", "text", text="Hi, I'm the hub.")
    store.add_event(conv, "user", "text", files=[{"name": "Acme RFP.docx", "size": 10}])
    db = store._db()
    db.execute("UPDATE conversations SET first_message = NULL, last_activity = NULL")
    Store._backfill_history(db)
    row = db.execute("SELECT first_message, last_activity FROM conversations WHERE id = ?", (conv,)).fetchone()
    last = db.execute("SELECT MAX(created_at) FROM events").fetchone()[0]
    assert row["first_message"] == "Acme RFP.docx" and row["last_activity"] == last
    assert json.loads(db.execute("SELECT payload FROM events WHERE n = 2").fetchone()["payload"])["files"]
    store.close()
