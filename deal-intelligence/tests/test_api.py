"""The whole product through HTTP, offline: the demo seed, a brief, an outcome that changes the next
brief, the before/after comparison and the Memory page. Hindsight and the model are fakes."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from deal_intelligence.api.v1 import demo, lessons, outcomes
from deal_intelligence.api.v1.db import Deal, Interaction
from deal_intelligence.main import app
from tests.builders import make_context
from tests.fake_llm import FakeLLM
from tests.fakes import FakeLessons, FakeMemory


def wait_job(client: TestClient, job_id: int, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.03)
    raise AssertionError(f"job {job_id} did not finish")


@pytest.fixture
def seeded(tmp_path):
    llm = FakeLLM()
    ctx = make_context(tmp_path, memory=FakeMemory(), lessons=FakeLessons(), llm=llm)
    seeded = demo.seed_demo(ctx)
    outcomes.rebuild_play_stats(ctx.db)
    lessons.collect_lessons(ctx.db)
    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as client:
            yield client, ctx, seeded, llm
    finally:
        app.state.v1_factory = None


def test_demo_workspace_is_listed_and_costs_no_model_calls(seeded):
    client, _ctx, info, llm = seeded
    deals = client.get("/v1/deals").json()["deals"]
    assert info["closed"] == 19 and info["open"] == 5 and len(deals) == 24
    assert llm.calls == []  # seeding and listing never touch the model
    status = client.get("/v1/status").json()
    assert status["counts"]["won"] == 9 and status["counts"]["lost"] == 10 and status["counts"]["open"] == 5
    assert status["hindsight"]["healthy"] is True
    assert len(client.get("/v1/plays").json()["plays"]) == 10


def test_demo_deal_shows_flags_and_memory_evidence(seeded):
    client, _ctx, info, _llm = seeded
    deal_id = int(info["demo_deal"][2:])
    detail = client.get(f"/v1/deals/{deal_id}").json()
    assert detail["result"] == "open" and detail["signals"]["objections"]
    assert any(f["code"] == "overdue_promise" for f in detail["flags"])
    rec = client.get(f"/v1/deals/{deal_id}/recommendations?mode=hindsight").json()
    assert rec["n_closed"] == 19 and rec["similar"]
    assert any(w["objection_type"] == "sso" for w in rec["warnings"])
    assert rec["plays"] and rec["plays"][0]["play_code"] in {"PLAY-01", "PLAY-10"}


def test_a_recorded_loss_changes_the_next_brief(seeded):
    client, _ctx, info, llm = seeded
    demo_id, closable_id = int(info["demo_deal"][2:]), int(info["closable_deal"][2:])
    first = client.post(f"/v1/deals/{demo_id}/brief", json={"mode": "hindsight"})
    assert first.status_code == 202
    assert wait_job(client, first.json()["job_id"])["status"] == "completed"
    before = client.get(f"/v1/deals/{demo_id}/brief?mode=hindsight").json()
    assert before["status"] == "ready" and before["content"]["next_steps"]
    assert before["content"]["warnings"]

    outcome = client.put(f"/v1/deals/{closable_id}/outcome", json={
        "result": "lost", "loss_reason": "unresolved_objection", "plays_used": ["PLAY-05"],
    })
    assert outcome.status_code == 200
    body = outcome.json()
    assert body["deal"]["result"] == "lost" and body["lessons_added"] >= 1 and body["credit"]

    second = client.post(f"/v1/deals/{demo_id}/brief", json={"mode": "hindsight"})
    assert wait_job(client, second.json()["job_id"])["status"] == "completed"
    after = client.get(f"/v1/deals/{demo_id}/brief?mode=hindsight").json()
    assert after["id"] != before["id"] and after["memory_state"] != before["memory_state"]

    diff = client.get(f"/v1/deals/{demo_id}/brief-diff", params={"from": before["id"], "to": after["id"]}).json()
    assert diff["added_similar"] or diff["changed_warnings"]  # the Larkfield loss now counts, with its D- id
    history = client.get(f"/v1/deals/{demo_id}/briefs").json()["briefs"]
    assert len(history) == 2 and "content" not in history[0]
    assert client.get("/v1/deals/9999/brief").json()["error"]["code"] in ("no_brief", "not_found")


def test_an_unread_deal_needs_its_signals_first(seeded):
    client, _ctx, _info, _llm = seeded
    tidewater = next(d for d in client.get("/v1/deals").json()["deals"] if d["account"] == "Tidewater Freight")
    assert tidewater["signals_status"] == "none"
    blocked = client.post(f"/v1/deals/{tidewater['id']}/brief", json={})
    assert blocked.status_code == 409 and blocked.json()["error"]["code"] == "signals_missing"

    started = client.post(f"/v1/deals/{tidewater['id']}/signals")
    assert started.status_code == 202
    assert wait_job(client, started.json()["job_id"])["status"] == "completed"
    assert client.get(f"/v1/deals/{tidewater['id']}").json()["signals_status"] == "ready"


def test_before_after_comparison_has_three_arms(seeded):
    client, _ctx, info, _llm = seeded
    deal_id = int(info["demo_deal"][2:])
    started = client.post(f"/v1/deals/{deal_id}/comparison", json={})
    assert started.status_code == 202
    assert wait_job(client, started.json()["job_id"])["status"] == "completed"
    comparison = client.get(f"/v1/deals/{deal_id}/comparison").json()
    assert set(comparison["arms"]) == {"none", "longctx", "hindsight"}
    assert comparison["prompt_hash_same"] is True and comparison["n_closed"] == 19
    assert set(comparison["differs"]["tokens"]) == {"none", "longctx", "hindsight"}


def test_memory_page_endpoints(seeded):
    client, _ctx, info, _llm = seeded
    closable_id = int(info["closable_deal"][2:])
    client.put(f"/v1/deals/{closable_id}/outcome", json={"result": "won", "plays_used": ["PLAY-02"]})
    assert client.get("/v1/memory/journal").json()["events"]
    lessons_view = client.get("/v1/memory/lessons").json()
    assert lessons_view["lessons"] and lessons_view["counts"]["pending"] + lessons_view["counts"]["retained"] > 0
    stats = {p["code"]: p for p in client.get("/v1/memory/play-stats").json()["plays"]}
    assert stats["PLAY-01"]["times_used"] >= 3 and stats["PLAY-05"]["lost_other"] >= 1
    history = client.get("/v1/memory/history").json()["deals"]
    assert len(history) == 20 and history[0]["summary"]
    assert client.get("/v1/memory/playbook").json()["state"] in ("missing", "ready")
    assert client.post("/v1/sync").json()["failed"] == 0


def test_omitting_plays_used_keeps_the_recorded_plays_and_an_empty_list_clears_them(seeded):
    client, _ctx, info, _llm = seeded
    deal_id = int(info["closable_deal"][2:])
    url = f"/v1/deals/{deal_id}/outcome"

    def recorded_plays() -> list[str]:
        return [p["code"] if isinstance(p, dict) else p for p in client.get(f"/v1/deals/{deal_id}").json()["signals"]["plays_used"]]

    first = client.put(url, json={"result": "won", "plays_used": ["PLAY-02"]})
    assert first.status_code == 200 and first.json()["plays_used"] == ["PLAY-02"] and recorded_plays() == ["PLAY-02"]
    omitted = client.put(url, json={"result": "won"})  # a client that does not send the field
    assert omitted.status_code == 200 and omitted.json()["plays_used"] == ["PLAY-02"] and recorded_plays() == ["PLAY-02"]
    cleared = client.put(url, json={"result": "won", "plays_used": []})  # an explicit "none"
    assert cleared.status_code == 200 and cleared.json()["plays_used"] == [] and recorded_plays() == []


def test_bad_requests_are_explained(seeded):
    client, _ctx, info, _llm = seeded
    closable_id = int(info["closable_deal"][2:])
    lost_without_reason = client.put(f"/v1/deals/{closable_id}/outcome", json={"result": "lost"})
    assert lost_without_reason.status_code == 422
    assert client.get("/v1/deals/9999").status_code == 404
    assert client.post("/v1/deals/1/brief", json={"mode": "magic"}).status_code == 422


def test_create_a_deal_from_a_note_and_a_file(seeded):
    client, _ctx, _info, _llm = seeded
    created = client.post(
        "/v1/deals",
        data={"name": "Pilot", "account": "Fernhill Dairy", "industry": "food", "segment": "smb", "amount": "40000"},
        files=[("files", ("2026-10-01_call_note.md", b"Call with Dana. She wants a pilot before Q1 and asked about SSO.", "text/markdown"))],
    )
    assert created.status_code == 201
    deal = created.json()
    assert deal["signals_status"] == "none" and deal["interactions"][0]["kind"] == "call_note"
    noted = client.post(f"/v1/deals/{deal['id']}/notes", json={"kind": "email", "text": "Thanks for the call, sending the quote."})
    assert len(noted.json()["interactions"]) == 2
    assert client.delete(f"/v1/deals/{deal['id']}").json() == {"ok": True}
    assert client.get(f"/v1/deals/{deal['id']}").status_code == 404


def test_a_bad_file_leaves_no_deal_behind_and_a_valid_create_still_works(seeded):
    client, ctx, _info, _llm = seeded

    def counts() -> tuple[int, int]:
        with ctx.db.session() as session:
            return session.query(Deal).count(), session.query(Interaction).count()

    before = counts()
    fields = {"name": "Pilot", "account": "Fernhill Dairy", "industry": "food", "segment": "smb", "amount": "40000"}
    good = ("files", ("2026-10-01_call_note.md", b"Call with Dana. She wants a pilot before Q1.", "text/markdown"))

    unsupported = client.post("/v1/deals", data=fields, files=[good, ("files", ("tool.exe", b"MZ", "application/octet-stream"))])
    assert unsupported.status_code == 415 and unsupported.json()["error"]["code"] == "unsupported_file_type"
    unreadable = client.post("/v1/deals", data=fields, files=[good, ("files", ("broken.pdf", b"not a pdf", "application/pdf"))])
    assert unreadable.status_code == 422 and unreadable.json()["error"]["code"] == "unreadable_file"
    assert counts() == before  # no empty deal and no interaction from the valid first file

    created = client.post("/v1/deals", data=fields, files=[good])
    assert created.status_code == 201 and created.json()["interactions"][0]["kind"] == "call_note"
    assert counts() == (before[0] + 1, before[1] + 1)


def test_a_deals_details_can_be_completed_later_for_free(seeded):
    client, _ctx, info, llm = seeded
    deal_id = int(info["demo_deal"][2:])
    out = client.patch(f"/v1/deals/{deal_id}", json={"industry": "Software", "segment": "smb"})
    assert out.status_code == 200 and out.json()["industry"] == "Software" and out.json()["segment"] == "smb"
    assert client.patch(f"/v1/deals/{deal_id}", json={"segment": "huge"}).status_code == 422
    assert client.patch(f"/v1/deals/{deal_id}", json={"name": "  "}).status_code == 422
    assert client.patch("/v1/deals/99999", json={"industry": "x"}).status_code == 404
    unchanged = client.patch(f"/v1/deals/{deal_id}", json={}).json()
    assert unchanged["industry"] == "Software" and llm.calls == []
