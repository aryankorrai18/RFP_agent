"""Reviewing drafted RFP answers one by one in the chat. Every decision is a call to the RFP assistant's own review route
(which scores it and learns from it); the hub shows the draft and evidence, collects the decision and logs who made it."""

from __future__ import annotations

import pytest

from agent_hub.auth import Auth
from agent_hub.intents import detect

from .conftest import make_engine
from .helpers import action, actions_of, cards, click, say, texts
from .test_rfp_flow import extracted

pytestmark = pytest.mark.anyio


async def drafted(engine, fakes, mode="partial"):  # noqa: ANN001, ANN201
    """A project with 6 requirements: 500 and 501 grounded, 502 flagged, 503 needs an expert, 504 failed, 505 never drafted."""
    fakes.rfp.draft_mode = mode
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv)
    out = await click(engine, conv, actions_of(extract)[0]["id"])
    return conv, out


def body_for(fakes, rid):  # noqa: ANN001, ANN201
    return [b for r, b in fakes.rfp.review_bodies if r == rid]


async def test_review_walks_through_each_answer_with_its_evidence_and_checks(engine, fakes):
    conv, out = await drafted(engine, fakes)
    first = await click(engine, conv, action(out, "Review them one by one")["id"])
    card = cards(first)[0]
    assert card["title"].endswith("5 answers left to review") and card["tone"] == "ok"
    headings = [s.get("heading") for s in card["sections"]]
    assert headings == ["Question", "Draft answer", "Evidence it cites"]  # the fake drafts cite no claims, so it says "None."
    assert [a["label"] for a in actions_of(first)] == ["Accept", "Edit", "Reject", "Skip for now"]
    assert fakes.rfp.model_calls == 1 + 6  # reviewing spends nothing: the 1 extraction and 6 drafts only


async def test_accept_edit_with_a_reason_reject_and_write_the_expert_answer(engine, fakes):
    conv, out = await drafted(engine, fakes)
    step = await click(engine, conv, action(out, "Review them one by one")["id"])
    step = await click(engine, conv, action(step, "Accept")["id"])                       # 500
    assert body_for(fakes, 500) == [{"action": "accepted"}] and "Accepted" in texts(step)
    step = await click(engine, conv, action(step, "Edit")["id"])                         # 501
    assert "Type the answer as it should go to the buyer" in texts(step)
    step = await say(engine, conv, "Yes. We support SAML 2.0 single sign-on with Okta and Azure AD.")
    assert "Why did you change it?" in texts(step) and body_for(fakes, 501) == []  # nothing saved until the reason is picked
    step = await click(engine, conv, action(step, "Outdated")["id"])
    assert body_for(fakes, 501) == [{"action": "edited", "final_text": "Yes. We support SAML 2.0 single sign-on with Okta and Azure AD.",
                                     "reason_tags": ["outdated"]}]
    assert cards(step)[0]["title"].startswith("R-003") and any("only partly supported" in b for s in cards(step)[0]["sections"]
                                                                   for b in s.get("bullets") or [])
    step = await click(engine, conv, action(step, "Reject")["id"])                       # 502, flagged
    step = await click(engine, conv, action(step, "Incorrect")["id"])
    assert body_for(fakes, 502) == [{"action": "rejected", "reason_tags": ["incorrect"]}] and "still needs an answer" in texts(step)
    assert [a["label"] for a in actions_of(step) if a["label"] in ("Write the answer", "Accept")] == ["Write the answer"]  # 503 needs an expert
    expert = await click(engine, conv, action(step, "Write the answer")["id"])
    assert "the answer your expert gives" in texts(expert)
    done = await say(engine, conv, "Priority-one incidents get a response within 1 hour, 24x7.")
    assert body_for(fakes, 503) == [{"action": "edited", "final_text": "Priority-one incidents get a response within 1 hour, 24x7."}]
    assert "approved answer the RFP assistant can reuse" in texts(done)


async def test_typed_text_under_review_is_never_read_as_a_command_or_by_a_model(engine, fakes):
    conv, out = await drafted(engine, fakes)
    step = await click(engine, conv, action(out, "Review them one by one")["id"])
    step = await click(engine, conv, action(step, "Accept")["id"])
    step = await click(engine, conv, action(step, "Edit")["id"])
    out = await say(engine, conv, "yes")  # anywhere else this would be read as a confirmation
    assert "Why did you change it?" in texts(out) and engine.load(conv)["awaiting"] is None
    await click(engine, conv, action(out, "Save without a reason")["id"])
    assert body_for(fakes, 501) == [{"action": "edited", "final_text": "yes"}]  # taken word for word


async def test_cancel_leaves_the_draft_as_it_was(engine, fakes):
    conv, out = await drafted(engine, fakes)
    step = await click(engine, conv, action(out, "Review them one by one")["id"])
    await click(engine, conv, action(step, "Edit")["id"])
    out = await say(engine, conv, "cancel")
    assert "left as it was" in texts(out) and body_for(fakes, 500) == []


async def test_an_answer_the_rfp_app_refuses_is_explained_and_can_be_typed_again(engine, fakes):
    conv, out = await drafted(engine, fakes)
    step = await click(engine, conv, action(out, "Review them one by one")["id"])
    for _ in range(3):  # accept 500 and 501, reject nothing: skip the flagged one
        label = "Accept" if any(a["label"] == "Accept" for a in actions_of(step)) else "Skip for now"
        step = await click(engine, conv, action(step, label)["id"])
    expert = await click(engine, conv, action(step, "Write the answer")["id"])
    bad = await say(engine, conv, "[SME input required: response time] Within one hour.")
    assert "didn't take that" in texts(bad) and engine.load(conv)["awaiting"]["type"] == "rfp_final_text"
    good = await say(engine, conv, "Within one hour, 24x7.")
    assert "Saved" in texts(good)


async def test_skipped_answers_come_back_and_a_finished_review_offers_the_export(engine, fakes):
    conv, out = await drafted(engine, fakes, mode="ok")
    step = await click(engine, conv, action(out, "Review them one by one")["id"])
    step = await click(engine, conv, action(step, "Skip for now")["id"])
    while any(a["label"] == "Accept" for a in actions_of(step)):
        step = await click(engine, conv, action(step, "Accept")["id"])
    again = await click(engine, conv, action(step, "Review the skipped ones")["id"])
    done = await click(engine, conv, action(again, "Accept")["id"])
    assert any(c["title"] == "Your response is ready" for c in cards(done))


async def test_saying_review_the_answers_starts_it_and_every_decision_is_logged(fakes, tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    eng = make_engine(fakes, tmp_path / "rv-data")
    eng.auth = Auth(eng.store)
    eng.auth.create_company("Acme", {"deals": ["ws-demo"], "rfp": ["ws-acme"]})
    eng.auth.create_user("dana@acme.com", "correct horse battery", "Acme")
    user = eng.auth.user_by_id(eng.store.sql_one("SELECT id FROM users")["id"])
    fakes.rfp.draft_mode = "partial"
    conv = eng.new_conversation(user)
    _, extract = await extracted(eng, conv)
    await click(eng, conv, actions_of(extract)[0]["id"])
    step = await say(eng, conv, "review the answers")
    await click(eng, conv, action(step, "Accept")["id"])
    entries = [e for e in eng.auth.list_audit(10) if e["action"] == "rfp.review"]
    assert entries and entries[0]["actor"] == "dana@acme.com" and "accepted" in entries[0]["detail"]
    eng.store.close()


def test_review_requests_are_recognised():
    for text in ("review the answers", "let's go through the drafts", "next answer", "check the rest"):
        assert detect(text, False).kind == "rfp_review", text
    assert detect("review the Acme deal", False).kind != "rfp_review"
