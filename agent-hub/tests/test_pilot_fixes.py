"""Fixes for what the Aiden.AI pilot found (2026-10-08): a deal's industry and size from the notes or
plain words, completing them later, a loose reply to the hub's own question, deal events in the activity log, the
administrator's welcome, and quote marks around a pasted answer."""

from __future__ import annotations

import pytest

from agent_hub.auth import Auth
from agent_hub.planner import Planner

from .conftest import make_engine
from .fake_llm import FakeHubLLM, tool
from .helpers import action, actions_of, cards, click, say, texts
from .test_review_flow import drafted

pytestmark = pytest.mark.anyio

NOTES = ("2026-04-08_discovery_call.md", b"# Call notes: Lumen Grid Software\n\nSeller: Aiden.AI. Deal: SYN Lumen. "
                                         b"Industry: software. Segment: smb.\n\n## 2026-04-08, discovery call\n- pricing came up\n")
PLAIN = ("2026-04-08_call.txt", b"They asked about pricing.")


def rows(out: list[dict]) -> dict:
    return dict(cards(out)[0]["sections"][0]["rows"])


# -- a deal's industry and size --------------------------------------------------------------------------------------------

async def test_industry_and_size_are_read_from_the_notes_and_shown_with_their_source(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "New deal SYN Lumen at Lumen Grid", [NOTES])
    assert fakes.deal.created_bodies[-1]["industry"] == "software" and fakes.deal.created_bodies[-1]["segment"] == "smb"
    assert rows(out)["Industry"] == "software (from the notes)" and rows(out)["Segment"] == "smb (from the notes)"
    assert "not given" not in texts(out) and fakes.all_spending() == []


async def test_industry_and_size_said_in_plain_words_are_kept_without_a_model(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "New deal SYN Meridian at Meridian Supply, they're a big manufacturing company", [PLAIN])
    body = fakes.deal.created_bodies[-1]
    assert body["industry"] == "manufacturing" and body["segment"] == "enterprise"


async def test_missing_details_are_named_with_how_to_add_them(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "New deal Lumen at Lumen Grid", [PLAIN])
    assert rows(out)["Industry"] == "not given" and rows(out)["Segment"] == "not given"
    assert "industry and segment are not given" in texts(out) and "D-007 is a small software company" in texts(out)


async def test_a_deals_details_can_be_completed_later_in_plain_words(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "New deal Lumen at Lumen Grid", [PLAIN])
    out = await say(engine, conv, "D-007 is a small software company")
    assert fakes.deal.update_bodies == [{"deal_id": 7, "industry": "software", "segment": "smb"}]
    assert "Updated D-007" in texts(out) and fakes.all_spending() == []


async def test_the_model_can_complete_details_too(fakes, tmp_path):
    llm = FakeHubLLM([tool("update_deal", deal="D-002", industry="Retail", segment="Mid-Market")])
    eng = make_engine(fakes, tmp_path / "upd", planner=Planner(lambda: llm))
    out = await say(eng, eng.new_conversation(), "juniper's actually a mid sized retailer")
    assert fakes.deal.update_bodies[-1] == {"deal_id": 2, "industry": "retail", "segment": "mid_market"}
    assert "Updated D-002" in texts(out)
    eng.store.close()


async def test_a_loose_reply_to_the_hubs_own_question_uses_the_models_reading(fakes, tmp_path):
    llm = FakeHubLLM([tool("new_deal"), tool("answer_pending", account="Lumen Grid Software", industry="software", segment="smb")])
    eng = make_engine(fakes, tmp_path / "slot", planner=Planner(lambda: llm))
    conv = eng.new_conversation()
    ask = await say(eng, conv, "i want to add a deal")
    assert "Which account" in texts(ask) or "What is the deal called" in texts(ask)
    out = await say(eng, conv, "lumen grid, they are a small software company", [PLAIN])
    body = fakes.deal.created_bodies[-1]
    assert body["account"] == "Lumen Grid Software" and body["name"] == "Lumen Grid Software"
    assert body["industry"] == "software" and body["segment"] == "smb" and "Created D-007" in texts(out)
    eng.store.close()


# -- the activity log shows deal events ------------------------------------------------------------------------------------

async def test_deal_creation_reads_and_outcomes_are_in_the_activity_log(fakes, tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    eng = make_engine(fakes, tmp_path / "audit")
    eng.auth = Auth(eng.store)
    eng.auth.create_company("Acme", {"deals": ["ws-demo"], "rfp": ["ws-acme"]})
    eng.auth.create_user("dana@acme.com", "correct horse battery", "Acme")
    user = eng.auth.user_by_id(eng.store.sql_one("SELECT id FROM users")["id"])
    conv = eng.new_conversation(user)
    created = await say(eng, conv, "New deal Lumen at Lumen Grid", [PLAIN])
    ask = await click(eng, conv, action(created, "Read D-007")["id"])
    await click(eng, conv, action(ask, "Yes, read it")["id"])
    ready = await say(eng, conv, "Lumen was won")
    await click(eng, conv, action(ready, "Yes, record it")["id"])
    logged = {e["action"] for e in eng.auth.list_audit(20) if e["actor"] == "dana@acme.com"}
    assert {"deal.create", "deal.read", "deal.outcome"} <= logged
    eng.store.close()


# -- the administrator's chat -------------------------------------------------------------------------------------------------

async def test_an_administrators_new_chat_offers_admin_examples_not_deal_ones(fakes, tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    eng = make_engine(fakes, tmp_path / "op")
    eng.auth = Auth(eng.store)
    eng.auth.create_first_admin("ops@hub.example", "correct horse battery", "Ops")
    admin = eng.auth.user_by_id(eng.store.sql_one("SELECT id FROM users")["id"])
    conv = eng.new_conversation(admin)
    welcome = eng.store.events(conv, 0)
    labels = [a["label"] for a in actions_of(welcome)]
    assert "How many companies are using this?" in labels and "Create a new deal from these call notes" not in labels
    assert "administrator" in texts(welcome)
    eng.store.close()


# -- an answer pasted from a spreadsheet cell -------------------------------------------------------------------------------

async def test_quote_marks_wrapping_a_pasted_answer_are_dropped(engine, fakes):
    conv, out = await drafted(engine, fakes)
    step = await click(engine, conv, action(out, "Review them one by one")["id"])
    step = await click(engine, conv, action(step, "Accept")["id"])
    await click(engine, conv, action(step, "Edit")["id"])
    asked = await say(engine, conv, '"Yes. We support SAML 2.0 single sign-on."')
    await click(engine, conv, action(asked, "Save without a reason")["id"])
    assert [b for r, b in fakes.rfp.review_bodies if r == 501] == [{"action": "edited", "final_text": "Yes. We support SAML 2.0 single sign-on."}]
