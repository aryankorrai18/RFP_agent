"""Feeding the agents' memory from the chat: company facts and past proposals (RFP), plays (Deal). Nothing is saved without a
click, facts and plays cost nothing, a past proposal costs one model call and its pairs are shown before they are kept."""

from __future__ import annotations

import json

import pytest

from agent_hub.auth import Auth
from agent_hub.flows.memory_feed import parse_facts, parse_plays, proposal_slots
from agent_hub.intents import memory_request

from .conftest import make_engine
from .helpers import action, actions_of, all_text, cards, click, say, texts

pytestmark = pytest.mark.anyio

FACT_SHEET = json.dumps({"company": "Acme Security", "facts": [
    {"id": "FACT-001", "topic": "Certifications", "statement": "We hold ISO 27001 certification.", "valid_to": "2027-03-31"},
    {"topic": "Support", "statement": "Support is available 24x7 for priority-one issues."}]}).encode()


# -- reading what was sent (no model) ------------------------------------------------------------------------------

def test_facts_are_read_from_a_fact_sheet_a_table_or_one_per_line():
    facts, company = parse_facts(FACT_SHEET.decode(), "facts.json")
    assert company == "Acme Security" and [f["valid_to"] for f in facts] == ["2027-03-31", None]
    table, _ = parse_facts("topic,statement,valid_until\nHosting,Data is hosted in the EU.,\nSOC,We have a SOC 2 Type II report.,2026-12-31\n", "f.csv")
    assert [(f["topic"], f["valid_to"]) for f in table] == [("Hosting", None), ("SOC", "2026-12-31")]
    lines, _ = parse_facts("Our facts\n- Certifications: ISO 27001, valid until 2027-03-31\n- We were founded in 2018.\n\n# notes\nSecurity:\n")
    assert [(f["topic"], f["statement"], f["valid_to"]) for f in lines] == [
        ("Certifications", "ISO 27001, valid until 2027-03-31", "2027-03-31"), ("", "We were founded in 2018.", None)]
    one_line, _ = parse_facts("- We hold ISO 27001 - Support is 24x7; We host in the EU")
    assert len(one_line) == 3


def test_plays_are_read_with_their_objections_and_category():
    plays = parse_plays("- Security review pack: send our SOC 2 report [objections: security, single sign-on] (category: security)\n"
                        "- Exec sponsor call - bring our VP\n")
    assert plays[0] == {"name": "Security review pack", "description": "send our SOC 2 report", "category": "security",
                        "addresses": ["security_review", "sso"]}
    assert plays[1]["name"] == "Exec sponsor call" and plays[1]["description"] == "bring our VP" and plays[1]["category"] == "process"
    table = parse_plays("name,description,objections\nPilot,Two-week pilot,pricing; timeline\n", "plays.csv")
    assert table[0]["addresses"] == ["pricing", "timeline"]


def test_a_past_proposals_details_are_read_from_the_sentence():
    assert proposal_slots("past proposal we lost on tech fit for Acme Bank (banking), submitted 2025-03-10") == {
        "result": "lost", "loss_reason": "technical fit", "client": "Acme Bank", "industry": "banking", "submitted_on": "2025-03-10"}
    assert proposal_slots("old bid we won in March 2024") == {"result": "won", "submitted_on": "2024-03-01"}
    assert proposal_slots("add this to the library")== {}


@pytest.mark.parametrize("text,files,kind", [
    ("Add these company facts:\n- ISO 27001", False, "mem_facts"), ("here is our fact sheet", True, "mem_facts"),
    ("add our sales plays", False, "mem_plays"), ("these are our plays", True, "mem_plays"),
    ("past proposal we won for Acme Bank", True, "mem_proposal"), ("add this to the library, we lost it on technical fit", True, "mem_proposal"),
    ("answer this RFP", True, None), ("what plays should I run on Acme?", False, None), ("We lost the Acme RFP because of price", False, None),
    ("New deal Acme renewal at Acme Ltd", True, None), ("what are the facts about the Acme deal?", False, None)])
def test_requests_to_feed_the_memory_are_recognised_and_others_are_not(text, files, kind):
    assert memory_request(text, files) == kind


# -- company facts ---------------------------------------------------------------------------------------------------

async def test_facts_are_shown_then_saved_only_on_a_click_and_cost_nothing(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "here is our fact sheet", [("facts.json", FACT_SHEET)])
    assert cards(out)[0]["title"] == "2 company facts to add" and fakes.rfp.company_puts == []
    save = action(out, "Save 2 facts")
    assert save.get("cost") == "free"
    done = await click(engine, conv, save["id"])
    sheet = fakes.rfp.company_sheets["ws-acme"]
    assert sheet["company"] == "Acme Security" and [f["id"] for f in sheet["facts"]] == ["FACT-001", "FACT-002"]
    assert "Saved 2 new facts" in texts(done) and fakes.rfp.model_calls == 0 and fakes.all_spending() == []


async def test_typed_facts_are_added_to_the_existing_ones_and_repeats_are_skipped(engine, fakes):
    fakes.rfp.company_sheets["ws-acme"] = {"company": "Acme Security", "facts": [{"id": "FACT-001", "topic": "", "statement": "We were founded in 2018.", "valid_to": None}]}
    conv = engine.new_conversation()
    out = await say(engine, conv, "Add these company facts:\n- We were founded in 2018.\n- Data is hosted in the EU.")
    assert cards(out)[0]["title"] == "1 company fact to add" and "1 already there" in cards(out)[0]["subtitle"]
    await click(engine, conv, action(out, "Save 1 fact")["id"])
    assert [f["statement"] for f in fakes.rfp.company_sheets["ws-acme"]["facts"]] == ["We were founded in 2018.", "Data is hosted in the EU."]


async def test_asking_without_facts_asks_for_them_and_the_next_message_is_read(engine, fakes):
    conv = engine.new_conversation()
    assert "Send the facts" in texts(await say(engine, conv, "I want to add our company facts"))
    out = await say(engine, conv, "We host all data in the EU\nSupport is 24x7")
    assert cards(out)[0]["title"] == "2 company facts to add"


async def test_the_read_only_sample_workspace_is_refused(engine, fakes):
    fakes.rfp.locked_workspaces.add("ws-acme")
    out = await say(engine, engine.new_conversation(), "Add these company facts: We are ISO 27001 certified")
    assert "built-in sample" in texts(out) and not actions_of(out) and fakes.rfp.company_puts == []


async def test_cancelling_saves_nothing(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "Add these company facts: We are ISO 27001 certified")
    await click(engine, conv, action(out, "Not now")["id"])
    assert fakes.rfp.company_puts == []


# -- past proposals ----------------------------------------------------------------------------------------------------

PROPOSAL = ("Acme bid 2025.docx", b"fake proposal bytes")


async def test_a_past_proposal_costs_one_call_and_its_pairs_are_kept_only_on_a_click(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "past proposal we lost on technical fit for Acme Bank (banking), submitted 2025-03-10", [PROPOSAL])
    yes = action(out, "Yes, read it")
    assert yes["cost"] == "1 model call" and "lost (technical fit)" in all_text(out) and fakes.rfp.model_calls == 0
    read = await click(engine, conv, yes["id"])
    assert fakes.rfp.model_calls == 1 and fakes.rfp.proposal_bodies[0] == {
        "filename": "Acme bid 2025.docx", "client": "Acme Bank", "industry": "banking", "submitted_on": "2025-03-10",
        "result": "lost", "loss_reason": "technical fit"}
    assert cards(read)[0]["title"] == "2 question-and-answer pairs in Acme bid 2025.docx" and fakes.rfp.kept_answers == []
    kept = await click(engine, conv, action(read, "Keep all 2")["id"])
    assert len(fakes.rfp.kept_answers) == 2 and "Added 2 approved answers" in texts(kept)


async def test_when_the_outcome_is_not_said_it_is_asked_for(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "add this past proposal to the library", [PROPOSAL])
    assert "won or lost" in texts(out) and fakes.rfp.model_calls == 0
    confirm = await click(engine, conv, action(out, "Won")["id"])
    await click(engine, conv, action(confirm, "Yes, read it")["id"])
    assert fakes.rfp.proposal_bodies[0]["result"] == "won"


async def test_the_file_can_come_after_the_request(engine, fakes):
    conv = engine.new_conversation()
    assert "Attach the past proposal" in texts(await say(engine, conv, "I want to import a past proposal we won"))
    out = await say(engine, conv, "", [PROPOSAL])
    assert action(out, "Yes, read it") and "won" in all_text(out)


async def test_a_proposal_that_cannot_be_read_adds_nothing_and_a_discard_is_honoured(engine, fakes):
    conv = engine.new_conversation()
    fakes.rfp.proposal_mode = "fail"
    out = await say(engine, conv, "past proposal we won", [PROPOSAL])
    failed = await click(engine, conv, action(out, "Yes, read it")["id"])
    assert "couldn't read" in all_text(failed) and "no readable text" in all_text(failed) and fakes.rfp.kept_answers == []
    fakes.rfp.proposal_mode = "ok"
    out = await say(engine, conv, "another past proposal we won", [("other.docx", b"other bytes")])
    read = await click(engine, conv, action(out, "Yes, read it")["id"])
    gone = await click(engine, conv, action(read, "Discard")["id"])
    assert "Nothing was added" in texts(gone) and fakes.rfp.proposals[2]["status"] == "discarded" and fakes.rfp.kept_answers == []


async def test_a_confirmed_read_cannot_be_repeated_by_a_double_click(engine, fakes):
    conv = engine.new_conversation()
    yes = action(await say(engine, conv, "past proposal we won", [PROPOSAL]), "Yes, read it")["id"]
    await click(engine, conv, yes)
    again = await click(engine, conv, yes)
    assert fakes.rfp.model_calls == 1 and "already done" in texts(again)


# -- plays ----------------------------------------------------------------------------------------------------------------

async def test_plays_are_shown_then_saved_free_and_unknown_objections_are_reported(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "add our sales plays:\n- Security pack: send the SOC 2 report [objections: security, budget freeze]\n- Pilot: two weeks")
    assert cards(out)[0]["title"] == "2 plays to save" and fakes.deal.play_bodies == []
    save = action(out, "Save 2 plays")
    assert save["cost"] == "free"
    done = await click(engine, conv, save["id"])
    assert [p["name"] for p in fakes.deal.plays_by_ws["ws-demo"]] == ["Security pack", "Pilot"]
    assert "budget_freeze" in texts(done) and fakes.deal.model_calls == 0


# -- a company's chat feeds only its own workspaces -----------------------------------------------------------------------

async def test_a_company_feeds_its_own_workspaces_and_the_save_is_logged(fakes, tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    eng = make_engine(fakes, tmp_path / "feed-data")
    eng.auth = Auth(eng.store)
    eng.auth.create_company("Globex", {"deals": ["ws-other"], "rfp": ["ws-globex"]})
    eng.auth.create_user("gus@globex.com", "correct horse battery", "Globex")
    user = eng.auth.user_by_id(eng.store.sql_one("SELECT id FROM users")["id"])
    conv = eng.new_conversation(user)
    out = await say(eng, conv, "Add these company facts: Globex hosts data in the EU")
    await click(eng, conv, action(out, "Save 1 fact")["id"])
    assert set(fakes.rfp.company_sheets) == {"ws-globex"}
    out = await say(eng, conv, "add our sales plays: Pilot: two weeks")
    await click(eng, conv, action(out, "Save 1 play")["id"])
    assert set(fakes.deal.plays_by_ws) == {"ws-other"}
    actions = [e["action"] for e in eng.auth.list_audit(10)]
    assert "rfp.facts" in actions and "deal.plays" in actions
    eng.store.close()


# -- when the hub's model is the one that understood the request ---------------------------------------------------------

async def test_the_model_can_route_loosely_worded_requests_to_the_same_flows(fakes, tmp_path):
    from agent_hub.planner import Planner

    from .fake_llm import FakeHubLLM, tool

    llm = FakeHubLLM([tool("add_company_facts"), tool("add_plays"), tool("usage_report")])
    eng = make_engine(fakes, tmp_path / "planned-data", planner=Planner(lambda: llm))
    conv = eng.new_conversation()
    assert "Send the facts" in texts(await say(eng, conv, "ok so what we can officially claim about ourselves is coming"))
    assert "Send your plays" in texts(await say(eng, conv, "the moves my reps make to close"))
    assert cards(await say(eng, conv, "how big has this thing gotten"))[0]["title"] == "Usage across companies"
    assert fakes.rfp.company_puts == [] and fakes.deal.play_bodies == []
    eng.store.close()
