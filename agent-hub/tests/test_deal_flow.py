"""The deal flows end to end against the fake Deal Intelligence app."""

from __future__ import annotations

import pytest

from .helpers import PDF, action, actions_of, all_text, assistant, cards, click, links, say, texts

pytestmark = pytest.mark.anyio


async def test_brief_an_existing_deal_whose_signals_are_ready(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Brief me on the Cedarline deal")
    text = texts(ask)
    assert "D-001 Cedarline Renewal" in text and "Halcyon Demo" in text
    assert "signals were already read" in text and "1 model call" in text
    assert fakes.all_spending() == []  # nothing is spent until the person says yes

    done = await click(engine, conv, action(ask, "Yes")["id"])
    assert fakes.deal.model_calls == 1
    assert fakes.deal.spending_calls() == [("POST", "/v1/deals/1/brief")]  # no signals call: already ready
    kinds = [e["kind"] for e in assistant(done)]
    assert "progress" in kinds and kinds[-1] == "cards"
    brief = cards(done)[0]
    assert brief["title"] == "Brief: D-001 Cedarline Renewal (Cedarline Systems)"
    headings = [s.get("heading") for s in brief["sections"]]
    assert "Summary" in headings and "Likely to backfire" in headings and "Recommended next steps" in headings
    assert brief["sections"][0]["chips"] == ["INT-0001"]
    assert links(done)[0]["url"] == "http://127.0.0.1:8002/#/deals/1"
    assert fakes.all_forbidden() == []


async def test_brief_a_deal_whose_signals_are_not_read_yet_costs_two_calls(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "brief me on Juniper")
    assert "2 model calls" in texts(ask) and "read" in texts(ask)
    assert actions_of(ask)[0]["cost"] == "2 model calls"
    done = await click(engine, conv, action(ask, "Yes")["id"])
    assert [c[1] for c in fakes.deal.spending_calls()] == ["/v1/deals/2/signals", "/v1/deals/2/brief"]
    assert fakes.deal.model_calls == 2
    assert cards(done)[0]["title"].startswith("Brief: D-002")
    progress = [e["progress"]["label"] for e in assistant(done) if e["kind"] == "progress"]
    assert any(p.startswith("Reading") for p in progress) and any(p.startswith("Writing the brief") for p in progress)


async def test_a_deal_without_emails_or_notes_is_not_briefed(engine, fakes):
    conv = engine.new_conversation()
    reply = await say(engine, conv, "Brief me on Brightwater")
    assert "no emails or notes yet" in texts(reply) and not actions_of(reply)
    assert fakes.all_spending() == []


async def test_ambiguous_deal_names_ask_which_one(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Brief me on Larkfield")
    assert "Did you mean" in texts(ask)
    labels = [a["label"] for a in actions_of(ask)]
    assert any("D-003" in label for label in labels) and any("D-004" in label for label in labels)
    confirm = await click(engine, conv, actions_of(ask)[1]["id"])
    assert "D-004 Larkfield Security Add-on" in texts(confirm) and "2 model calls" in texts(confirm)
    assert fakes.all_spending() == []
    conv2 = engine.new_conversation()  # a typed answer works too
    await say(engine, conv2, "Brief me on Larkfield")
    again = await say(engine, conv2, "the second one")
    assert "D-004" in texts(again)


async def test_unknown_deal_lists_what_exists(engine, fakes):
    conv = engine.new_conversation()
    reply = await say(engine, conv, "Brief me on the Zebra Logistics deal")
    assert "couldn't find a deal matching" in texts(reply) and "D-001 Cedarline Renewal" in texts(reply)
    assert fakes.all_spending() == []


async def test_job_failure_is_explained_with_the_suggested_action(engine, fakes):
    fakes.deal.fail_jobs = "brief"
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Brief me on Cedarline")
    done = await click(engine, conv, action(ask, "Yes")["id"])
    error = next(e for e in done if e["kind"] == "error")
    assert "The model quota is used up" in error["text"] and "What to do: Wait for the quota to reset" in error["text"]
    assert "Models are changed in Deal Intelligence itself" in error["text"]
    assert not any(c["title"].startswith("Brief:") for c in cards(done))
    assert fakes.deal.forbidden_calls() == []


async def test_create_a_deal_from_files_and_slots_uses_no_model_calls(engine, fakes):
    conv = engine.new_conversation()
    created = await say(engine, conv, "New deal Tidewater at Tidewater Freight worth $120k", [PDF, ("notes.txt", b"call notes")])
    assert fakes.deal.created_bodies == [
        {"name": "Tidewater", "account": "Tidewater Freight", "amount": "120000", "segment": None, "files": [PDF[0], "notes.txt"],
         "industry": "logistics"}]  # "Freight" in the account says logistics; the card marks it "from your message"
    assert "Created D-007 Tidewater (Tidewater Freight)" in texts(created) and "no model calls" in texts(created)
    assert fakes.deal.model_calls == 0 and fakes.all_spending() == []
    assert action(created, "Brief me on D-007")
    assert links(created)[0]["url"].endswith("/#/deals/7")


async def test_files_with_no_instruction_ask_which_deal_then_slots(engine, fakes):
    conv = engine.new_conversation()
    first = await say(engine, conv, "", [PDF])
    assert "Which deal are these files for?" in texts(first) and fakes.deal.created_bodies == []
    second = await say(engine, conv, "New deal Tidewater")
    assert "Which account" in texts(second)
    third = await say(engine, conv, "Tidewater Freight")
    assert "Created D-007" in texts(third)
    assert fakes.deal.created_bodies[0]["files"] == [PDF[0]]


async def test_a_bad_file_leaves_no_deal_behind(engine, fakes):
    conv = engine.new_conversation()
    reply = await say(engine, conv, "New deal Tidewater at Tidewater Freight", [("bad.txt", b"CORRUPT data")])
    assert "could not be read" in all_text(reply)
    assert fakes.deal.created_bodies == [] and len(fakes.deal.deals) == 6


async def test_files_of_the_wrong_type_get_a_friendly_message(engine, fakes):
    conv = engine.new_conversation()
    reply = await say(engine, conv, "New deal Tidewater at Tidewater Freight", [("virus.exe", b"MZ")])
    assert "can't read .exe files" in texts(reply)


async def test_add_a_note_and_add_files_to_an_existing_deal(engine, fakes):
    conv = engine.new_conversation()
    note = await say(engine, conv, "Add a call note to Juniper: they want SSO before Q3")
    assert fakes.deal.note_bodies == [{"kind": "call_note", "text": "they want SSO before Q3"}]
    assert "Added the note to D-002" in texts(note)
    files = await say(engine, conv, "add these to the Juniper deal", [PDF])
    assert "Added 1 file to D-002" in texts(files)
    assert fakes.all_spending() == []


async def test_outcome_with_a_missing_reason_asks_then_sends_plays_explicitly(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "We lost Larkfield Platform")
    assert "Why was D-003" in texts(ask) and actions_of(ask)
    assert fakes.deal.outcome_bodies == []
    confirm = await say(engine, conv, "the SSO objection never got resolved")
    text = texts(confirm)
    assert "LOST because" in text and "0 model calls" in text and "PLAY-03" in text
    assert fakes.deal.outcome_bodies == []  # still nothing written
    done = await click(engine, conv, action(confirm, "Yes")["id"])
    assert fakes.deal.outcome_bodies == [
        {"deal_id": 3, "result": "lost", "loss_reason": "unresolved_objection", "plays_used": ["PLAY-03"]}]
    assert "Outcome recorded" in cards(done)[0]["title"] and "PLAY-03" in all_text(done)
    assert fakes.deal.model_calls == 0
    offer = await click(engine, conv, action(done, "Brief the open deals again")["id"])
    assert "open deal" in texts(offer) and fakes.deal.model_calls == 0  # asks first, does not run


async def test_outcome_for_an_unread_deal_sends_an_empty_play_list(engine, fakes):
    conv = engine.new_conversation()
    confirm = await say(engine, conv, "We lost the Juniper deal because of price")
    assert "hasn't been read yet" in texts(confirm)
    await click(engine, conv, action(confirm, "Yes")["id"])
    assert fakes.deal.outcome_bodies == [{"deal_id": 2, "result": "lost", "loss_reason": "price", "plays_used": []}]


async def test_a_won_deal_needs_no_reason_and_a_closed_deal_is_not_overwritten(engine, fakes):
    conv = engine.new_conversation()
    confirm = await say(engine, conv, "We won the Cedarline deal")
    assert "WON" in texts(confirm)
    await click(engine, conv, action(confirm, "Yes")["id"])
    assert fakes.deal.outcome_bodies[0]["result"] == "won" and fakes.deal.outcome_bodies[0]["loss_reason"] is None
    again = await say(engine, conv, "We lost the Cedarline deal because of price")
    assert "already recorded as won" in texts(again) and len(fakes.deal.outcome_bodies) == 1


async def test_an_unclear_reason_is_asked_again_not_guessed(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "We lost the Juniper deal")
    unclear = await say(engine, conv, "no idea really")
    assert "didn't catch a reason" in texts(unclear)
    two = await say(engine, conv, "the competitor was cheaper and the price was lower")
    assert "could mean" in texts(two) and fakes.deal.outcome_bodies == []


async def test_the_workspace_changing_mid_flow_stops_before_spending(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Brief me on Cedarline")
    fakes.deal.workspace = {"id": "ws-other", "name": "Other Co", "kind": "company"}
    stopped = await click(engine, conv, action(ask, "Yes")["id"])
    assert "workspace changed" in texts(stopped) and "Other Co" in texts(stopped)
    assert fakes.all_spending() == []
