"""Found in the pilot's first deal (D-001, 2026-10-08): a declined file was swept into the next deal, the industry and segment
said in the message were dropped when the model read it, and a deal could not be read without also writing a brief."""

from __future__ import annotations

import pytest

from agent_hub.intents import detect
from agent_hub.planner import Planner

from .conftest import make_engine
from .fake_llm import FakeHubLLM, tool
from .helpers import RFP_DOC, action, actions_of, cards, click, say, texts

pytestmark = pytest.mark.anyio

NOTES = ("2026-03-04_discovery_call.md", b"Call notes: they need a CRM integration before peak season.")


def not_now(events: list[dict]) -> str:
    return next(a["id"] for a in actions_of(events) if a["label"] == "Not now")


# -- 1. a declined or abandoned file stops waiting ---------------------------------------------------------------------

async def test_a_file_declined_with_not_now_is_not_added_to_the_next_deal(engine, fakes):
    conv = engine.new_conversation()
    offer = await say(engine, conv, "", [RFP_DOC])  # read as an RFP to answer: the person meant something else
    assert "1 model call" in str(actions_of(offer))
    await click(engine, conv, not_now(offer))
    await say(engine, conv, "New deal SYN Kestrel rollout at Kestrel Logistics", [NOTES])
    assert fakes.deal.created_bodies[-1]["files"] == [NOTES[0]]
    assert fakes.all_spending() == []


async def test_a_file_left_behind_when_the_person_moves_on_is_not_added_either(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "", [RFP_DOC])  # offered, never answered
    await say(engine, conv, "New deal SYN Kestrel rollout at Kestrel Logistics", [NOTES])
    assert fakes.deal.created_bodies[-1]["files"] == [NOTES[0]]


async def test_files_sent_first_and_named_next_still_make_the_deal(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "", [NOTES])  # no offer was made for these, so they keep waiting
    out = await say(engine, conv, "New deal SYN Kestrel rollout at Kestrel Logistics")
    assert "Created D-007" in texts(out) and fakes.deal.created_bodies[-1]["files"] == [NOTES[0]]


# -- 2. details the model left out are kept --------------------------------------------------------------------------

@pytest.fixture
def with_model(fakes, tmp_path):
    made = []

    def build(*script):  # noqa: ANN002
        llm = FakeHubLLM(list(script))
        eng = make_engine(fakes, tmp_path / f"m{len(made)}", planner=Planner(lambda: llm))
        made.append(eng)
        return eng

    yield build
    for eng in made:
        eng.store.close()


async def test_industry_and_segment_said_in_the_message_survive_the_model_reading(with_model, fakes):
    eng = with_model(tool("new_deal", name="SYN Kestrel rollout", account="Kestrel Logistics"))
    conv = eng.new_conversation()
    out = await say(eng, conv, "New deal SYN Kestrel rollout at Kestrel Logistics, logistics mid-market", [NOTES])
    body = fakes.deal.created_bodies[-1]
    assert body["industry"] == "logistics" and body["segment"] == "mid_market"
    rows = dict(cards(out)[0]["sections"][0]["rows"])
    assert rows["Industry"] == "logistics" and rows["Segment"] == "mid market"


async def test_a_segment_in_the_model_s_own_words_is_normalised_or_left_out(with_model, fakes):
    eng = with_model(tool("new_deal", name="A", account="Acme", segment="Mid-Market"),
                     tool("new_deal", name="B", account="Beta", segment="galactic"))
    conv = eng.new_conversation()
    await say(eng, conv, "make a deal called A for Acme", [NOTES])
    assert fakes.deal.created_bodies[-1]["segment"] == "mid_market"
    await say(eng, conv, "and another, B for Beta", [NOTES])
    assert fakes.deal.created_bodies[-1]["segment"] is None  # refused by the Deal agent, so not sent


async def test_a_loss_reason_said_in_the_message_survives_the_model_reading(with_model, fakes):
    eng = with_model(tool("record_outcome", deal="Juniper", result="lost"))  # the model missed the reason
    conv = eng.new_conversation()
    out = await say(eng, conv, "Juniper was lost because of security and compliance")
    assert "Why was" not in texts(out) and "Ready to record" in texts(out)
    assert "security or compliance review failed" in texts(out)


async def test_clicking_a_loss_reason_button_records_it_without_asking_again(with_model, fakes):
    # the model reads the click as a fresh outcome with no reason: before the fix this asked "Why was ... lost?" forever
    eng = with_model(tool("record_outcome", deal="Juniper", result="lost"), tool("record_outcome", deal="Juniper", result="lost"))
    conv = eng.new_conversation()
    ask = await say(eng, conv, "Juniper was lost")
    assert "Why was" in texts(ask)
    out = await click(eng, conv, action(ask, "Security review")["id"])
    assert "Why was" not in texts(out) and "security or compliance review failed" in texts(out)
    typed = eng.new_conversation()
    await say(eng, typed, "Juniper was lost")
    out = await say(eng, typed, "security and compliance")  # typed, not clicked: a clear answer is read as one too
    assert "Why was" not in texts(out) and "security or compliance review failed" in texts(out)


# -- 3. read a deal without a brief ----------------------------------------------------------------------------------

async def test_a_new_deal_offers_a_read_and_the_read_writes_no_brief(engine, fakes):
    conv = engine.new_conversation()
    created = await say(engine, conv, "New deal SYN Kestrel rollout at Kestrel Logistics", [NOTES])
    assert [a["label"] for a in actions_of(created)] == ["Read D-007", "Brief me on D-007"]
    ask = await click(engine, conv, action(created, "Read D-007")["id"])
    assert "No brief is written" in texts(ask) and fakes.deal.model_calls == 0
    done = await click(engine, conv, action(ask, "Yes, read it")["id"])
    assert fakes.deal.model_calls == 1 and 7 not in fakes.deal.briefs
    assert "Here is what I read in D-007" in texts(done) and "This used 1 model call" in texts(done)
    card = cards(done)[0]
    assert card["title"] == "Read: D-007 SYN Kestrel rollout (Kestrel Logistics)"
    assert [s["heading"] for s in card["sections"]][:3] == ["Objections", "Competitors", "Plays used"]


async def test_reading_a_deal_that_was_already_read_spends_nothing(engine, fakes):
    conv = engine.new_conversation()
    created = await say(engine, conv, "New deal SYN Kestrel rollout at Kestrel Logistics", [NOTES])
    ask = await click(engine, conv, action(created, "Read D-007")["id"])
    await click(engine, conv, action(ask, "Yes, read it")["id"])
    again = await say(engine, conv, "read SYN Kestrel rollout")
    assert "already been read" in texts(again) and cards(again)[0]["title"].startswith("Read: D-007")
    assert fakes.deal.model_calls == 1  # only the first read


async def test_a_closed_deal_can_be_read_then_closed(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "New deal SYN Brightwater at Brightwater Credit Union", [NOTES])
    ask = await say(engine, conv, "read SYN Brightwater")
    await click(engine, conv, action(ask, "Yes, read it")["id"])
    out = await say(engine, conv, "SYN Brightwater was lost because of security and compliance")
    assert "Ready to record" in texts(out) and fakes.deal.model_calls == 1


def test_read_requests_are_recognised_and_rfps_are_not_mistaken_for_deals():
    for text, subject in (("read SYN Kestrel rollout", "SYN Kestrel rollout"), ("Read D-007", "D-007"),
                          ("please analyse the Juniper deal", "Juniper"), ("read it", None)):
        intent = detect(text, False)
        assert intent is not None and intent.kind == "deal_read", text
        assert intent.subject == subject, (text, intent.subject)
    rfp = detect("read this RFP", False)
    assert rfp is None or rfp.kind != "deal_read"
