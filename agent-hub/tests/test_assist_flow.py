"""Questions about a deal and follow-up drafts through the chat, against the fake Deal Intelligence app."""

from __future__ import annotations

import pytest

from agent_hub.intents import detect, mentioned_deals

from .fakes import BRIEF_CONTENT
from .helpers import action, actions_of, all_text, assistant, cards, click, say, texts

pytestmark = pytest.mark.anyio

DEALS = [
    {"id": 1, "code": "D-001", "name": "Cedarline Renewal", "account": "Cedarline Systems"},
    {"id": 2, "code": "D-002", "name": "Juniper Expansion", "account": "Juniper Health"},
    {"id": 3, "code": "D-003", "name": "Larkfield Platform", "account": "Larkfield Bank"},
    {"id": 4, "code": "D-004", "name": "Larkfield Security Add-on", "account": "Larkfield Bank"},
]


# -- reading the message ----------------------------------------------------------------------------

@pytest.mark.parametrize("text, kind", [
    ("Draft a follow-up for Cedarline", "email"),
    ("write an email to Cedarline", "email"),
    ("Can you draft a call agenda for D-002?", "call_agenda"),
    ("follow up with Juniper", "email"),
])
def test_followup_requests_are_recognised(text, kind):
    intent = detect(text, False)
    assert intent.kind == "deal_followup" and intent.slots["kind"] == kind and intent.subject


def test_followup_without_a_deal_uses_the_conversation():
    intent = detect("Draft a follow-up email", False)
    assert intent.kind == "deal_followup" and intent.subject is None


def test_an_rfp_email_is_not_a_deal_followup():
    assert detect("write an email about the RFP", False).kind != "deal_followup"


def test_questions_are_recognised_but_briefs_and_outcomes_keep_priority():
    assert detect("What did Dana say about SSO on Cedarline?", False).kind == "question"
    assert detect("Is the Cedarline champion engaged?", False).kind == "question"
    assert detect("What's going on with Cedarline?", False).kind == "deal_brief"
    assert detect("Why did we lose Larkfield?", False).kind == "question"
    assert detect("We lost Larkfield because of price", False).kind == "outcome"


def test_mentioned_deals_finds_names_inside_a_sentence():
    pick = lambda text: [c.deal["code"] for c in mentioned_deals(text, DEALS)]  # noqa: E731
    assert pick("What did Dana say about SSO on Cedarline?") == ["D-001"]
    assert pick("What is the status of D-003?") == ["D-003"]
    assert pick("Tell me about the Larkfield deal") == ["D-003", "D-004"]
    assert pick("who is the champion?") == []
    assert pick("how is Junipr doing") == ["D-002"]


# -- questions ---------------------------------------------------------------------------------------

async def test_a_question_asks_first_then_answers_with_citations(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "What did Dana say about SSO on Cedarline?")
    assert "D-001 Cedarline Renewal" in texts(ask) and "1 model call" in texts(ask)
    assert fakes.deal.ask_bodies == [] and fakes.all_spending() == []  # nothing is spent until the person says yes

    done = await click(engine, conv, action(ask, "Yes, go ahead")["id"])
    assert fakes.deal.ask_bodies == [{"deal_id": 1, "question": "What did Dana say about SSO on Cedarline?"}]
    assert fakes.deal.model_calls == 1
    card = cards(done)[0]
    assert card["title"] == "Answer: D-001 Cedarline Renewal (Cedarline Systems)" and card["tone"] == "info"
    assert card["sections"][0]["text"].startswith("Dana said SSO") and card["sections"][0]["chips"] == ["INT-0001"]
    assert "Draft a follow-up" in [a["label"] for a in actions_of(done)]
    assert fakes.all_forbidden() == []


async def test_a_confirmation_cannot_be_used_twice(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Is the Cedarline champion engaged?")
    yes = action(ask, "Yes, go ahead")["id"]
    await click(engine, conv, yes)
    again = await click(engine, conv, yes)
    assert "already done" in texts(again) and fakes.deal.model_calls == 1


async def test_ask_freely_skips_later_confirmations_but_still_counts(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "What did Dana say about SSO on Cedarline?")
    await click(engine, conv, action(ask, "don't ask again")["id"])
    assert fakes.deal.model_calls == 1
    second = await say(engine, conv, "Who else is on the Cedarline thread?")
    assert fakes.deal.model_calls == 2 and cards(second)[0]["title"].startswith("Answer:")
    assert not actions_of([e for e in second if e["kind"] == "text"]) or "Yes" not in all_text(second)


async def test_a_new_chat_asks_again(engine, fakes):
    first = engine.new_conversation()
    ask = await say(engine, first, "What did Dana say about SSO on Cedarline?")
    await click(engine, first, action(ask, "don't ask again")["id"])
    other = engine.new_conversation()
    ask = await say(engine, other, "What did Dana say about SSO on Cedarline?")
    assert "1 model call" in texts(ask) and fakes.deal.model_calls == 1


async def test_a_follow_up_question_uses_the_deal_the_chat_is_about(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Brief me on the Cedarline deal")
    await click(engine, conv, action(ask, "Yes")["id"])
    fakes.deal.calls.clear()
    nxt = await say(engine, conv, "Why is it stalled?")
    assert "D-001 Cedarline Renewal" in texts(nxt) and "1 model call" in texts(nxt)


async def test_an_ambiguous_deal_is_asked_about(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "What did they say about security at Larkfield?")
    assert "Did you mean" in texts(ask) and fakes.deal.ask_bodies == []
    done = await click(engine, conv, [a for a in actions_of(ask) if "D-004" in a["label"]][0]["id"])
    assert "1 model call" in texts(done)


async def test_a_deal_without_notes_is_not_asked_about(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "What is the status of Brightwater?")
    assert "no emails or notes" in texts(out) and fakes.deal.ask_bodies == []


async def test_an_answer_the_record_cannot_give_is_not_dressed_up(engine, fakes):
    fakes.deal.answer = {"found": False, "answer": "The notes do not mention a budget.", "sources": [], "grounded": True}
    conv = engine.new_conversation()
    ask = await say(engine, conv, "What is the Cedarline budget?")
    done = await click(engine, conv, action(ask, "Yes, go ahead")["id"])
    assert "not guessing" in all_text(done) and cards(done)[0]["tone"] == "info"


async def test_an_ungrounded_answer_is_flagged(engine, fakes):
    fakes.deal.answer = {"found": True, "answer": "They love it.", "sources": [], "grounded": False}
    conv = engine.new_conversation()
    ask = await say(engine, conv, "How do they feel about Cedarline?")
    done = await click(engine, conv, action(ask, "Yes, go ahead")["id"])
    assert cards(done)[0]["tone"] == "warn" and "Unverified" in all_text(done)


async def test_a_question_that_is_not_about_a_deal_falls_back_to_help(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "what is the capital of France?")
    assert "not sure which agent" in texts(out) and fakes.deal.ask_bodies == []
    hi = await say(engine, conv, "how are you?")
    assert fakes.deal.model_calls == 0 and assistant(hi)


async def test_a_question_when_the_deal_app_is_down_is_a_friendly_message(engine, fakes):
    fakes.deal.down = True
    conv = engine.new_conversation()
    out = await say(engine, conv, "What did Dana say about SSO on Cedarline?")
    assert "isn't running" in texts(out) or "isn't running" in all_text(out)


# -- follow-up drafts --------------------------------------------------------------------------------

async def test_followup_needs_a_brief_first(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "Draft a follow-up for Cedarline")
    assert "doesn't have a brief yet" in texts(out)
    assert action(out, "Brief me on D-001") and fakes.deal.followup_bodies == []


async def test_followup_drafts_the_next_step_after_a_yes(engine, fakes):
    fakes.deal.briefs[1] = {"id": 7, "deal_id": 1, "mode": "hindsight", "status": "ready", "error": None, "content": BRIEF_CONTENT}
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Draft a follow-up for Cedarline")
    assert "Executive sponsor" in texts(ask) and "1 model call" in texts(ask) and fakes.deal.followup_bodies == []
    done = await click(engine, conv, action(ask, "Yes, go ahead")["id"])
    assert fakes.deal.followup_bodies == [{"deal_id": 1, "kind": "email", "play_code": "PLAY-03"}]
    card = cards(done)[0]
    assert card["title"] == "Email draft: D-001 Cedarline Renewal (Cedarline Systems)"
    assert card["sections"][0]["heading"] == "Next steps on the data-residency clause"
    assert card["sections"][0]["chips"] == ["INT-0001"] and "[bracketed]" in all_text(done)
    assert "Nothing is sent" in texts(done) and fakes.deal.model_calls == 1
    assert "Make it a call agenda" in [a["label"] for a in actions_of(done)]


async def test_call_agenda_kind_is_sent(engine, fakes):
    fakes.deal.briefs[1] = {"id": 7, "deal_id": 1, "mode": "hindsight", "status": "ready", "error": None, "content": BRIEF_CONTENT}
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Draft a call agenda for D-001")
    done = await click(engine, conv, action(ask, "Yes, go ahead")["id"])
    assert fakes.deal.followup_bodies[0]["kind"] == "call_agenda" and cards(done)[0]["title"].startswith("Call agenda draft")


async def test_no_followup_for_a_closed_deal(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "Draft a follow-up for Oakhurst")
    assert "already closed" in texts(out) and fakes.deal.followup_bodies == []


async def test_followup_cancel_spends_nothing(engine, fakes):
    fakes.deal.briefs[1] = {"id": 7, "deal_id": 1, "mode": "hindsight", "status": "ready", "error": None, "content": BRIEF_CONTENT}
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Draft a follow-up for Cedarline")
    out = await click(engine, conv, action(ask, "Not now")["id"])
    assert "haven't done anything" in texts(out) and fakes.deal.model_calls == 0
