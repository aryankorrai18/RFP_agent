"""Questions about the whole company's records, and a saved default workspace per company."""

from __future__ import annotations

import pytest

from agent_hub.intents import company_targets, detect

from .helpers import action, actions_of, all_text, cards, click, say, texts

pytestmark = pytest.mark.anyio


# -- reading the message ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, targets", [
    ("Why do we lose fintech deals?", ["deals"]),
    ("How many deals did we lose to price?", ["deals"]),
    ("What is the win rate across our pipeline?", ["deals"]),
    ("What have we told clients about SOC 2 before?", ["rfp"]),
    ("Do we have ISO 27001?", ["rfp"]),
    ("Which deals are stalled and what is our security policy?", ["deals", "rfp"]),
    ("What is going wrong across everything we have?", ["deals", "rfp"]),
    ("What is the capital of France?", []),
    ("What did Dana say about SSO on Cedarline?", []),
])
def test_which_agents_hold_the_answer(text, targets):
    assert company_targets(text) == targets


@pytest.mark.parametrize("text, kind, subject", [
    ("remember these workspaces as Accenture", "workspace_save", "Accenture"),
    ("make these workspaces the default", "workspace_save", None),
    ("forget the saved workspaces", "workspace_forget", None),
])
def test_saving_and_forgetting_requests(text, kind, subject):
    intent = detect(text, False)
    assert intent.kind == kind and intent.subject == subject


# -- asking the whole company ---------------------------------------------------------------------------

async def test_a_pipeline_question_goes_to_the_deals_after_a_yes(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Why do we lose deals?")
    assert "your deals (Halcyon Demo)" in texts(ask) and "1 model call" in texts(ask)
    assert fakes.all_spending() == [] and fakes.deal.portfolio_bodies == []

    done = await click(engine, conv, action(ask, "Yes, go ahead")["id"])
    assert fakes.deal.portfolio_bodies == [{"question": "Why do we lose deals?"}] and fakes.rfp.library_bodies == []
    card = cards(done)[0]
    assert card["title"] == "From your deals" and card["sections"][0]["chips"] == ["D-006"]
    stats = card["sections"][-1]
    assert stats["heading"].startswith("The counts it used") and ["Deals", "6 (0 won, 1 lost, 5 open)"] in stats["rows"]
    assert "This used 1 model call" in texts(done) and fakes.all_forbidden() == []


async def test_a_library_question_goes_to_the_rfp_assistant(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "What have we told clients about SOC 2 before?")
    assert "your RFP library and company facts (Acme Security)" in texts(ask) and "1 model call" in texts(ask)
    done = await click(engine, conv, action(ask, "Yes, go ahead")["id"])
    card = cards(done)[0]
    assert card["title"] == "From your RFP library and company facts" and card["sections"][0]["chips"] == ["FACT-003", "ANS-0012"]
    assert "FACT-003: Certifications" in all_text(done) and fakes.deal.portfolio_bodies == []


async def test_a_question_that_needs_both_asks_both_and_says_so(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "What is going wrong across everything we have?")
    assert "2 model calls" in texts(ask) and "one per agent" in texts(ask)
    done = await click(engine, conv, action(ask, "Yes, go ahead")["id"])
    assert [c["title"] for c in cards(done)[:2]] == ["From your deals", "From your RFP library and company facts"]
    assert "This used 2 model calls" in texts(done) and fakes.deal.model_calls + fakes.rfp.model_calls == 2


async def test_a_named_deal_beats_the_company_wide_reading(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Why did we lose Oakhurst?")
    assert "D-006 Oakhurst Renewal" in texts(ask) and fakes.deal.portfolio_bodies == []


async def test_the_deal_the_chat_is_about_does_not_swallow_a_company_question(engine, fakes):
    conv = engine.new_conversation()
    brief = await say(engine, conv, "Brief me on Cedarline")
    await click(engine, conv, action(brief, "Yes")["id"])
    ask = await say(engine, conv, "Why do we lose deals?")
    assert "your deals" in texts(ask) and "D-001" not in texts(ask)


async def test_ask_freely_covers_company_questions_too(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Why do we lose deals?")
    await click(engine, conv, action(ask, "don't ask again")["id"])
    again = await say(engine, conv, "Why do we lose fintech deals?")
    assert cards(again)[0]["title"] == "From your deals" and fakes.deal.model_calls == 2


async def test_an_unanswerable_or_ungrounded_answer_is_said_plainly(engine, fakes):
    fakes.deal.portfolio_answer = {"found": False, "answer": "The deals do not say.", "sources": [], "grounded": True}
    fakes.rfp.library_answer = {"found": True, "answer": "Probably yes.", "sources": [], "evidence": [], "grounded": False}
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Which deals are stalled and what is our security policy?")
    done = await click(engine, conv, action(ask, "Yes, go ahead")["id"])
    text = all_text(done)
    assert "not guessing" in text and "Unverified" in text and cards(done)[1]["tone"] == "warn"


async def test_an_empty_library_is_explained_and_the_other_answer_still_arrives(engine, fakes):
    fakes.rfp.library_empty = True
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Which deals are stalled and what is our security policy?")
    done = await click(engine, conv, action(ask, "Yes, go ahead")["id"])
    assert "no company facts and no approved answers" in all_text(done) and cards(done)[0]["title"] == "From your deals"


async def test_the_chats_workspace_is_the_one_that_is_searched(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "use the Brightwater workspace")
    fakes.deal.workspace_headers.clear()
    ask = await say(engine, conv, "Why do we lose deals?")
    done = await click(engine, conv, action(ask, "Yes, go ahead")["id"])
    assert ["Deals", "2 (0 won, 0 lost, 2 open)"] in cards(done)[0]["sections"][-1]["rows"]
    assert ("/v1/portfolio/ask", "ws-other") in fakes.deal.workspace_headers


async def test_a_question_that_is_not_about_the_company_is_unchanged(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "what is the capital of France?")
    assert "not sure which agent" in texts(out) and fakes.all_spending() == []


# -- a saved default per company ------------------------------------------------------------------------

async def test_remember_saves_the_workspaces_for_every_new_chat(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "use the Brightwater workspace")
    out = await say(engine, conv, "remember these workspaces as Accenture")
    assert "Saved as Accenture" in texts(out) and engine.load(conv)["company_name"] == "Accenture"
    fresh = engine.new_conversation()
    st = engine.load(fresh)
    assert st["workspaces"] == {"deals": "ws-other"} and st["company_name"] == "Accenture"
    brief = await say(engine, fresh, "Brief me on Zephyr")
    assert "D-001 Zephyr Pilot" in texts(brief)  # the saved workspace, without saying "use ..." again


async def test_remember_needs_a_choice_first(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "remember these workspaces")
    assert "Choose the workspaces first" in texts(out) and engine.store.get_setting("company") is None


async def test_forget_goes_back_to_each_agents_own_workspace(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "use the Brightwater workspace")
    await say(engine, conv, "remember these workspaces")
    out = await say(engine, conv, "forget the saved workspaces")
    assert "Forgotten" in texts(out) and engine.store.get_setting("company") is None
    assert engine.load(engine.new_conversation()).get("workspaces") in (None, {})


async def test_the_page_is_told_the_company_name(engine, fakes):
    from agent_hub.main import public_state

    conv = engine.new_conversation()
    await say(engine, conv, "use the Brightwater workspace")
    await say(engine, conv, "remember these workspaces as Accenture")
    assert public_state(engine.load(conv))["company"] == "Accenture"
    assert actions_of([]) == []


async def test_a_partial_answer_with_evidence_is_not_called_a_guess(engine, fakes):
    fakes.rfp.library_answer = {"found": False, "answer": "No past answer, but the fact sheet says annual testing.",
                                "sources": ["FACT-009"], "evidence": [{"id": "FACT-009", "label": "Testing"}], "grounded": True,
                                "degraded": True, "warning": "Library search unavailable (bank missing)"}
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Do we have a pen test policy?")
    text = all_text(await click(engine, conv, action(ask, "Yes, go ahead")["id"]))
    assert "not guessing" not in text and "FACT-009: Testing" in text and "bank missing" in text


# -- counts are free, and an empty workspace is not charged for ---------------------------------------------

@pytest.mark.parametrize("text, headline", [
    ("how many deals do we have right?", "6 deals in Halcyon Demo"),
    ("How many open deals do we have?", "5 open deals in Halcyon Demo"),
    ("how many deals did we lose to price?", "1 deal lost to price (the price)"),
    ("how many lost deals are there", "1 lost deal in Halcyon Demo"),
])
async def test_counting_deals_is_free_and_exact(engine, fakes, text, headline):
    conv = engine.new_conversation()
    out = await say(engine, conv, text)
    assert headline.split(" (")[0] in texts(out) and "no model calls" in texts(out)
    assert fakes.all_spending() == [] and fakes.rfp.model_calls == 0 and fakes.deal.model_calls == 0
    assert not any(a["label"].startswith("Yes") for a in actions_of(out))


async def test_a_count_with_a_condition_it_cannot_test_goes_to_the_model_after_a_yes(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "how many deals are stalled?")
    assert "1 model call" in texts(ask) and fakes.deal.model_calls == 0


async def test_how_many_deals_do_we_have_is_not_mistaken_for_a_library_question(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "how many deals do we have right?")
    assert fakes.rfp.library_bodies == [] and "RFP library" not in all_text(out)


async def test_an_empty_deals_workspace_points_at_one_with_deals_instead_of_failing(engine, fakes):
    conv = engine.new_conversation()
    st = engine.load(conv)
    st["workspaces"] = {"deals": "ws-empty"}
    engine.save(conv, st)
    out = await say(engine, conv, "Why do we lose deals?")
    assert "no deals in" in texts(out) and "Halcyon Demo has 6" in texts(out) and fakes.deal.model_calls == 0
    assert "Use Halcyon Demo" in [a["label"] for a in actions_of(out)] and fakes.all_spending() == []
    count = await say(engine, conv, "how many deals do we have?")
    assert "no deals in My deals" in texts(count)
    switched = await click(engine, conv, action(count, "Use Halcyon Demo")["id"])
    assert "Halcyon Demo" in texts(switched)


async def test_with_an_empty_deals_workspace_the_library_part_is_still_asked(engine, fakes):
    conv = engine.new_conversation()
    st = engine.load(conv)
    st["workspaces"] = {"deals": "ws-empty"}
    engine.save(conv, st)
    out = await say(engine, conv, "Which deals are stalled and what is our security policy?")
    assert "no deals in" in texts(out) and "your RFP library" in texts(out) and "1 model call" in texts(out)


# -- listing deals is free too ------------------------------------------------------------------------------

@pytest.mark.parametrize("text, kind", [
    ("what are those deals?", ""), ("list the open deals", "open"), ("show me all our lost deals", "lost"),
    ("which deals do we have?", ""), ("do you have the deals of halycon software?", ""),
])
def test_listing_requests_are_recognised(text, kind):
    intent = detect(text, False)
    assert intent.kind == "deal_list" and intent.slots["filter"] == kind


async def test_listing_deals_shows_each_one_for_free(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "what are those deals?")
    bullets = cards(out)[0]["sections"][0]["bullets"]
    assert len(bullets) == 6 and "D-001 Cedarline Renewal (Cedarline Systems): open, proposal" in bullets
    assert "D-006 Oakhurst Renewal (Oakhurst): lost (price)" in bullets
    assert "6 deals in Halcyon Demo" in texts(out) and "no model calls" in texts(out)
    assert fakes.all_spending() == [] and fakes.deal.model_calls == 0


async def test_listing_can_be_filtered(engine, fakes):
    conv = engine.new_conversation()
    assert len(cards(await say(engine, conv, "list the open deals"))[0]["sections"][0]["bullets"]) == 5
    lost = await say(engine, conv, "show me the lost deals")
    assert [b.split(":")[0] for b in cards(lost)[0]["sections"][0]["bullets"]] == ["D-006 Oakhurst Renewal (Oakhurst)"]


async def test_naming_the_teams_own_workspace_is_not_a_condition(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "do you have the deals of halcyon demo?")
    assert "6 deals in Halcyon Demo" in texts(out) and fakes.deal.portfolio_bodies == [] and fakes.deal.model_calls == 0


async def test_listing_in_an_empty_workspace_points_elsewhere(engine, fakes):
    conv = engine.new_conversation()
    st = engine.load(conv)
    st["workspaces"] = {"deals": "ws-empty"}
    engine.save(conv, st)
    out = await say(engine, conv, "list the deals")
    assert "no deals in My deals" in texts(out) and "Halcyon Demo has 6" in texts(out)


async def test_a_typo_of_the_workspace_name_still_lists_for_free(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "do you have the deals of halcyn demo?")
    assert "6 deals in Halcyon Demo" in texts(out) and fakes.deal.model_calls == 0


async def test_a_listing_with_a_real_condition_still_goes_to_the_model_after_asking(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "list the deals with an overdue security promise")
    assert "1 model call" in texts(ask) or "Did you mean" in texts(ask) or "not sure" in texts(ask)
    assert fakes.deal.model_calls == 0


async def test_asking_for_active_and_won_and_lost_lists_everything_for_free(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "what are all my deals currently the active one and the ones which we lost or won?")
    assert len(cards(out)[0]["sections"][0]["bullets"]) == 6 and "6 deals in Halcyon Demo" in texts(out)
    assert fakes.deal.model_calls == 0 and fakes.all_spending() == []
