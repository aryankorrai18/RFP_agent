"""A chat chooses its own workspace in each agent; the hub sends it with every request and never switches the
workspace the agents' own screens show."""

from __future__ import annotations

import pytest

from agent_hub.intents import detect

from .helpers import RFP_DOC, action, actions_of, all_text, cards, click, say, texts

pytestmark = pytest.mark.anyio


def deal_headers(fakes, path_part: str = "/v1/") -> set[str | None]:
    return {h for p, h in fakes.deal.workspace_headers if path_part in p and p != "/v1/workspaces"}


# -- reading the message ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["show workspaces", "list my workspaces", "which workspace am I in?", "workspaces"])
def test_listing_requests(text):
    assert detect(text, False).kind == "workspace_list"


@pytest.mark.parametrize("text, subject, agent", [
    ("use the Halcyon local demo workspace", "Halcyon local demo", None),
    ("switch to workspace Brightwater", "Brightwater", None),
    ("use the Globex workspace for rfps", "Globex", "rfp"),
    ("use the default workspace", "default", None),
])
def test_choosing_requests(text, subject, agent):
    intent = detect(text, False)
    assert intent.kind == "workspace_use" and intent.subject == subject and intent.slots["agent"] == agent


def test_ordinary_use_and_open_requests_are_not_workspace_requests():
    assert detect("open the Cedarline deal", False).kind != "workspace_use"
    assert detect("use the Cedarline notes", False).kind not in ("workspace_use", "workspace_list")


# -- choosing --------------------------------------------------------------------------------------

async def test_show_lists_each_agents_workspaces_and_costs_nothing(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "show workspaces")
    card = cards(out)[0]
    text = all_text(out)
    assert "Halcyon Demo" in text and "Brightwater Team" in text and "Acme Security" in text and "Globex Bids" in text
    assert "this chat" in text and "open in the app" not in text.split("Brightwater")[0]
    assert "Use Brightwater Team for deals" in [a["label"] for a in actions_of(out)]
    assert card["title"] == "Workspaces" and fakes.deal.model_calls == 0 and fakes.all_forbidden() == []


async def test_choose_by_name_sets_only_this_chat(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "use the Brightwater workspace")
    assert "Brightwater Team" in texts(out) and "Nothing was switched" in texts(out) and "2 deals (2 open)" in texts(out)
    assert engine.load(conv)["workspaces"] == {"deals": "ws-other"}
    assert fakes.deal.workspace["id"] == "ws-demo" and fakes.all_forbidden() == []  # the agent's own workspace is untouched


async def test_a_chat_then_works_in_the_chosen_workspace(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "use the Brightwater workspace")
    fakes.deal.workspace_headers.clear()
    ask = await say(engine, conv, "Brief me on Zephyr")
    assert "D-001 Zephyr Pilot" in texts(ask) and "Brightwater Team" in texts(ask)
    done = await click(engine, conv, action(ask, "Yes")["id"])
    assert cards(done)[0]["title"].startswith("Brief: D-001 Zephyr Pilot")
    assert deal_headers(fakes) == {"ws-other"}  # every request, including the job polls
    assert list(fakes.deal.stores["ws-other"]["briefs"]) == [1] and fakes.deal.stores["ws-demo"]["briefs"] == {}
    assert fakes.all_forbidden() == []


async def test_two_chats_work_in_two_workspaces_at_once(engine, fakes):
    mine, theirs = engine.new_conversation(), engine.new_conversation()
    await say(engine, theirs, "use the Brightwater workspace")
    a = await say(engine, mine, "Brief me on Cedarline")
    b = await say(engine, theirs, "Brief me on Cedarline")
    assert "D-001 Cedarline Renewal" in texts(a) and "D-002 Cedarline Pilot" in texts(b)


async def test_going_back_to_the_default(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "use the Brightwater workspace")
    out = await say(engine, conv, "use the default workspace")
    assert engine.load(conv)["workspaces"] == {} and "own open workspace" in texts(out)
    again = await say(engine, conv, "Brief me on Cedarline")
    assert "D-001 Cedarline Renewal" in texts(again)


async def test_a_buttons_choice_and_an_unknown_name(engine, fakes):
    conv = engine.new_conversation()
    listed = await say(engine, conv, "show workspaces")
    out = await click(engine, conv, action(listed, "Use Globex Bids for RFPs")["id"])
    assert engine.load(conv)["workspaces"] == {"rfp": "ws-globex"} and "Globex Bids" in texts(out)
    missing = await say(engine, conv, "use the Nowhere workspace")
    assert "couldn't find a workspace" in texts(missing) and "Brightwater Team" in texts(missing)


async def test_choosing_forgets_what_the_chat_was_about(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Brief me on Cedarline")
    await click(engine, conv, action(ask, "Yes")["id"])
    assert engine.load(conv)["deal_id"] == 1
    await say(engine, conv, "use the Brightwater workspace")
    st = engine.load(conv)
    assert st["deal_id"] is None and st["flow"] is None  # deal 1 means something else over there


async def test_the_rfp_agent_gets_the_chosen_workspace(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "use the Globex workspace for rfps")
    fakes.rfp.workspace_headers.clear()
    out = await say(engine, conv, "Answer this questionnaire", [RFP_DOC])
    assert "Globex Bids" in all_text(out)
    assert {h for p, h in fakes.rfp.workspace_headers if p == "/v1/workspace"} == {"ws-globex"}
    assert fakes.deal.workspace_headers == [] or all(h is None for _p, h in fakes.deal.workspace_headers)


async def test_a_workspace_that_disappears_is_explained(engine, fakes):
    conv = engine.new_conversation()
    st = engine.load(conv)
    st["workspaces"] = {"deals": "ws-gone"}
    engine.save(conv, st)
    out = await say(engine, conv, "Brief me on Cedarline")
    assert "ws-gone" in all_text(out) and fakes.deal.model_calls == 0


# -- finding a deal in another workspace --------------------------------------------------------------

async def test_a_deal_in_another_workspace_is_offered_not_missed(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "Brief me on Zephyr")
    text = texts(out)
    assert "D-001 Zephyr Pilot is in Brightwater Team" in text and "Halcyon Demo" in text
    assert fakes.deal.model_calls == 0

    switched = await click(engine, conv, action(out, "Use Brightwater Team")["id"])
    assert "Brightwater Team" in texts(switched)
    again = await click(engine, conv, action(switched, "Ask it again")["id"])
    assert "D-001 Zephyr Pilot" in texts(again) and "1 model call" in texts(again) or "2 model calls" in texts(again)


async def test_a_question_about_a_deal_in_another_workspace_is_offered_too(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "What did the buyer say about Zephyr?")
    assert "Zephyr Pilot is in Brightwater Team" in texts(out) and fakes.deal.ask_bodies == []


async def test_a_deal_that_is_nowhere_still_gets_the_normal_answer(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "Brief me on Nonexistent Holdings")
    assert "couldn't find a deal matching" in texts(out)


async def test_the_page_can_see_which_workspaces_a_chat_uses(engine, fakes):
    from agent_hub.main import public_state

    conv = engine.new_conversation()
    await say(engine, conv, "use the Brightwater workspace")
    assert public_state(engine.load(conv))["workspaces"] == {"deals": "Brightwater Team"}
