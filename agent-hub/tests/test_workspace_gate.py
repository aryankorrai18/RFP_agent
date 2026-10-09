"""A chat that has not chosen a workspace is asked, once, instead of silently working in the one an agent has open."""

from __future__ import annotations

import pytest

from .conftest import make_engine
from .helpers import RFP_DOC, action, actions_of, cards, click, say, texts

pytestmark = pytest.mark.anyio


@pytest.fixture
def asking(fakes, tmp_path):
    eng = make_engine(fakes, tmp_path / "ask-data", ask_workspace=True)
    yield eng
    eng.store.close()


def labels(events) -> list[str]:
    return [a["label"] for a in actions_of(events)]


async def test_a_deal_request_asks_which_workspace_before_doing_anything(asking, fakes):
    conv = asking.new_conversation()
    out = await say(asking, conv, "Brief me on Cedarline")
    assert "3 workspaces in Deal Intelligence" in texts(out) and "Which one should this chat use for deals" in texts(out)
    assert labels(out) == ["Halcyon Demo (6 deals, open in the app)", "Brightwater Team (2 deals)", "My deals (0 deals)"]
    assert fakes.all_spending() == [] and fakes.deal.portfolio_bodies == []


async def test_picking_one_carries_on_with_the_original_request(asking, fakes):
    conv = asking.new_conversation()
    out = await say(asking, conv, "Brief me on Cedarline")
    done = await click(asking, conv, action(out, "Brightwater Team")["id"])
    assert "Using Brightwater Team for Deal Intelligence." in texts(done)
    assert "D-002 Cedarline Pilot" in texts(done) and "1 model call" in texts(done)  # Brightwater's Cedarline, not Halcyon's
    assert asking.load(conv)["workspaces"] == {"deals": "ws-other"}
    assert fakes.all_forbidden() == []


async def test_it_asks_once_and_then_remembers_the_choice(asking, fakes):
    conv = asking.new_conversation()
    out = await say(asking, conv, "list the open deals")
    await click(asking, conv, action(out, "Halcyon Demo")["id"])
    again = await say(asking, conv, "how many deals do we have?")
    assert "workspaces in" not in texts(again) and "6 deals in Halcyon Demo" in texts(again)


async def test_ignoring_the_question_does_not_ask_again(asking, fakes):
    conv = asking.new_conversation()
    await say(asking, conv, "list the open deals")
    again = await say(asking, conv, "list the open deals")
    assert "workspaces in" not in texts(again) and "Halcyon Demo" in texts(again)  # the agent's open workspace, named
    assert asking.load(conv)["workspace_names"]["deals"] == "Halcyon Demo"


async def test_a_saved_company_default_means_no_question(asking, fakes):
    first = asking.new_conversation()
    await say(asking, first, "use the Brightwater workspace")
    await say(asking, first, "remember these workspaces as Accenture")
    fresh = asking.new_conversation()
    out = await say(asking, fresh, "Brief me on Zephyr")
    assert "workspaces in" not in texts(out) and "D-001 Zephyr Pilot" in texts(out)


async def test_company_questions_ask_but_off_topic_ones_do_not(asking, fakes):
    conv = asking.new_conversation()
    chat = await say(asking, conv, "what is the capital of France?")
    assert "workspaces in" not in texts(chat) and "not sure which agent" in texts(chat)
    ask = await say(asking, conv, "Why do we lose deals?")
    assert "workspaces in Deal Intelligence" in texts(ask)
    done = await click(asking, conv, action(ask, "Halcyon Demo")["id"])
    assert "your deals (Halcyon Demo)" in texts(done) and "1 model call" in texts(done)


async def test_an_rfp_asks_about_the_rfp_workspace(asking, fakes):
    conv = asking.new_conversation()
    out = await say(asking, conv, "Answer this questionnaire", [RFP_DOC])
    assert "2 workspaces in RFP Memory Assistant" in texts(out) and "Acme Security" in " ".join(labels(out))


async def test_commands_and_replies_are_never_gated(asking, fakes):
    conv = asking.new_conversation()
    for text in ("show workspaces", "hello", "yes", "no"):
        out = await say(asking, conv, text)
        assert "Which one should this chat use" not in texts(out)


async def test_a_chat_that_already_chose_is_not_asked(asking, fakes):
    conv = asking.new_conversation()
    await say(asking, conv, "use the Brightwater workspace")
    out = await say(asking, conv, "Brief me on Zephyr")
    assert "workspaces in" not in texts(out) and "D-001 Zephyr Pilot" in texts(out)
