"""What each company has used (people, chats, model calls and tokens, storage), for the administrator: the report, the admin
API, the chat answer (admin only, no model), and the hub's own ledger of the calls it makes to read a message."""

from __future__ import annotations

import pytest

from agent_hub import usage
from agent_hub.auth import Auth
from agent_hub.intents import is_usage_question
from agent_hub.llm import LLMError
from agent_hub.planner import Planner

from .conftest import make_engine
from .fake_llm import FakeHubLLM, tool
from .helpers import all_text, cards, say, texts

pytestmark = pytest.mark.anyio

PASSWORD = "correct horse battery"


@pytest.fixture
def world(fakes, tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    llm = FakeHubLLM(tool("list_deals", status="all"))
    eng = make_engine(fakes, tmp_path / "usage-data", planner=Planner(lambda: llm))
    eng.auth = Auth(eng.store)
    a = eng.auth
    a.create_company("Accenture", {"deals": ["ws-demo"], "rfp": ["ws-acme"]})
    a.create_company("Globex", {"deals": ["ws-other"], "rfp": ["ws-globex"]})
    a.create_user("boss@acc.com", PASSWORD, "Accenture", "Boss", "admin")
    a.create_user("dana@acc.com", PASSWORD, "Accenture")
    a.create_user("gus@globex.com", PASSWORD, "Globex")
    fakes.deal.usage_data = {"ws-demo": {"calls": 10, "failed": 1, "input_tokens": 5000, "output_tokens": 800, "bytes": 2048, "since": "2026-10-06T10:00:00+00:00"},
                             "ws-other": {"calls": 2, "input_tokens": 300, "output_tokens": 40, "bytes": 1024},
                             "ws-empty": {"calls": 1, "input_tokens": 70, "output_tokens": 9, "bytes": 512}}
    fakes.rfp.usage_data = {"ws-acme": {"calls": 6, "input_tokens": 2000, "output_tokens": 500, "bytes": 4096}}

    def chat(email):
        user = a.user_by_id(a.store.sql_one("SELECT id FROM users WHERE email = ?", (email,))["id"])
        return eng.new_conversation(user)

    eng.chat, eng.llm = chat, llm
    yield eng
    eng.store.close()


def company(report, name):  # noqa: ANN001, ANN201
    return next(c for c in report["companies"] if c["name"] == name)


# -- the report ----------------------------------------------------------------------------------------------------

async def test_a_companys_numbers_are_the_sum_of_its_workspaces_and_its_chats(world):
    boss = world.chat("boss@acc.com")
    await say(world, boss, "hi")
    await say(world, world.chat("dana@acc.com"), "hello")
    world.store.record_usage("accenture", None, "understand", "m", 100, 20, True)
    world.store.record_usage("accenture", None, "understand", "m", 200, 30, False)
    report = await usage.build_report(world)
    acc, globex = company(report, "Accenture"), company(report, "Globex")
    assert (acc["people"], acc["chats"], acc["messages"]) == (1, 1, 1) and (globex["people"], globex["chats"], globex["messages"]) == (1, 0, 0)
    assert report["operator"]["people"] == 1 and "Hub administrators" not in [c["name"] for c in report["companies"]]
    # Dana's message was read by the (fake) model, 100 in / 20 out, on top of the two recorded by hand; the administrator's
    # chat never uses a model and is not Accenture's
    assert acc["hub"] == {"calls": 3, "failed": 1, "input_tokens": 400, "output_tokens": 70}
    assert (acc["calls"], acc["failed"], acc["input_tokens"], acc["output_tokens"]) == (3 + 10 + 6, 1 + 1, 400 + 5000 + 2000, 70 + 800 + 500)
    assert acc["storage_bytes"] == 2048 + 4096 and acc["agents"]["deals"]["calls"] == 10 and acc["agents"]["rfp"]["calls"] == 6
    assert (globex["calls"], globex["storage_bytes"]) == (2, 1024)


async def test_workspaces_no_company_has_are_reported_apart_and_the_totals_add_up(world):
    report = await usage.build_report(world)
    spare = report["unassigned"]
    assert [(w["agent"], w["id"]) for w in spare["workspaces"]] == [("Deal Intelligence", "ws-empty")]
    assert (spare["calls"], spare["storage_bytes"]) == (1, 512)
    t = report["totals"]
    assert t["companies"] == 2 and t["people"] == 2 and t["calls"] == 10 + 2 + 1 + 6  # customers only; the administrator is the control plane
    assert t["input_tokens"] == 5000 + 300 + 70 + 2000 and t["storage_bytes"] >= 2048 + 1024 + 512 + 4096  # plus the hub's own files
    assert report["tracking_since"] == "2026-10-06T10:00:00+00:00" and report["agents_down"] == []


async def test_an_agent_that_is_down_is_named_and_its_numbers_are_not_invented(world, fakes):
    fakes.rfp.down = True
    report = await usage.build_report(world)
    assert report["agents_down"] == ["RFP Memory Assistant"]
    acc = company(report, "Accenture")
    assert acc["agents"]["rfp"]["up"] is False and acc["agents"]["rfp"]["calls"] == 0 and acc["calls"] == 10


async def test_looking_across_companies_does_not_leave_the_callers_limits_off(world):
    from agent_hub.clients import TENANT, WORKSPACES

    TENANT.set({"deals": ["ws-demo"], "rfp": []})
    WORKSPACES.set({"deals": "ws-demo"})
    await usage.build_report(world)
    assert TENANT.get() == {"deals": ["ws-demo"], "rfp": []} and WORKSPACES.get() == {"deals": "ws-demo"}


# -- asking in the chat ---------------------------------------------------------------------------------------------

async def test_an_administrator_asking_in_the_chat_gets_the_numbers_and_no_model_is_used(world, fakes):
    out = await say(world, world.chat("boss@acc.com"), "how many comapnies are currently working in this application?")
    assert "2 companies and 2 people" in texts(out)
    card = cards(out)[0]
    assert card["title"] == "Usage across companies" and {s.get("heading") for s in card["sections"]} >= {"Everyone", "Accenture", "Globex"}
    assert world.llm.calls == 0 and fakes.all_spending() == [] and "no model was used" in card["subtitle"]


async def test_a_member_is_told_this_is_for_administrators_and_sees_no_numbers(world):
    out = await say(world, world.chat("dana@acc.com"), "how many tokens did each company use")
    assert "only shown to administrators" in texts(out) and not cards(out) and "Globex" not in all_text(out) and world.llm.calls == 0


async def test_with_sign_in_off_the_person_at_this_computer_can_ask(fakes, tmp_path):
    eng = make_engine(fakes, tmp_path / "open-data")
    out = await say(eng, eng.new_conversation(), "show usage")
    assert cards(out) and cards(out)[0]["title"] == "Usage across companies"
    eng.store.close()


@pytest.mark.parametrize("text", [
    "how many comapnies are currently working in this application?", "how many companies are working with us", "how many tokens did each company use",
    "what is the storage used by every company", "show usage", "usage", "how many model calls have we made across all companies",
    "how much storage is the platform using", "how many companies have we onboarded", "how many users are registered on the hub", "how many compnies use our app"])
def test_usage_questions_are_recognised(text):
    assert is_usage_question(text)


@pytest.mark.parametrize("text", [
    "how many companies did we lose to price?", "What have we said about data storage before?", "how many deals are with companies in fintech",
    "Brief me on the Cedarline deal", "Which companies use SSO in our deals?", "Answer this security questionnaire", "how many open deals do I have",
    "draft a follow-up for Cedarline", "what is our encryption storage policy?", "tell me about common ground with the company"])
def test_questions_about_deals_and_rfps_are_not_taken_for_usage_questions(text):
    assert not is_usage_question(text)


# -- the hub's own ledger ---------------------------------------------------------------------------------------------

async def test_each_understanding_call_is_recorded_against_the_company_that_made_it(world):
    await say(world, world.chat("dana@acc.com"), "yo show me everything we got in the pipeline")
    await say(world, world.chat("gus@globex.com"), "what is in the pipeline right now")
    rows = world.store.sql_all("SELECT company_id, purpose, model, input_tokens, output_tokens, ok FROM model_usage ORDER BY company_id")
    assert [(r["company_id"], r["purpose"], r["model"], r["input_tokens"], r["output_tokens"], r["ok"]) for r in rows] == [
        ("accenture", "understand", "fake-hub-model", 100, 20, 1), ("globex", "understand", "fake-hub-model", 100, 20, 1)]


async def test_a_failed_understanding_call_is_counted_as_failed(fakes, tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    eng = make_engine(fakes, tmp_path / "fail-data", planner=Planner(lambda: FakeHubLLM(LLMError("quota", "The model quota is used up."))))
    eng.auth = Auth(eng.store)
    eng.auth.create_company("Accenture", {"deals": ["ws-demo"], "rfp": ["ws-acme"]})
    eng.auth.create_user("dana@acc.com", PASSWORD, "Accenture")
    user = eng.auth.user_by_id(eng.store.sql_one("SELECT id FROM users")["id"])
    await say(eng, eng.new_conversation(user), "show me the pipeline please")
    assert [(r["company_id"], r["ok"], r["input_tokens"]) for r in eng.store.sql_all("SELECT * FROM model_usage")] == [("accenture", 0, 0)]
    eng.store.close()


async def test_a_message_the_hub_reads_without_a_model_records_nothing(world):
    await say(world, world.chat("boss@acc.com"), "show usage")
    assert world.store.sql_all("SELECT * FROM model_usage") == []
