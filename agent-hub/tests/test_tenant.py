"""A company's chats only ever reach that company's workspaces, whatever the person types or the flows ask for."""

from __future__ import annotations

import pytest

from agent_hub.auth import Auth
from agent_hub.clients import TENANT, UpstreamError, WORKSPACES

from .conftest import make_engine
from .helpers import action, actions_of, all_text, cards, click, say, texts

pytestmark = pytest.mark.anyio

PASSWORD = "correct horse battery"


@pytest.fixture
def world(fakes, tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    eng = make_engine(fakes, tmp_path / "tenant-data", ask_workspace=True)
    eng.auth = Auth(eng.store)
    a = eng.auth
    a.create_company("Accenture", {"deals": ["ws-demo"], "rfp": ["ws-acme"]})
    a.create_company("Globex", {"deals": ["ws-other"], "rfp": ["ws-globex"]})
    a.create_company("Multi", {"deals": ["ws-demo", "ws-other"], "rfp": ["ws-acme"]})
    a.create_company("NoRfp", {"deals": ["ws-demo"]})
    for email, company in (("dana@acc.com", "Accenture"), ("gus@globex.com", "Globex"), ("mia@multi.com", "Multi"),
                           ("ned@norfp.com", "NoRfp")):
        a.create_user(email, PASSWORD, company)

    def chat(email):
        user = a.user_by_id(a.store.sql_one("SELECT id FROM users WHERE email = ?", (email,))["id"])
        return eng.new_conversation(user)

    eng.chat = chat
    yield eng
    eng.store.close()


def deal_workspaces_used(fakes) -> set[str | None]:
    return {h for p, h in fakes.deal.workspace_headers if p.startswith("/v1/deals") or p.startswith("/v1/portfolio")}


async def test_a_company_with_one_workspace_per_agent_is_placed_in_it(world, fakes):
    conv = world.chat("dana@acc.com")
    assert world.load(conv)["workspaces"] == {"deals": "ws-demo", "rfp": "ws-acme"}
    out = await say(world, conv, "list the deals")
    assert "workspaces in" not in texts(out) and "6 deals in Halcyon Demo" in texts(out)
    assert deal_workspaces_used(fakes) == {"ws-demo"}


async def test_another_company_gets_its_own_data_and_never_the_first_ones(world, fakes):
    conv = world.chat("gus@globex.com")
    out = await say(world, conv, "list the deals")
    assert "2 deals in Brightwater Team" in texts(out) and "Zephyr Pilot" in all_text(out) and "Oakhurst" not in all_text(out)
    assert deal_workspaces_used(fakes) == {"ws-other"}


async def test_the_workspace_list_shows_only_the_companys_own(world, fakes):
    conv = world.chat("dana@acc.com")
    out = await say(world, conv, "show workspaces")
    seen = all_text(out)
    assert "Halcyon Demo" in seen and "Acme Security" in seen
    for hidden in ("Brightwater Team", "My deals", "Globex Bids"):
        assert hidden not in seen


async def test_asking_for_another_companys_workspace_finds_nothing(world, fakes):
    conv = world.chat("dana@acc.com")
    for text in ("use the Brightwater workspace", "use workspace ws-other", "use the Globex workspace for rfps"):
        out = await say(world, conv, text)
        assert "couldn't find a workspace" in texts(out)
    assert world.load(conv)["workspaces"] == {"deals": "ws-demo", "rfp": "ws-acme"}
    assert "ws-other" not in {h for _p, h in fakes.deal.workspace_headers}


async def test_a_deal_that_only_exists_in_another_companys_workspace_is_not_offered(world, fakes):
    conv = world.chat("dana@acc.com")
    out = await say(world, conv, "Brief me on Zephyr")  # Zephyr lives in Globex's workspace
    assert "Brightwater Team" not in all_text(out) and "Zephyr Pilot" not in all_text(out) and "couldn't find a deal" in texts(out)


async def test_a_tampered_chat_state_is_corrected_before_anything_is_sent(world, fakes):
    conv = world.chat("dana@acc.com")
    st = world.store.get_state(conv)
    st["workspaces"] = {"deals": "ws-other", "rfp": "ws-globex"}  # as if someone edited the database
    world.store.set_state(conv, st)
    out = await say(world, conv, "list the deals")
    assert "6 deals in Halcyon Demo" in texts(out) and deal_workspaces_used(fakes) == {"ws-demo"}


async def test_the_client_itself_refuses_a_workspace_that_is_not_the_companys(world, fakes):
    t, w = TENANT.set({"deals": ["ws-demo"], "rfp": []}), WORKSPACES.set({"deals": "ws-other"})
    try:
        with pytest.raises(UpstreamError) as exc:
            await world.deal.list_deals()
        assert exc.value.code == "forbidden_workspace"
        WORKSPACES.set({})
        with pytest.raises(UpstreamError) as none:
            await world.deal.list_deals()  # no workspace named at all is not "the agent's open one" for a company
        assert none.value.code == "no_workspace"
    finally:
        WORKSPACES.reset(w)
        TENANT.reset(t)
    assert [c for c in fakes.deal.calls if c[1].startswith("/v1/deals")] == []


async def test_a_company_with_no_workspace_for_an_agent_gets_a_clear_message_and_no_call(world, fakes):
    conv = world.chat("ned@norfp.com")
    out = await say(world, conv, "Answer this questionnaire", [("Acme Security Questionnaire.docx", b"x")])
    assert "workspace is set up for your company" in all_text(out)
    assert [c for c in fakes.rfp.calls if c[1] != "/health" and c[1] != "/v1/workspaces"] == []


async def test_a_company_with_two_workspaces_chooses_between_those_two_only(world, fakes):
    conv = world.chat("mia@multi.com")
    assert world.load(conv)["workspaces"] == {"rfp": "ws-acme"}  # one RFP workspace is placed, deals needs a choice
    out = await say(world, conv, "list the deals")
    labels = [a["label"] for a in actions_of(out)]
    assert len(labels) == 2 and not any("My deals" in label for label in labels)
    done = await click(world, conv, action(out, "Brightwater Team")["id"])
    assert "2 deals in Brightwater Team" in texts(done)


async def test_a_chat_with_no_owner_while_sign_in_is_on_reaches_nothing(world, fakes):
    conv = world.store.create_conversation()  # the kind of chat made before sign-in existed
    out = await say(world, conv, "list the deals")
    assert [c for c in fakes.deal.calls if c[1].startswith("/v1/deals")] == [] and "workspace is set up" in all_text(out)


async def test_saving_default_workspaces_is_not_a_thing_for_a_company(world, fakes):
    conv = world.chat("dana@acc.com")
    out = await say(world, conv, "remember these workspaces as Mine")
    assert "set by your administrator" in texts(out) and world.store.get_setting("company") is None
    assert "set by your administrator" in texts(await say(world, conv, "forget the saved workspaces"))


async def test_a_new_chat_for_a_company_ignores_the_machines_saved_default(world, fakes):
    world.store.set_setting("company", {"name": "Someone", "workspaces": {"deals": "ws-other"}, "workspace_names": {}})
    conv = world.chat("dana@acc.com")
    assert world.load(conv)["workspaces"]["deals"] == "ws-demo"


async def test_changing_a_companys_workspaces_reaches_chats_already_open(world, fakes):
    conv = world.chat("dana@acc.com")
    await say(world, conv, "list the deals")
    world.auth.set_company_workspaces("Accenture", {"deals": ["ws-other"]})
    out = await say(world, conv, "list the deals")
    assert "2 deals in Brightwater Team" in texts(out)


async def test_the_company_name_is_shown_in_the_state(world, fakes):
    from agent_hub.main import public_state

    conv = world.chat("dana@acc.com")
    assert public_state(world.load(conv))["company"] == "Accenture"


async def test_without_sign_in_nothing_changes(fakes, tmp_path):
    eng = make_engine(fakes, tmp_path / "open")  # no auth
    conv = eng.new_conversation()
    out = await say(eng, conv, "show workspaces")
    assert "Brightwater Team" in all_text(out) and eng.load(conv).get("tenant") is None
    eng.store.close()


async def test_the_hub_presents_each_agents_service_token_when_one_is_set(fakes, tmp_path, monkeypatch):
    monkeypatch.setenv("HUB_DEAL_SERVICE_TOKEN", "deal-token")
    eng = make_engine(fakes, tmp_path / "svc")
    conv = eng.new_conversation()
    await say(eng, conv, "list the deals")
    assert set(fakes.deal.service_tokens) == {"deal-token"} and set(fakes.rfp.service_tokens) <= {None}
    monkeypatch.delenv("HUB_DEAL_SERVICE_TOKEN")
    fakes.deal.service_tokens.clear()
    await say(eng, conv, "list the deals")
    assert set(fakes.deal.service_tokens) == {None}
    eng.store.close()


async def test_a_client_account_is_never_sent_to_an_agents_own_screen_and_an_administrator_gets_no_company_data(world, fakes):
    admin = world.auth.create_user("boss@acc.com", PASSWORD, "Accenture", "Boss", "admin")
    boss_user = world.auth.user_by_id(admin)
    assert boss_user.is_operator and boss_user.company_name == "Hub administrators"  # an administrator never joins a customer company
    member = world.chat("dana@acc.com")
    out = await say(world, member, "list the deals")
    assert not [c for c in cards(out) if c.get("links")] and "6 deals" in texts(out)
    assert any(c["title"] == "Your deals" for c in cards(out))
    before = len(fakes.deal.calls)
    out = await say(world, world.new_conversation(boss_user), "list the deals")
    assert "hub administrator" in texts(out) and "6 deals" not in texts(out) and len(fakes.deal.calls) == before


async def test_without_sign_in_the_links_stay(fakes, tmp_path):
    eng = make_engine(fakes, tmp_path / "links-open")
    conv = eng.new_conversation()
    assert any(c.get("links") for c in cards(await say(eng, conv, "list the deals")))
    eng.store.close()
