"""Administrators run the hub and are not a customer: they belong to the built-in "Hub administrators" team, which can never
hold company data, and their chat reaches no company's data. Enforced in the backend, not only on the admin page."""

from __future__ import annotations

import pytest

from agent_hub.auth import OPERATOR_ID, Auth, AuthError
from agent_hub.clients import UpstreamError

from .conftest import make_engine
from .helpers import action, cards, say, texts
from .test_admin_api import PASSWORD, hub, staffed  # noqa: F401  (fixtures)


@pytest.fixture
def accounts(tmp_path, fakes):  # noqa: ANN001, ANN201
    eng = make_engine(fakes, tmp_path / "op-data")
    eng.auth = Auth(eng.store)
    eng.auth.create_company("Accenture", {"deals": ["ws-demo"], "rfp": ["ws-acme"]})
    yield eng
    eng.store.close()


# -- the rules, in the account code ----------------------------------------------------------------------------------

def test_an_administrator_always_joins_the_hubs_own_team_whatever_company_was_given(accounts):
    a = accounts.auth
    uid = a.create_user("boss@x.com", PASSWORD, "Accenture", "Boss", "admin")
    user = a.user_by_id(uid)
    assert user.company_id == OPERATOR_ID and user.is_operator and user.workspaces == {"deals": (), "rfp": ()}


def test_a_member_cannot_be_put_in_the_administrators_team(accounts):
    with pytest.raises(AuthError) as exc:
        accounts.auth.create_user("x@x.com", PASSWORD, "Hub administrators")
    assert exc.value.code == "operator_company"
    accounts.auth.create_user("m@x.com", PASSWORD, "Accenture")
    with pytest.raises(AuthError):
        accounts.auth.update_user("m@x.com", company="Hub administrators")


def test_the_administrators_team_can_never_hold_data_be_renamed_or_deleted(accounts):
    a = accounts.auth
    for attempt in (lambda: a.update_company(OPERATOR_ID, workspaces={"deals": ["ws-demo"]}),
                    lambda: a.set_company_workspaces(OPERATOR_ID, {"rfp": ["ws-acme"]}),
                    lambda: a.update_company(OPERATOR_ID, name="Acme"), lambda: a.delete_company(OPERATOR_ID)):
        with pytest.raises(AuthError) as exc:
            attempt()
        assert exc.value.code == "operator_company"
    for name in ("Hub administrators", "hub ADMINISTRATORS"):
        with pytest.raises(AuthError) as exc:
            a.create_company(name)
        assert exc.value.code == "reserved_name"
    with pytest.raises(AuthError):
        a.update_company("Accenture", name="Hub administrators")


def test_admins_left_inside_a_customer_company_by_an_older_hub_are_moved_out(tmp_path, fakes):
    eng = make_engine(fakes, tmp_path / "old-data")
    a = Auth(eng.store)
    a.create_company("Accenture", {"deals": ["ws-demo"]})
    a.create_user("dana@acc.com", PASSWORD, "Accenture")
    a.create_user("boss@acc.com", PASSWORD, "Accenture", role="member")
    eng.store.sql_exec("UPDATE users SET role = 'admin' WHERE email = 'boss@acc.com'")  # what an older hub allowed
    again = Auth(eng.store)
    rows = {r["email"]: r["company_id"] for r in eng.store.sql_all("SELECT email, company_id FROM users")}
    assert rows == {"dana@acc.com": "accenture", "boss@acc.com": OPERATOR_ID}
    assert "operator.move" in [e["action"] for e in again.list_audit(10)]
    eng.store.close()


# -- the admin API ---------------------------------------------------------------------------------------------------

def test_the_api_refuses_to_give_the_administrators_data_however_it_is_asked(staffed, fakes):  # noqa: F811
    assert staffed.patch("/api/admin/companies/hub-administrators", json={"deals": ["ws-demo"]}).status_code == 409
    assert staffed.patch("/api/admin/companies/hub-administrators", json={"name": "Mine"}).status_code == 409
    assert staffed.post("/api/admin/companies/hub-administrators/provision").status_code == 404
    assert staffed.post("/api/admin/companies/hub-administrators/delete", json={}).status_code == 404
    assert staffed.post("/api/admin/companies", json={"name": "Hub administrators"}).status_code == 409
    assert fakes.deal.called("POST", "^/v1/workspaces") == [] and "Hub administrators" not in [
        c["name"] for c in staffed.get("/api/admin/overview").json()["companies"]]


def test_making_someone_an_administrator_moves_them_out_of_their_company(staffed):  # noqa: F811
    assert staffed.patch("/api/admin/users/dana@acc.com", json={"role": "admin"}).status_code == 200
    users = {u["email"]: u for u in staffed.get("/api/admin/overview").json()["users"]}
    assert users["dana@acc.com"]["company"] == "Hub administrators" and users["dana@acc.com"]["operator"]


# -- the chat --------------------------------------------------------------------------------------------------------

@pytest.mark.anyio
async def test_an_administrators_chat_reaches_no_company_data_and_never_uses_a_model(accounts, fakes):
    uid = accounts.auth.create_user("boss@x.com", PASSWORD, "Accenture", "Boss", "admin")
    conv = accounts.new_conversation(accounts.auth.user_by_id(uid))
    calls = len(fakes.deal.calls) + len(fakes.rfp.calls)
    for text in ("list the deals", "Brief me on the Cedarline deal", "here is our fact sheet", "answer this RFP"):
        out = await say(accounts, conv, text)
        assert "hub administrator" in texts(out)
    out = await say(accounts, conv, "past proposal we won", [("bid.docx", b"bytes")])
    assert "don't keep files for administrators" in texts(out)
    assert len(fakes.deal.calls) + len(fakes.rfp.calls) == calls and fakes.all_spending() == []
    assert accounts.store.sql_all("SELECT * FROM model_usage") == []


@pytest.mark.anyio
async def test_an_administrator_can_still_ask_how_the_hub_is_used(accounts):
    uid = accounts.auth.create_user("boss@x.com", PASSWORD, "Accenture", "Boss", "admin")
    conv = accounts.new_conversation(accounts.auth.user_by_id(uid))
    out = await say(accounts, conv, "hello")
    assert action(out, "Show usage")
    out = await say(accounts, conv, "show usage")
    assert cards(out)[0]["title"] == "Usage across companies" and "1 company" in texts(out)


@pytest.mark.anyio
async def test_the_refusal_is_in_the_backend_not_only_in_the_chat(accounts):
    uid = accounts.auth.create_user("boss@x.com", PASSWORD, "Accenture", "Boss", "admin")
    conv = accounts.new_conversation(accounts.auth.user_by_id(uid))
    st = accounts.load(conv)
    assert st["tenant"]["operator"] and st["tenant"]["allowed"] == {"deals": [], "rfp": []}
    with accounts.scope(st):  # even a flow that ignored the chat's rule could not reach a company's data
        with pytest.raises(UpstreamError) as exc:
            await accounts.deal.list_deals()
    assert exc.value.code == "no_workspace"
