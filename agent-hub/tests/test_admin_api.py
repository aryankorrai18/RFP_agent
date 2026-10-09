"""The admin page's API: who may use it, what it changes, and the guard rails (last administrator, first-run setup)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent_hub import deps, main
from agent_hub.auth import Auth

PASSWORD = "correct horse battery"


@pytest.fixture
def hub(engine, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    engine.auth = Auth(engine.store)
    main.app.state.engine = engine
    client = TestClient(main.app, base_url="http://hub.test")
    client.engine, client.accounts = engine, engine.auth
    with client:
        yield client
    main.app.state.engine = None


def browser() -> TestClient:
    return TestClient(main.app, base_url="http://hub.test")


def sign_in(client: TestClient, email: str, password: str = PASSWORD) -> None:
    assert client.post("/api/auth/login", json={"email": email, "password": password}).status_code == 200


@pytest.fixture
def staffed(hub):
    """A company, an administrator (signed in on `hub`) and an ordinary member (signed in on `member`)."""
    a = hub.accounts
    a.create_company("Accenture", {"deals": ["ws-demo"], "rfp": ["ws-acme"]})
    a.create_user("admin@acc.com", PASSWORD, "Accenture", "Ada", "admin")
    a.create_user("dana@acc.com", PASSWORD, "Accenture", "Dana")
    sign_in(hub, "admin@acc.com")
    hub.member = browser()
    sign_in(hub.member, "dana@acc.com")
    return hub


# -- first-run setup ---------------------------------------------------------------------------------------

SETUP = {"company": "Accenture", "deals": ["ws-demo"], "rfp": ["ws-acme"], "email": "boss@acc.com", "name": "Boss", "password": PASSWORD}


def test_setup_is_only_offered_on_this_machine_before_any_account(hub, monkeypatch):
    assert hub.get("/api/auth/me").json()["setup_available"] is False and hub.post("/api/admin/setup", json=SETUP).status_code == 403
    monkeypatch.setattr(deps, "is_loopback", lambda _r: True)
    assert hub.get("/api/auth/me").json()["setup_available"] is True


def test_setup_makes_the_first_administrator_in_the_hubs_own_team_and_signs_them_in(hub, monkeypatch):
    monkeypatch.setattr(deps, "is_loopback", lambda _r: True)
    response = hub.post("/api/admin/setup", json=SETUP)
    assert response.status_code == 200 and response.json()["user"]["role"] == "admin"
    assert "httponly" in response.headers["set-cookie"].lower()
    assert hub.get("/api/auth/me").json() == {"auth": "on", "setup_available": False, "user": {
        "email": "boss@acc.com", "name": "Boss", "company": "Hub administrators", "role": "admin", "operator": True}}
    assert hub.accounts.list_companies() == []  # setup makes no customer company: those are added on the admin page
    assert hub.get("/api/admin/overview").status_code == 200
    assert hub.post("/api/admin/setup", json={**SETUP, "email": "evil@x.com"}).status_code == 403  # never twice


def test_setup_checks_the_password_before_creating_anything(hub, monkeypatch):
    monkeypatch.setattr(deps, "is_loopback", lambda _r: True)
    assert hub.post("/api/admin/setup", json={**SETUP, "password": "short"}).status_code == 400
    assert hub.accounts.list_companies() == [] and hub.accounts.list_users() == []


def test_setup_is_refused_when_sign_in_was_switched_off_on_purpose(hub, monkeypatch):
    monkeypatch.setattr(deps, "is_loopback", lambda _r: True)
    monkeypatch.setenv("HUB_AUTH", "off")
    assert hub.post("/api/admin/setup", json=SETUP).status_code == 403


# -- who may use it ----------------------------------------------------------------------------------------

ROUTES = [("get", "/api/admin/overview"), ("get", "/api/admin/audit"), ("post", "/api/admin/companies"),
          ("patch", "/api/admin/companies/accenture"), ("delete", "/api/admin/companies/accenture"),
          ("post", "/api/admin/users"), ("patch", "/api/admin/users/dana@acc.com")]


@pytest.mark.parametrize("method, path", ROUTES)
def test_anonymous_callers_are_refused(staffed, method, path):
    assert getattr(browser(), method)(path, **({"json": {}} if method in ("post", "patch") else {})).status_code == 401


@pytest.mark.parametrize("method, path", ROUTES)
def test_an_ordinary_member_is_refused(staffed, method, path):
    assert getattr(staffed.member, method)(path, **({"json": {}} if method in ("post", "patch") else {})).status_code == 403


def test_a_member_cannot_make_themselves_an_administrator(staffed):
    assert staffed.member.patch("/api/admin/users/dana@acc.com", json={"role": "admin"}).status_code == 403
    assert [u["role"] for u in staffed.accounts.list_users() if u["email"] == "dana@acc.com"] == ["member"]


def test_the_admin_page_is_public_but_holds_no_data(staffed):
    page = browser().get("/admin")
    assert page.status_code == 200 and "dana@acc.com" not in page.text and "scrypt" not in page.text


def test_a_post_from_another_site_is_refused(staffed):
    assert staffed.post("/api/admin/companies", json={"name": "X"}, headers={"Origin": "http://evil.example"}).status_code == 403


# -- reading ---------------------------------------------------------------------------------------------------

def test_the_overview_lists_companies_people_and_what_the_agents_have(staffed):
    body = staffed.get("/api/admin/overview").json()
    assert body["me"]["role"] == "admin" and body["me"]["operator"] and body["companies"][0]["users"] == 1  # the admin is not an Accenture person
    assert [c["name"] for c in body["companies"]] == ["Accenture"] and body["operator"]["name"] == "Hub administrators"
    assert {u["email"] for u in body["users"]} == {"admin@acc.com", "dana@acc.com"}
    assert {w["id"] for w in body["agents"]["deals"]["workspaces"]} == {"ws-demo", "ws-other", "ws-empty"}
    assert body["agents"]["deals"]["up"] and {w["id"] for w in body["agents"]["rfp"]["workspaces"]} == {"ws-acme", "ws-globex"}
    text = staffed.get("/api/admin/overview").text
    assert "scrypt" not in text and "password" not in text.lower() and "token_hash" not in text and "hub_session" not in text


def test_the_overview_works_when_an_agent_is_down(staffed, fakes):
    fakes.rfp.down = True
    body = staffed.get("/api/admin/overview").json()
    assert body["agents"]["rfp"] == {"up": False, "name": "RFP Memory Assistant", "workspaces": []} and body["agents"]["deals"]["up"]


# -- companies -------------------------------------------------------------------------------------------------

def test_a_company_can_be_added_renamed_given_workspaces_and_removed(staffed):
    assert staffed.post("/api/admin/companies", json={"name": "Globex", "deals": ["ws-other"], "rfp": ["ws-globex"]}).status_code == 201
    assert staffed.patch("/api/admin/companies/globex", json={"name": "Globex Corp", "deals": ["ws-other", "ws-empty"]}).status_code == 200
    mine = next(c for c in staffed.get("/api/admin/overview").json()["companies"] if c["name"] == "Globex Corp")
    assert mine["workspaces"] == {"deals": ["ws-other", "ws-empty"], "rfp": ["ws-globex"]}
    assert staffed.delete("/api/admin/companies/globex").status_code == 200
    assert [c["name"] for c in staffed.get("/api/admin/overview").json()["companies"]] == ["Accenture"]


def test_company_mistakes_are_explained_and_change_nothing(staffed):
    assert staffed.post("/api/admin/companies", json={"name": "accenture"}).status_code == 409
    assert staffed.post("/api/admin/companies", json={"name": ""}).status_code == 400
    typo = staffed.post("/api/admin/companies", json={"name": "Initech", "deals": ["ws-typo"]})
    assert typo.status_code == 422 and "ws-typo" in typo.json()["detail"]
    assert staffed.delete("/api/admin/companies/accenture").status_code == 409  # it still has people
    assert staffed.patch("/api/admin/companies/missing", json={"name": "x"}).status_code == 404
    assert [c["name"] for c in staffed.accounts.list_companies()] == ["Accenture"]


def test_workspace_ids_are_accepted_as_typed_when_the_agent_is_down(staffed, fakes):
    fakes.deal.down = True
    assert staffed.post("/api/admin/companies", json={"name": "Initech", "deals": ["whatever-id"]}).status_code == 201


# -- people ----------------------------------------------------------------------------------------------------

def test_an_account_can_be_added_with_a_role_and_signs_in(staffed):
    staffed.post("/api/admin/companies", json={"name": "Globex", "deals": ["ws-other"]})
    response = staffed.post("/api/admin/users", json={"email": "Gus@Globex.com", "name": "Gus", "company": "Globex",
                                                      "role": "member", "password": PASSWORD})
    assert response.status_code == 201
    gus = browser()
    sign_in(gus, "gus@globex.com")
    assert gus.get("/api/auth/me").json()["user"] == {"email": "gus@globex.com", "name": "Gus", "company": "Globex", "role": "member", "operator": False}


@pytest.mark.parametrize("body, status", [
    ({"email": "x@y.com", "company": "Accenture", "password": "short"}, 400),
    ({"email": "not-an-email", "company": "Accenture", "password": PASSWORD}, 400),
    ({"email": "dana@acc.com", "company": "Accenture", "password": PASSWORD}, 409),
    ({"email": "x@y.com", "company": "Nobody", "password": PASSWORD}, 404),
    ({"email": "x@y.com", "company": "Accenture", "password": PASSWORD, "role": "owner"}, 400)])
def test_account_mistakes_are_refused(staffed, body, status):
    assert staffed.post("/api/admin/users", json=body).status_code == status


def test_changing_a_persons_name_company_and_role(staffed):
    staffed.post("/api/admin/companies", json={"name": "Globex", "deals": ["ws-other"]})
    assert staffed.patch("/api/admin/users/dana@acc.com", json={"name": "Dana C", "company": "Globex", "role": "admin"}).status_code == 200
    row = next(u for u in staffed.get("/api/admin/overview").json()["users"] if u["email"] == "dana@acc.com")
    assert (row["name"], row["company"], row["role"]) == ("Dana C", "Hub administrators", "admin")  # an administrator joins the hub's own team
    assert staffed.patch("/api/admin/users/dana@acc.com", json={"role": "member", "company": "Globex"}).status_code == 200
    row = next(u for u in staffed.get("/api/admin/overview").json()["users"] if u["email"] == "dana@acc.com")
    assert (row["company"], row["role"]) == ("Globex", "member")


def test_a_move_between_companies_ends_the_persons_open_sessions(staffed):
    staffed.post("/api/admin/companies", json={"name": "Globex", "deals": ["ws-other"]})
    assert staffed.member.get("/api/agents").status_code == 200
    staffed.patch("/api/admin/users/dana@acc.com", json={"company": "Globex"})
    assert staffed.member.get("/api/agents").status_code == 401


def test_disabling_ends_the_session_at_once_and_enabling_lets_them_back(staffed):
    assert staffed.patch("/api/admin/users/dana@acc.com", json={"disabled": True}).status_code == 200
    assert staffed.member.get("/api/agents").status_code == 401
    assert browser().post("/api/auth/login", json={"email": "dana@acc.com", "password": PASSWORD}).status_code == 401
    staffed.patch("/api/admin/users/dana@acc.com", json={"disabled": False})
    sign_in(browser(), "dana@acc.com")


def test_resetting_a_password_ends_sessions_and_a_weak_one_changes_nothing(staffed):
    assert staffed.patch("/api/admin/users/dana@acc.com", json={"password": "short", "name": "Changed"}).status_code == 400
    assert next(u for u in staffed.accounts.list_users() if u["email"] == "dana@acc.com")["name"] == "Dana"  # untouched
    assert staffed.member.get("/api/agents").status_code == 200
    assert staffed.patch("/api/admin/users/dana@acc.com", json={"password": "a brand new password"}).status_code == 200
    assert staffed.member.get("/api/agents").status_code == 401
    assert browser().post("/api/auth/login", json={"email": "dana@acc.com", "password": PASSWORD}).status_code == 401
    sign_in(browser(), "dana@acc.com", "a brand new password")


def test_the_only_administrator_cannot_be_demoted_or_disabled(staffed):
    for change in ({"role": "member"}, {"disabled": True}):
        refused = staffed.patch("/api/admin/users/admin@acc.com", json=change)
        assert refused.status_code == 409 and refused.json()["code"] == "last_admin"
    assert staffed.get("/api/admin/overview").status_code == 200  # still signed in as an administrator


def test_with_a_second_administrator_the_first_can_step_down(staffed):
    staffed.patch("/api/admin/users/dana@acc.com", json={"role": "admin"})
    assert staffed.patch("/api/admin/users/admin@acc.com", json={"role": "member"}).status_code == 400  # a member needs a customer company
    assert staffed.patch("/api/admin/users/admin@acc.com", json={"role": "member", "company": "Accenture"}).status_code == 200
    assert staffed.get("/api/admin/overview").status_code == 401  # a demotion ends their sessions at once
    sign_in(staffed, "admin@acc.com")
    assert staffed.get("/api/admin/overview").status_code == 403  # and they are no longer an administrator
    assert staffed.patch("/api/admin/users/dana@acc.com", json={"disabled": True}).status_code == 403


# -- the activity log --------------------------------------------------------------------------------------------

def test_every_change_is_logged_with_who_made_it_and_never_a_password(staffed):
    staffed.post("/api/admin/companies", json={"name": "Globex"})
    staffed.post("/api/admin/users", json={"email": "gus@globex.com", "company": "Globex", "password": PASSWORD})
    staffed.patch("/api/admin/users/gus@globex.com", json={"password": "a brand new password", "disabled": True})
    entries = [e for e in staffed.get("/api/admin/audit").json()["entries"] if not e["action"].startswith("auth.")]  # sign-ins are logged too
    assert [e["action"] for e in entries][:3] == ["user.update", "user.add", "company.add"]
    assert all(e["actor"] == "admin@acc.com" for e in entries) and "gus@globex.com: disabled, password" in entries[0]["detail"]
    blob = str(entries)
    assert PASSWORD not in blob and "a brand new password" not in blob and "scrypt" not in blob
    assert len(staffed.get("/api/admin/audit?limit=1").json()["entries"]) == 1


def test_setup_is_logged(hub, monkeypatch):
    monkeypatch.setattr(deps, "is_loopback", lambda _r: True)
    hub.post("/api/admin/setup", json=SETUP)
    assert [e["action"] for e in hub.get("/api/admin/audit").json()["entries"]][:2] == ["auth.signin", "setup"]  # set up, then signed in


def test_the_setup_form_can_list_the_workspaces_only_while_setup_is_open(hub, monkeypatch):
    assert hub.get("/api/admin/setup-info").status_code == 403
    monkeypatch.setattr(deps, "is_loopback", lambda _r: True)
    info = hub.get("/api/admin/setup-info").json()
    assert {w["id"] for w in info["agents"]["deals"]["workspaces"]} == {"ws-demo", "ws-other", "ws-empty"}
    hub.post("/api/admin/setup", json=SETUP)
    assert browser().get("/api/admin/setup-info").status_code == 403


# -- the security check on the specialists ---------------------------------------------------------------------

def test_the_overview_says_when_an_agent_can_be_read_without_the_hub(staffed):
    security = staffed.get("/api/admin/overview").json()["security"]
    assert security["deals"]["state"] == "open" and security["rfp"]["state"] == "open"
    assert security["deals"]["hub_token"] is False and security["deals"]["token_env"] == "HUB_DEAL_SERVICE_TOKEN"


def test_a_protected_agent_is_reported_protected_and_whether_the_hub_holds_its_token(staffed, fakes, monkeypatch):
    fakes.deal.require_token = "s3cret"
    monkeypatch.setenv("HUB_DEAL_SERVICE_TOKEN", "s3cret")
    security = staffed.get("/api/admin/overview").json()["security"]
    assert security["deals"]["state"] == "protected" and security["deals"]["hub_token"] is True and security["rfp"]["state"] == "open"
    monkeypatch.delenv("HUB_DEAL_SERVICE_TOKEN")
    assert staffed.get("/api/admin/overview").json()["security"]["deals"]["hub_token"] is False  # protected, and the hub cannot get in


def test_an_agent_that_is_down_is_reported_down_not_open(staffed, fakes):
    fakes.rfp.down = True
    assert staffed.get("/api/admin/overview").json()["security"]["rfp"]["state"] == "down"
