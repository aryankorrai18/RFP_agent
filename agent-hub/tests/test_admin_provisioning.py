"""Giving a new company its own workspaces from the admin page, and deleting a company with or without its data."""

from __future__ import annotations

from pathlib import Path

from agent_hub import main

from .test_admin_api import PASSWORD, hub, staffed  # noqa: F401  (fixtures)


def add_company(hub, name="Globex", **extra):  # noqa: ANN001, ANN201
    return hub.post("/api/admin/companies", json={"name": name, "deals": [], "rfp": [], **extra})


def company(hub, name):  # noqa: ANN001, ANN201
    return next(c for c in hub.get("/api/admin/overview").json()["companies"] if c["name"] == name)


def audit_actions(hub):  # noqa: ANN001, ANN201
    return [e["action"] for e in hub.get("/api/admin/audit").json()["entries"]]


def test_a_new_company_gets_no_workspaces_unless_the_admin_asks(staffed, fakes):  # noqa: F811
    assert add_company(staffed).status_code == 201
    assert company(staffed, "Globex")["workspaces"] == {"deals": [], "rfp": []}
    assert fakes.deal.called("POST", "^/v1/workspaces") == [] and fakes.rfp.called("POST", "^/v1/workspaces") == []


def test_the_admin_can_have_the_company_created_with_its_own_empty_workspace_in_each_agent(staffed, fakes):  # noqa: F811
    response = add_company(staffed, provision=True)
    assert response.status_code == 201
    created = response.json()["created"]
    assert set(created) == {"deals", "rfp"}
    assert company(staffed, "Globex")["workspaces"] == {"deals": [created["deals"]], "rfp": [created["rfp"]]}
    made = fakes.deal.extra + fakes.rfp.extra
    assert {w["name"] for w in made} == {"Globex - My deals", "Globex - RFPs"} and all(w["kind"] == "company" for w in made)
    assert fakes.deal.stores[created["deals"]]["deals"] == {}  # empty
    assert audit_actions(staffed).count("workspace.create") == 2 and "company.add" in audit_actions(staffed)


def test_creating_the_workspaces_does_not_leave_the_agents_own_screens_on_the_new_one(staffed, fakes):  # noqa: F811
    add_company(staffed, provision=True)
    assert fakes.deal.current_active() == "ws-demo" and fakes.rfp.current_active() == "ws-acme"  # put back after each create
    assert fakes.deal.called("POST", "/v1/workspaces/ws-demo/activate") and fakes.rfp.called("POST", "/v1/workspaces/ws-acme/activate")


def test_each_company_gets_a_different_workspace_even_with_the_same_name_stem(staffed, fakes):  # noqa: F811
    a = add_company(staffed, "Globex", provision=True).json()["created"]
    b = add_company(staffed, "Globex Ltd", provision=True).json()["created"]
    assert a["deals"] != b["deals"] and a["rfp"] != b["rfp"]
    assert not set(company(staffed, "Globex")["workspaces"]["deals"]) & set(company(staffed, "Globex Ltd")["workspaces"]["deals"])


def test_ticked_workspaces_are_kept_alongside_the_new_ones(staffed):  # noqa: F811
    created = add_company(staffed, provision=True, deals=["ws-empty"]).json()["created"]
    assert company(staffed, "Globex")["workspaces"]["deals"] == ["ws-empty", created["deals"]]


def test_if_an_agent_is_down_nothing_is_created_and_the_company_is_not_left_half_made(staffed, fakes):  # noqa: F811
    fakes.rfp.down = True
    response = add_company(staffed, provision=True)
    assert response.status_code == 502 and "Nothing was created" in response.json()["detail"]
    assert [c["name"] for c in staffed.get("/api/admin/overview").json()["companies"]] == ["Accenture"]
    assert fakes.deal.extra == [] and len(fakes.deal.removed) == 1  # the one it made before the failure was taken back out


def test_a_duplicate_company_name_creates_no_workspaces(staffed, fakes):  # noqa: F811
    response = add_company(staffed, "Accenture", provision=True)
    assert response.status_code == 409 and fakes.deal.extra == [] and fakes.rfp.extra == []


def test_only_an_administrator_can_provision_or_delete(staffed, fakes):  # noqa: F811
    assert add_company(staffed.member, provision=True).status_code == 403
    assert staffed.member.get("/api/admin/companies/accenture/deletion-plan").status_code == 403
    assert staffed.member.post("/api/admin/companies/accenture/delete", json={}).status_code == 403
    assert fakes.deal.extra == []


def test_the_plan_says_which_workspaces_could_be_deleted_and_why_others_cannot(staffed):  # noqa: F811
    staffed.accounts.create_company("Initech", {"deals": ["ws-demo", "ws-empty"], "rfp": []})
    created = add_company(staffed, provision=True, deals=["ws-demo", "ws-empty"]).json()["created"]  # shares two with Initech
    plan = {w["key"]: w for w in staffed.get("/api/admin/companies/globex/deletion-plan").json()["workspaces"]}
    assert plan[f"deals:{created['deals']}"]["deletable"] and plan[f"rfp:{created['rfp']}"]["deletable"]
    assert not plan["deals:ws-demo"]["deletable"] and "Initech" in plan["deals:ws-demo"]["reason"] and "currently has open" in plan["deals:ws-demo"]["reason"]
    assert not plan["deals:ws-empty"]["deletable"]  # shared with Initech, and it is the agent's original workspace


def test_deleting_a_company_keeps_its_data_by_default(staffed, fakes):  # noqa: F811
    created = add_company(staffed, provision=True).json()["created"]
    response = staffed.post("/api/admin/companies/globex/delete", json={})
    assert response.status_code == 200 and response.json()["deleted"] == [] and len(response.json()["kept"]) == 2
    assert [c["name"] for c in staffed.get("/api/admin/overview").json()["companies"]] == ["Accenture"]
    assert fakes.deal.called("DELETE", "^/v1/workspaces") == [] and fakes.rfp.called("DELETE", "^/v1/workspaces") == []
    assert created["deals"] in fakes.deal.workspace_ids() and "company.delete" in audit_actions(staffed)


def test_deleting_with_the_data_removes_exactly_the_confirmed_workspaces(staffed, fakes):  # noqa: F811
    created = add_company(staffed, provision=True).json()["created"]
    confirm = [f"deals:{created['deals']}", f"rfp:{created['rfp']}"]
    response = staffed.post("/api/admin/companies/globex/delete", json={"delete_data": True, "confirm": confirm, "password": PASSWORD})
    assert response.status_code == 200 and sorted(response.json()["deleted"]) == sorted(confirm)
    assert created["deals"] not in fakes.deal.workspace_ids() and created["rfp"] not in fakes.rfp.workspace_ids()
    assert audit_actions(staffed).count("workspace.delete") == 2


def test_a_stale_or_missing_confirmation_deletes_nothing(staffed, fakes):  # noqa: F811
    created = add_company(staffed, provision=True).json()["created"]
    for confirm in ([], [f"deals:{created['deals']}"], [f"deals:{created['deals']}", f"rfp:{created['rfp']}", "deals:ws-other"]):
        r = staffed.post("/api/admin/companies/globex/delete", json={"delete_data": True, "confirm": confirm, "password": PASSWORD})
        assert r.status_code == 409 and r.json()["code"] == "confirm_mismatch"
    assert company(staffed, "Globex") and fakes.deal.removed == [] and fakes.rfp.removed == []


def test_a_company_that_still_has_people_cannot_be_deleted_and_nothing_is_touched(staffed, fakes):  # noqa: F811
    created = add_company(staffed, provision=True).json()["created"]
    staffed.accounts.create_user("sam@globex.com", PASSWORD, "Globex")
    confirm = [f"deals:{created['deals']}", f"rfp:{created['rfp']}"]
    r = staffed.post("/api/admin/companies/globex/delete", json={"delete_data": True, "confirm": confirm})
    assert r.status_code == 409 and r.json()["code"] == "company_in_use"
    assert company(staffed, "Globex") and fakes.deal.removed == [] and fakes.rfp.removed == []


def test_a_workspace_another_company_uses_or_that_is_open_in_an_agent_is_kept_and_reported(staffed, fakes):  # noqa: F811
    created = add_company(staffed, provision=True, deals=["ws-demo"]).json()["created"]  # ws-demo is the Deal agent's open workspace
    staffed.accounts.create_company("Initech", {"deals": [created["deals"]], "rfp": []})  # shares the new Deal workspace
    plan = staffed.get("/api/admin/companies/globex/deletion-plan").json()["workspaces"]
    confirm = sorted(w["key"] for w in plan if w["deletable"])
    assert confirm == [f"rfp:{created['rfp']}"]
    r = staffed.post("/api/admin/companies/globex/delete", json={"delete_data": True, "confirm": confirm, "password": PASSWORD}).json()
    assert r["deleted"] == [f"rfp:{created['rfp']}"]
    reasons = {k["key"]: k["reason"] for k in r["kept"]}
    assert "currently has open" in reasons["deals:ws-demo"] and "also uses it" in reasons[f"deals:{created['deals']}"]
    assert created["deals"] in fakes.deal.workspace_ids() and "ws-demo" in fakes.deal.workspace_ids()


def test_the_usage_report_is_admin_only_and_lists_every_company(staffed, fakes):  # noqa: F811
    fakes.deal.usage_data = {"ws-demo": {"calls": 3, "input_tokens": 900, "output_tokens": 100, "bytes": 4096}}
    add_company(staffed, "Globex")
    body = staffed.get("/api/admin/usage").json()
    assert [c["name"] for c in body["companies"]] == ["Accenture", "Globex"] and body["totals"]["companies"] == 2
    assert next(c for c in body["companies"] if c["name"] == "Accenture")["agents"]["deals"]["calls"] == 3
    assert staffed.member.get("/api/admin/usage").status_code == 403


def test_the_chat_never_reaches_the_workspace_creating_code():
    src = Path(main.__file__).parent
    for name in ("engine.py", "intents.py", "planner.py", "llm.py", "clients.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "provisioning" not in text, name


# -- an existing company that has no workspaces yet ----------------------------------------------------------

def test_an_existing_company_can_be_given_its_own_workspaces_in_one_click(staffed, fakes):  # noqa: F811
    add_company(staffed)  # no workspaces
    response = staffed.post("/api/admin/companies/globex/provision")
    assert response.status_code == 200 and set(response.json()["created"]) == {"deals", "rfp"}
    created = response.json()["created"]
    assert company(staffed, "Globex")["workspaces"] == {"deals": [created["deals"]], "rfp": [created["rfp"]]}
    assert fakes.deal.current_active() == "ws-demo" and fakes.rfp.current_active() == "ws-acme"
    assert audit_actions(staffed).count("workspace.create") == 2


def test_only_the_missing_agent_gets_a_workspace_and_existing_ones_are_kept(staffed, fakes):  # noqa: F811
    add_company(staffed, deals=["ws-empty"])
    response = staffed.post("/api/admin/companies/globex/provision")
    assert set(response.json()["created"]) == {"rfp"} and fakes.deal.extra == []
    assert company(staffed, "Globex")["workspaces"]["deals"] == ["ws-empty"]


def test_a_company_that_already_has_both_is_refused_and_nothing_is_created(staffed, fakes):  # noqa: F811
    r = staffed.post("/api/admin/companies/accenture/provision")  # the fixture's company already has one in each
    assert r.status_code == 409 and r.json()["code"] == "already_provisioned" and fakes.deal.extra == [] and fakes.rfp.extra == []


def test_providing_workspaces_to_an_existing_company_is_all_or_nothing_and_admin_only(staffed, fakes):  # noqa: F811
    add_company(staffed)
    assert staffed.member.post("/api/admin/companies/globex/provision").status_code == 403
    fakes.rfp.down = True
    r = staffed.post("/api/admin/companies/globex/provision")
    assert r.status_code == 502 and "Nothing was created" in r.json()["detail"]
    assert fakes.deal.extra == [] and company(staffed, "Globex")["workspaces"] == {"deals": [], "rfp": []}
    assert staffed.post("/api/admin/companies/nobody/provision").status_code == 404


def test_deleting_data_needs_the_administrators_own_password(staffed, fakes):  # noqa: F811
    created = add_company(staffed, provision=True).json()["created"]
    confirm = [f"deals:{created['deals']}", f"rfp:{created['rfp']}"]
    for password in (None, "", "not my password"):
        body = {"delete_data": True, "confirm": confirm, **({"password": password} if password is not None else {})}
        r = staffed.post("/api/admin/companies/globex/delete", json=body)
        assert r.status_code == 403 and r.json()["code"] == "password_required"
    assert company(staffed, "Globex") and fakes.deal.removed == [] and fakes.rfp.removed == []
    assert audit_actions(staffed).count("auth.reauth_failed") == 3


def test_deleting_a_company_but_keeping_its_data_needs_no_password(staffed, fakes):  # noqa: F811
    add_company(staffed, provision=True)
    assert staffed.post("/api/admin/companies/globex/delete", json={"delete_data": False}).status_code == 200
