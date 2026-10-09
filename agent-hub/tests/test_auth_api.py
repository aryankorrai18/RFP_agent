"""Signing in over HTTP: who may call what, and that one person's chats are invisible to everyone else."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent_hub import main
from agent_hub.auth import Auth

PASSWORD = "correct horse battery"


@pytest.fixture
def web(engine, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    monkeypatch.delenv("HUB_COOKIE_SECURE", raising=False)
    engine.auth = Auth(engine.store)
    auth = engine.auth
    auth.create_company("Accenture", {"deals": ["ws-demo"], "rfp": ["ws-acme"]})
    auth.create_company("Globex", {"deals": ["ws-other"], "rfp": ["ws-globex"]})
    main.app.state.engine = engine
    with TestClient(main.app, base_url="http://hub.test") as client:
        client.engine = engine
        yield client
    main.app.state.engine = None


def make_user(web, email="dana@accenture.com", company="Accenture"):
    web.engine.auth.create_user(email, PASSWORD, company, "Dana")


def sign_in(web, email="dana@accenture.com", password=PASSWORD):
    return web.post("/api/auth/login", json={"email": email, "password": password})


def second_browser(web) -> TestClient:
    return TestClient(main.app, base_url="http://hub.test")


# -- open until the first account exists -------------------------------------------------------------------

def test_with_no_accounts_the_hub_is_open_as_before(web):
    assert web.get("/api/auth/me").json() == {"auth": "off", "user": None, "setup_available": False}  # the test client is not this machine
    assert web.post("/api/conversations").status_code == 200 and web.get("/api/agents").status_code == 200


def test_login_is_refused_while_sign_in_is_off(web):
    assert sign_in(web).status_code == 400


def test_hub_auth_off_keeps_it_open_even_with_accounts(web, monkeypatch):
    make_user(web)
    monkeypatch.setenv("HUB_AUTH", "off")
    assert web.post("/api/conversations").status_code == 200


# -- once an account exists ----------------------------------------------------------------------------------

@pytest.mark.parametrize("method, path", [
    ("get", "/api/agents"), ("post", "/api/chat"), ("post", "/api/conversations"),
    ("get", "/api/conversations/x"), ("get", "/api/conversations/x/events"), ("post", "/api/conversations/x/messages"),
    ("post", "/api/conversations/x/actions"), ("get", "/api/conversations/x/downloads/t")])
def test_every_data_route_needs_a_session(web, method, path):
    make_user(web)
    assert getattr(web, method)(path, **({"json": {"message": "hi"}} if path == "/api/chat" else {})).status_code == 401


def test_the_page_the_health_check_and_the_session_check_stay_open(web):
    make_user(web)
    assert web.get("/").status_code == 200 and web.get("/health").status_code == 200
    assert web.get("/api/auth/me").json() == {"auth": "on", "user": None, "setup_available": False}


def test_signing_in_sets_a_safe_cookie_and_unlocks_the_hub(web):
    make_user(web)
    response = sign_in(web)
    cookie = response.headers["set-cookie"].lower()
    assert response.status_code == 200 and response.json()["user"] == {"email": "dana@accenture.com", "name": "Dana", "company": "Accenture", "role": "member", "operator": False}
    assert "httponly" in cookie and "samesite=lax" in cookie and "path=/" in cookie and "secure" not in cookie
    assert web.get("/api/auth/me").json()["user"]["company"] == "Accenture" and web.post("/api/conversations").status_code == 200


def test_the_cookie_is_secure_on_https_or_when_asked(web, monkeypatch):
    make_user(web)
    monkeypatch.setenv("HUB_COOKIE_SECURE", "1")
    assert "secure" in sign_in(web).headers["set-cookie"].lower()


def test_wrong_password_and_unknown_email_get_the_same_answer(web):
    make_user(web)
    wrong, unknown = sign_in(web, password="wrong password!!"), sign_in(web, email="nobody@x.com")
    assert wrong.status_code == unknown.status_code == 401 and wrong.json() == unknown.json()
    assert "set-cookie" not in wrong.headers


def test_repeated_failures_lock_the_account_out_with_a_retry_after(web):
    make_user(web)
    for _ in range(5):
        sign_in(web, password="wrong password!!")
    locked = sign_in(web)
    assert locked.status_code == 429 and int(locked.headers["retry-after"]) > 0


def test_signing_out_ends_the_session_even_if_the_cookie_is_replayed(web):
    make_user(web)
    sign_in(web)
    stolen = web.cookies.get("hub_session")
    assert web.post("/api/auth/logout").json()["user"] is None
    assert web.post("/api/conversations").status_code == 401
    web.cookies.set("hub_session", stolen)
    assert web.post("/api/conversations").status_code == 401


def test_a_made_up_cookie_is_not_a_session(web):
    make_user(web)
    web.cookies.set("hub_session", "x" * 43)
    assert web.get("/api/agents").status_code == 401


# -- one person's chats are invisible to everyone else ----------------------------------------------------------

def test_another_users_chat_looks_like_it_does_not_exist(web):
    make_user(web)
    make_user(web, "gus@globex.com", "Globex")
    sign_in(web)
    mine = web.post("/api/conversations").json()["id"]
    gus = second_browser(web)
    gus.post("/api/auth/login", json={"email": "gus@globex.com", "password": PASSWORD})
    for response in (gus.get(f"/api/conversations/{mine}"), gus.get(f"/api/conversations/{mine}/events"),
                     gus.post(f"/api/conversations/{mine}/messages", data={"text": "hi"}),
                     gus.post(f"/api/conversations/{mine}/actions", json={"action_id": "x"}),
                     gus.get(f"/api/conversations/{mine}/downloads/t")):
        assert response.status_code == 404
    assert web.get(f"/api/conversations/{mine}").status_code == 200
    assert gus.get("/api/conversations/does-not-exist").status_code == 404


def test_a_chat_made_before_sign_in_belongs_to_nobody(web):
    old = web.post("/api/conversations").json()["id"]  # sign-in is off, so no owner
    make_user(web)
    sign_in(web)
    assert web.get(f"/api/conversations/{old}").status_code == 404


def test_a_disabled_user_loses_access_at_once(web):
    make_user(web)
    sign_in(web)
    web.engine.auth.disable_user("dana@accenture.com")
    make_user(web, "gus@globex.com", "Globex")  # keeps sign-in switched on
    assert web.get("/api/agents").status_code == 401


# -- requests from another website ------------------------------------------------------------------------------

def test_a_post_from_another_site_is_refused_even_with_a_valid_cookie(web):
    make_user(web)
    sign_in(web)
    assert web.post("/api/conversations", headers={"Origin": "http://evil.example"}).status_code == 403
    assert web.post("/api/conversations", headers={"Origin": "http://hub.test"}).status_code == 200
    assert web.post("/api/auth/logout", headers={"Origin": "http://evil.example"}).status_code == 403


def test_a_login_from_another_site_is_refused(web):
    make_user(web)
    response = web.post("/api/auth/login", json={"email": "dana@accenture.com", "password": PASSWORD},
                        headers={"Origin": "http://evil.example"})
    assert response.status_code == 403 and "set-cookie" not in response.headers
