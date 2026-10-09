"""Session management: lifetimes and idle limits (shorter for administrators), background polling doesn't count as use,
a person can see and end their sessions, an administrator can sign someone out everywhere, sign-ins are in the
activity log, and deleting a company's data needs the administrator's password."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_hub import auth as auth_module
from agent_hub.auth import Auth, describe_browser

from .test_auth_api import PASSWORD, make_user, second_browser, sign_in, web  # noqa: F401

CHROME_WIN = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"
SAFARI_IOS = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(auth_module, "_now", c)
    return c


@pytest.fixture
def auth(engine, clock, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    a = Auth(engine.store)
    a.create_company("Acme", {"deals": ["ws-demo"], "rfp": ["ws-acme"]})
    a.create_user("dana@acme.com", PASSWORD, "Acme", "Dana")
    return a


def test_browser_descriptions_are_short_and_never_the_raw_header():
    assert describe_browser(CHROME_WIN) == "Chrome on Windows"
    assert describe_browser(SAFARI_IOS) == "Safari on iPhone"
    assert describe_browser("Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/131.0") == "Firefox on Linux"
    assert describe_browser("") == "Unknown device" and describe_browser("curl/8.5") == "Unknown browser"


def test_a_member_session_ends_after_twelve_idle_hours_and_use_keeps_it_alive(auth, clock):
    token, _ = auth.login("dana@acme.com", PASSWORD, "10.0.0.5", CHROME_WIN)
    clock.now += timedelta(hours=11)
    assert auth.user_for_token(token) is not None  # used: the idle clock restarts
    clock.now += timedelta(hours=11)
    assert auth.user_for_token(token) is not None
    clock.now += timedelta(hours=12, minutes=1)
    assert auth.user_for_token(token) is None
    assert auth.list_sessions(auth.user_id_for_email("dana@acme.com")) == []  # removed, not just refused


def test_background_polling_does_not_keep_a_session_alive(auth, clock):
    token, _ = auth.login("dana@acme.com", PASSWORD)
    for _ in range(13):
        clock.now += timedelta(hours=1)
        if clock.now < datetime(2026, 10, 9, 21, 0, tzinfo=timezone.utc):
            assert auth.user_for_token(token, touch=False) is not None
    assert auth.user_for_token(token, touch=False) is None


def test_a_member_session_never_outlives_seven_days_even_when_used(auth, clock):
    token, _ = auth.login("dana@acme.com", PASSWORD)
    for _ in range(7 * 2):
        clock.now += timedelta(hours=12) - timedelta(minutes=1)
        auth.user_for_token(token)
    clock.now += timedelta(hours=12)
    assert auth.user_for_token(token) is None


def test_administrators_get_twelve_hours_and_an_hour_idle(engine, clock, monkeypatch):
    monkeypatch.delenv("HUB_AUTH", raising=False)
    a = Auth(engine.store)
    a.create_first_admin("ops@hub.example", PASSWORD, "Ops")
    token, admin = a.login("ops@hub.example", PASSWORD)
    assert a.session_seconds(admin) == 12 * 3600
    clock.now += timedelta(minutes=59)
    assert a.user_for_token(token) is not None
    clock.now += timedelta(minutes=61)
    assert a.user_for_token(token) is None
    token, _ = a.login("ops@hub.example", PASSWORD)
    for _ in range(14):  # 12 h 50 min, never idle for an hour
        clock.now += timedelta(minutes=55)
        last = a.user_for_token(token)
    assert last is None  # 12 hours is the most, however active


def test_sessions_are_listed_and_ended_one_at_a_time_or_all_others(auth, clock):
    laptop, _ = auth.login("dana@acme.com", PASSWORD, "10.0.0.5", CHROME_WIN)
    clock.now += timedelta(minutes=5)
    phone, _ = auth.login("dana@acme.com", PASSWORD, "10.0.0.9", SAFARI_IOS)
    clock.now += timedelta(minutes=5)
    tablet, _ = auth.login("dana@acme.com", PASSWORD, "10.0.0.7", "")
    dana = auth.user_id_for_email("dana@acme.com")
    seen = auth.list_sessions(dana, laptop)
    assert [s["browser"] for s in seen] == ["Unknown device", "Safari on iPhone", "Chrome on Windows"]
    assert [s["current"] for s in seen] == [False, False, True] and all(len(s["id"]) == 16 for s in seen)
    assert auth.end_session(dana, seen[1]["id"]) == 1 and auth.user_for_token(phone) is None
    assert auth.end_session(dana, "not-a-session-id") == 0
    assert auth.end_other_sessions(dana, laptop) == 1 and auth.user_for_token(tablet) is None
    assert auth.user_for_token(laptop) is not None


def test_sign_ins_failures_lockouts_and_sign_outs_are_in_the_activity_log(auth):
    for _ in range(5):
        with pytest.raises(auth_module.AuthError):
            auth.login("dana@acme.com", "wrong password", "10.0.0.5", CHROME_WIN)
    with pytest.raises(auth_module.AuthError):
        auth.login("dana@acme.com", PASSWORD, "10.0.0.5", CHROME_WIN)
    token, _ = auth.login("dana@acme.com", PASSWORD, "10.0.0.6", CHROME_WIN)
    auth.logout(token)
    actions = [e["action"] for e in reversed(auth.list_audit(50)) if e["actor"] == "dana@acme.com"]
    assert actions.count("auth.signin_failed") == 5 and "auth.locked" in actions
    assert actions[-2:] == ["auth.signin", "auth.signout"]
    detail = next(e["detail"] for e in auth.list_audit(50) if e["action"] == "auth.signin")
    assert detail == "from 10.0.0.6, Chrome on Windows" and PASSWORD not in str(auth.list_audit(50))


def test_cleanup_removes_sessions_past_their_limits(auth, clock):
    auth.login("dana@acme.com", PASSWORD)
    clock.now += timedelta(hours=13)
    assert auth.cleanup_sessions() == 1


# -- over HTTP ---------------------------------------------------------------------------------------------------------

def test_your_sessions_over_http(web):  # noqa: F811
    make_user(web)
    assert web.post("/api/auth/login", json={"email": "dana@accenture.com", "password": PASSWORD},
                    headers={"user-agent": CHROME_WIN}).status_code == 200
    other = second_browser(web)
    assert other.post("/api/auth/login", json={"email": "dana@accenture.com", "password": PASSWORD},
                      headers={"user-agent": SAFARI_IOS}).status_code == 200
    mine = web.get("/api/auth/sessions").json()
    assert mine["limits"] == {"idle_minutes": 720, "lifetime_hours": 168}
    assert sorted(s["browser"] for s in mine["sessions"]) == ["Chrome on Windows", "Safari on iPhone"]
    assert sum(s["current"] for s in mine["sessions"]) == 1
    assert web.post("/api/auth/sessions/sign-out-others").json() == {"ended": 1}
    assert other.get("/api/auth/sessions").status_code == 401  # the phone was signed out
    me = web.get("/api/auth/sessions").json()["sessions"][0]
    out = web.delete(f"/api/auth/sessions/{me['id']}")
    assert out.json() == {"ok": True, "signed_out": True} and web.get("/api/auth/sessions").status_code == 401
    assert web.delete("/api/auth/sessions/0123456789abcdef").status_code == 401


def test_an_administrator_can_see_and_end_someones_sessions(web):  # noqa: F811
    web.engine.auth.create_first_admin("ops@hub.test", PASSWORD, "Ops")
    make_user(web)
    member = second_browser(web)
    assert sign_in(member).status_code == 200
    assert sign_in(web, "ops@hub.test").status_code == 200
    seen = web.get("/api/admin/users/dana@accenture.com/sessions").json()["sessions"]
    assert len(seen) == 1 and "current" not in seen[0]
    assert web.post("/api/admin/users/dana@accenture.com/sessions/end").json() == {"ended": 1}
    assert member.get("/api/auth/sessions").status_code == 401
    assert member.post("/api/admin/users/ops@hub.test/sessions/end").status_code == 401
    assert any(e["action"] == "user.sessions_ended" for e in web.engine.auth.list_audit(20))
