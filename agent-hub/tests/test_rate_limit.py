"""Per-person request limits: past a bucket's limit the hub answers 429 with Retry-After; people don't share buckets;
sign-in attempts are counted per address; the window slides."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent_hub import main, ratelimit


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def limited(engine, monkeypatch):
    monkeypatch.setenv("HUB_RATE_LIMIT", "on")
    monkeypatch.delenv("HUB_AUTH", raising=False)
    clock = Clock()
    main.app.state.engine = engine
    main.app.state.rate_limiter = ratelimit.RateLimiter(clock)
    with TestClient(main.app) as client:
        client.clock = clock
        yield client
    main.app.state.engine = None
    main.app.state.rate_limiter = None


def test_new_chats_past_the_limit_get_429_with_retry_after_and_the_window_slides(limited):
    for _ in range(20):
        assert limited.post("/api/conversations").status_code == 200
    refused = limited.post("/api/conversations")
    assert refused.status_code == 429 and refused.headers["retry-after"] == "60"
    assert "Try again in 60 seconds" in refused.json()["detail"] and refused.headers["x-frame-options"] == "DENY"
    limited.clock.now += 30
    assert limited.post("/api/conversations").headers["retry-after"] == "30"
    limited.clock.now += 31
    assert limited.post("/api/conversations").status_code == 200


def test_reading_is_not_slowed_by_a_writing_limit(limited):
    conv = limited.post("/api/conversations").json()["id"]
    for _ in range(19):
        limited.post("/api/conversations")
    assert limited.post("/api/conversations").status_code == 429
    assert limited.get(f"/api/conversations/{conv}/events").status_code == 200  # its own, larger bucket


def test_people_do_not_share_a_bucket(limited):
    limited.cookies.set(main.SESSION_COOKIE, "session-of-dana")
    for _ in range(20):
        limited.post("/api/conversations")
    assert limited.post("/api/conversations").status_code == 429
    limited.cookies.set(main.SESSION_COOKIE, "session-of-sam")
    assert limited.post("/api/conversations").status_code != 429


def test_sign_in_attempts_are_counted_per_address_whatever_the_session():
    clock = Clock()
    limiter = ratelimit.RateLimiter(clock)
    rule = ratelimit.rule_for("POST", "/api/auth/login")
    sessions = [f"cookie-{i}" for i in range(11)]
    results = [limiter.hit(rule, ratelimit.who(s, "203.0.113.9", rule)) for s in sessions]
    assert results[:10] == [None] * 10 and results[10] == 60
    assert limiter.hit(rule, ratelimit.who(None, "198.51.100.4", rule)) is None


def test_the_rules_cover_the_routes_that_matter_and_never_the_pages():
    assert ratelimit.rule_for("POST", "/api/conversations/abc/messages").name == "messages"
    assert ratelimit.rule_for("POST", "/api/conversations/abc/actions").name == "actions"
    assert ratelimit.rule_for("PATCH", "/api/conversations/abc").name == "chat changes"
    assert ratelimit.rule_for("POST", "/api/admin/users").name == "admin changes"
    assert ratelimit.rule_for("GET", "/api/admin/usage").name == "api"
    assert ratelimit.rule_for("GET", "/") is None and ratelimit.rule_for("GET", "/admin") is None
    assert ratelimit.who("tok", "1.2.3.4", ratelimit.RULES[1]).startswith("s:") and "tok" not in ratelimit.who("tok", "", ratelimit.RULES[1])


def test_limits_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("HUB_RATE_LIMIT", "off")
    assert ratelimit.enabled() is False
    monkeypatch.setenv("HUB_RATE_LIMIT", "on")
    assert ratelimit.enabled() is True
