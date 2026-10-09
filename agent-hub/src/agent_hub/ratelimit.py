"""Per-person request limits for the hub's API, so one browser tab, script or account can't flood it.

A request is counted against a bucket for whoever sent it: the signed-in session if there is one, else the address it
came from. Each bucket allows `limit` requests in a sliding `window` of seconds; past that the hub answers 429 with a
Retry-After header, which the pages already honour (they wait and retry reads, and never repeat a write by themselves).
Counts live in memory: a restart clears them, which is fine for limits measured in minutes.

Set HUB_RATE_LIMIT=off to turn the limits off (the tests do, except the ones that check them)."""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    methods: tuple[str, ...]
    pattern: re.Pattern[str]
    limit: int
    window: float = 60.0
    by_address: bool = False  # count per address even when signed in (sign-in attempts)


_ANY = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE")
_WRITE = ("POST", "PUT", "PATCH", "DELETE")

# First match wins, so the specific rules come before the catch-all.
RULES = (
    Rule("sign-in", ("POST",), re.compile(r"^/api/(?:auth/login|admin/setup)$"), 10, by_address=True),
    Rule("messages", ("POST",), re.compile(r"^/api/conversations/[^/]+/messages$"), 30),
    Rule("actions", ("POST",), re.compile(r"^/api/conversations/[^/]+/actions$"), 60),
    Rule("new chats", ("POST",), re.compile(r"^/api/conversations$"), 20),
    Rule("chat changes", ("PATCH", "DELETE"), re.compile(r"^/api/conversations/[^/]+$"), 30),
    Rule("admin changes", _WRITE, re.compile(r"^/api/admin/"), 60),
    Rule("api", _ANY, re.compile(r"^/api/"), 600),  # polling an open chat every second is ~60 a minute
)


def enabled() -> bool:
    return os.environ.get("HUB_RATE_LIMIT", "on").strip().lower() not in ("off", "0", "false", "no")


def rule_for(method: str, path: str) -> Rule | None:
    return next((r for r in RULES if method in r.methods and r.pattern.match(path)), None)


class RateLimiter:
    def __init__(self, clock=time.monotonic) -> None:  # noqa: ANN001 - a callable returning seconds
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()
        self._checks = 0

    def hit(self, rule: Rule, who: str) -> int | None:
        """Count one request. None if it is allowed; otherwise the whole seconds to wait before the next one is."""
        now = self._clock()
        key = (rule.name, who)
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= rule.window:
                hits.popleft()
            if len(hits) >= rule.limit:
                return max(1, math.ceil(rule.window - (now - hits[0])))
            hits.append(now)
            self._checks += 1
            if self._checks % 1000 == 0:  # forget people who have gone quiet, so memory stays small
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < 3600}
            return None


def who(session_token: str | None, address: str, rule: Rule) -> str:
    """The bucket's owner: a hash of the session (never the token itself), or the address."""
    if session_token and not rule.by_address:
        return "s:" + hashlib.sha256(session_token.encode()).hexdigest()[:24]
    return "a:" + (address or "unknown")
