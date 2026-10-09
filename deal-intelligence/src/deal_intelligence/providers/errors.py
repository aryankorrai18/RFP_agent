"""What a failed model call means for the person using the app, and what to do about it.

Every provider adapter raises LLMError with a `reason` from REASONS. Briefs, extractions and
jobs store their error as plain text, so `reason_of(message)` reads the reason back out of a stored
message: the adapters' messages carry a stable phrase for each reason (tests check the round trip),
and older wordings are recognised too so errors stored before this module still explain themselves.

`provider_health` remembers the last call per (provider, model) in this process, so the status bar
can say "Gemini quota used up for this model, switch model" before anyone opens a failed brief.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ..config import PROVIDER_KEY_VARS

REASONS = (
    "quota_exhausted",  # daily quota or credit used up: retrying won't help until it resets
    "rate_limited",  # per-minute limit hit even after automatic retries
    "model_unavailable",  # the key can't use this model (not found, retired, not enabled)
    "auth",  # key missing or rejected
    "unreachable",  # network: the request never got an answer
    "provider_error",  # the provider failed on its side (5xx, overloaded)
    "too_large",  # the request is over this model's or plan's size limit
    "unsupported_input",  # this provider can't read this kind of input
    "refused",  # the provider's safety filter declined
    "malformed",  # the output didn't match the required format, twice
    "unknown",
)

# A reason in this set will fail every remaining call in a job the same way, so a job stops
# instead of spending the rest of its calls (and minutes of retries) on the same error.
BLOCKING = frozenset({"quota_exhausted", "model_unavailable", "auth"})
# Reasons about the provider or model as a whole (not about one question), shown in the status bar.
PROVIDER_WIDE = BLOCKING | {"rate_limited", "provider_error", "unreachable"}

# The phrase each adapter puts in its message; reason_of() looks for these first.
PHRASE = {
    "quota_exhausted": "quota used up",
    "rate_limited": "rate limit reached",
    "model_unavailable": "model not available",
    "auth": "API key missing or rejected",
    "unreachable": "could not reach",
    "provider_error": "provider error",
    "too_large": "request too large",
    "unsupported_input": "can't read this file",
    "refused": "declined",
    "malformed": "unusable output",
}

# Wordings used before PHRASE existed (and still stored in older rows), checked after PHRASE.
# Kept narrow on purpose: the same fields also hold non-model errors ("No deal found",
# "Stopped by you."), which must not be read as provider problems.
_LEGACY = [
    ("auth", r"no \w+ (api key|credentials) found|rejected the (api key|credentials)"),
    ("model_unavailable", r"model not found|not found on groq|is not found for api version|not supported for generatecontent"),
    ("quota_exhausted", r"perday|per day \(|credit balance is too low|limit: 0\b"),
    ("rate_limited", r"rate limit|quota reached|resource.?exhausted"),
    ("too_large", r"larger than this \w+ model|context length|maximum context|exceeds the maximum number of tokens"),
    ("unreachable", r"could not reach the \w+ api"),
    ("provider_error", r"api error 5\d\d|overloaded"),
    ("refused", r"\w+ declined|blocked the request"),
    ("malformed", r"did not match the expected schema|empty response \(finish|no structured output|did not produce valid json"),
]

PROVIDER_NAMES = {"gemini": "Gemini", "groq": "Groq", "anthropic": "Anthropic"}


def reason_of(message: str | None) -> str | None:
    """The reason behind a stored error message, or None when there is no error."""
    if not message:
        return None
    text = message.lower()
    # Our own wording comes before the provider's raw message, which adapters put in parentheses;
    # matching phrases only there stops a stray word in the provider's text from misleading us.
    ours = text.split("(", 1)[0]
    for reason, phrase in PHRASE.items():
        if phrase.lower() in ours:
            return reason
    for reason, pattern in _LEGACY:
        if re.search(pattern, text):
            return reason
    return "unknown"


def provider_of(message: str | None, default: str | None = None) -> str | None:
    text = (message or "").lower()
    for provider, name in PROVIDER_NAMES.items():
        if name.lower() in text:
            return provider
    return default


def quota_is_hard(*texts: str) -> bool:
    """True when a 429 won't clear within a minute: a daily quota, used-up credit, or a model this
    plan gets no quota for at all (Gemini says "limit: 0"). A per-minute limit returns False."""
    joined = " ".join(t for t in texts if t).lower()
    return bool(re.search(r"per ?day|perday|daily|\(tpd\)|\(rpd\)|credit balance|limit: 0\b", joined))


def retry_after_seconds(text: str | None) -> float | None:
    """'Please retry in 37.5s' or 'retryDelay': '37s' → 37.5 / 37."""
    if not text:
        return None
    match = re.search(r"retry in ([\d.]+)\s*s|retrydelay['\"]?:\s*['\"]?([\d.]+)s", text.lower())
    if not match:
        return None
    return float(match.group(1) or match.group(2))


@dataclass(frozen=True)
class Explanation:
    reason: str
    title: str
    detail: str
    action: str
    retry_helps: bool
    switch_model: bool
    blocking: bool

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "title": self.title, "detail": self.detail, "action": self.action,
                "retry_helps": self.retry_helps, "switch_model": self.switch_model, "blocking": self.blocking}


def explain(reason: str | None, provider: str | None = None, model: str | None = None) -> Explanation | None:
    """Plain-language title, what happened, what to do, and whether retrying or switching helps."""
    if reason is None:
        return None
    name = PROVIDER_NAMES.get(provider or "", "The provider")
    key_var = PROVIDER_KEY_VARS.get(provider or "", ("the API key",))[0]
    this_model = model or "this model"
    text = {
        "quota_exhausted": (
            f"{name} quota used up for {this_model}",
            f"The key has used its quota for {this_model}. On Gemini's free tier the daily limit resets at midnight "
            "Pacific time. Retrying now fails the same way.",
            "Pick another model with the model button in the top bar and try again, or wait for the quota to reset.",
            False, True),
        "rate_limited": (
            f"Too many requests to {name} right now",
            "The per-minute limit was still hit after the app's automatic retries.",
            "Wait a minute, then try again. If it keeps happening, set DEAL_CONCURRENCY=1 in .env or pick another model.",
            True, True),
        "model_unavailable": (
            f"{this_model} isn't available to this key",
            f"{name} doesn't offer {this_model} to your key: it may be retired, renamed or not enabled.",
            "Pick another model with the model button in the top bar and try again.",
            False, True),
        "auth": (
            f"The {name} API key is missing or was rejected",
            "Every call will fail until the key is fixed. Switching to another model from the same provider won't help.",
            f"Check {key_var} in deal-intelligence/.env. The app re-reads .env on every request, so no restart is needed.",
            False, False),
        "unreachable": (
            f"Couldn't reach {name}",
            "The request didn't get an answer (network, DNS, proxy or firewall).",
            "Check the internet connection, then try again.",
            True, False),
        "provider_error": (
            f"{name} had a problem on its side",
            "The provider returned a server error or said it was overloaded.",
            "Wait a few minutes and try again, or pick another model.",
            True, True),
        "too_large": (
            f"The request is too large for {this_model}",
            "The deal records, call notes and recalled memories together are over this model's or plan's limit.",
            "Pick a model with a larger limit, or use a shorter retrieval mode.",
            False, True),
        "unsupported_input": (
            f"{name} can't read this file",
            "This provider doesn't accept this kind of input.",
            "Pick another provider or model, or upload a text-based version.",
            False, True),
        "refused": (
            "The model declined to answer",
            "The provider's safety filter blocked this request or its answer.",
            "Regenerate the brief, or try another model.",
            False, True),
        "malformed": (
            "The model returned an unusable answer",
            "Its output didn't match the required format, twice in a row.",
            "Try again: this is usually a one-off. If it repeats, pick a stronger model.",
            True, True),
        "unknown": (
            "The model call failed",
            "The error didn't match a known cause; the details are below.",
            "Try again. If it fails again, pick another model and check the details.",
            True, True),
    }[reason if reason in REASONS else "unknown"]
    title, detail, action, retry_helps, switch_model = text
    return Explanation(reason, title, detail, action, retry_helps, switch_model, reason in BLOCKING)


def explain_message(message: str | None, provider: str | None = None, model: str | None = None) -> dict | None:
    """For API views: the explanation of a stored error, or None when there is no error or it didn't
    come from a model call (a file that can't be parsed, an empty upload, a stop the person asked for)."""
    reason = reason_of(message)
    if reason is None or (reason == "unknown" and provider_of(message) is None):
        return None
    return explain(reason, provider_of(message, provider), model).as_dict()


class StopOnBlocking:
    """Stops a batch of model calls after the first error that every remaining call would repeat
    (quota used up, model not available, key rejected). Calls already in flight finish normally."""

    def __init__(self) -> None:
        self.reason: str | None = None
        self.message: str | None = None
        self.skipped = 0

    @property
    def tripped(self) -> bool:
        return self.reason is not None

    def record(self, error: str | None) -> None:
        reason = reason_of(error)
        if self.reason is None and reason in BLOCKING:
            self.reason, self.message = reason, error

    def summary(self, attempted: int, total: int, provider: str | None, model: str | None,
                action: str | None = None) -> str | None:
        """One line for a job's warning or error. It starts "Stopped early, <phrase>:" so reason_of()
        reads the reason back from the stored text, and it never contains "; "."""
        if not self.tripped:
            return None
        info = explain(self.reason, provider_of(self.message, provider), model)
        return (f"Stopped early, {PHRASE[self.reason]}: {info.title}. {self.skipped} of {total} weren't attempted, so no "
                f"more calls were spent on an error that would repeat ({attempted} were tried). {action or info.action}")


# --- provider health (this process only) --------------------------------------------------------


@dataclass
class _Health:
    ok_at: float | None = None
    failed_at: float | None = None
    reason: str | None = None
    message: str | None = None


@dataclass
class ProviderHealth:
    """The last outcome per (provider, model). A success clears an earlier failure."""

    _state: dict[tuple[str, str], _Health] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def success(self, provider: str, model: str) -> None:
        with self._lock:
            entry = self._state.setdefault((provider, model), _Health())
            entry.ok_at, entry.reason, entry.message = time.time(), None, None

    def failure(self, provider: str, model: str, reason: str, message: str) -> None:
        if reason not in PROVIDER_WIDE:
            return  # about one question (refused, malformed, too large), not the model as a whole
        with self._lock:
            entry = self._state.setdefault((provider, model), _Health())
            entry.failed_at, entry.reason, entry.message = time.time(), reason, message

    def view(self, provider: str, model: str) -> dict[str, Any]:
        with self._lock:
            entry = self._state.get((provider, model))
        if entry is None or entry.reason is None:
            return {"state": "ok" if entry and entry.ok_at else "unknown", "model": model}
        return {
            "state": "failing", "model": model, "since": entry.failed_at, "message": entry.message,
            "explanation": explain(entry.reason, provider, model).as_dict(),
        }

    def clear(self) -> None:
        with self._lock:
            self._state.clear()


provider_health = ProviderHealth()
