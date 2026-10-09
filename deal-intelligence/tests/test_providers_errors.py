"""Provider errors explain themselves: what happened, what to do, whether to switch model. A batch
stops after an error every remaining call would repeat. Offline."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from deal_intelligence.providers.base import LLMError, LLMResult, MonitoredLLM, TokenUsage
from deal_intelligence.providers.errors import (
    BLOCKING, PHRASE, REASONS, StopOnBlocking, explain, explain_message, provider_health, provider_of, quota_is_hard,
    reason_of, retry_after_seconds,
)

# Real wordings from the adapters and from earlier versions of the app, as stored in a failed row.
RETIRED_MODEL = ("brief: model not found; check DEAL_MODEL (This model models/gemini-2.5-flash-lite is no "
                 "longer available to new users. Please update your code to use models/gemini-3.5-flash-lite.")
OVERLOADED = ("signal extraction: Gemini API error 503 (This model is currently experiencing high demand. "
              "Spikes in demand are usually temporary. Please try again later.)")
OLD_RATE_LIMIT = "brief: Gemini rate limit or quota reached, even after retries (Resource has been exhausted)."
QUOTA = LLMError("api_error", "brief: Gemini quota used up for gemini-3.5-flash-lite (You exceeded your current quota)",
                 reason="quota_exhausted")


class Verdict(BaseModel):
    answer: str


@pytest.mark.parametrize("message, reason", [
    (RETIRED_MODEL, "model_unavailable"),
    (OVERLOADED, "provider_error"),
    (OLD_RATE_LIMIT, "rate_limited"),
    ("brief: No Gemini API key found. Set GEMINI_API_KEY in .env (see README).", "auth"),
    ("brief: Gemini declined (finish reason: SAFETY)", "refused"),
    ("brief: empty response (finish reason: MAX_TOKENS)", "malformed"),
    ("brief: Anthropic quota used up (Your credit balance is too low)", "quota_exhausted"),
    ("brief: could not reach the Groq API (ConnectError)", "unreachable"),
    ("brief: request too large for gemini-3.5-flash-lite", "too_large"),
])
def test_stored_messages_are_understood(message, reason):
    assert reason_of(message) == reason
    assert explain_message(message, "gemini", "gemini-2.5-flash-lite")["reason"] == reason


@pytest.mark.parametrize("message", [
    "Stopped by you.",
    "No deal was found for this id.",
    "The uploaded file is empty.",
    None,
    "",
])
def test_errors_that_are_not_model_errors_get_no_provider_explanation(message):
    assert explain_message(message, "gemini", "gemini-3.5-flash-lite") is None


def test_our_wording_beats_a_stray_word_in_the_providers_text():
    message = "brief: Gemini quota used up for m (the request was too large, rate limit, model not found)"
    assert reason_of(message) == "quota_exhausted"


def test_provider_is_read_from_the_message():
    assert provider_of("brief: Gemini quota used up") == "gemini"
    assert provider_of("brief: Anthropic rejected the request", "groq") == "anthropic"
    assert provider_of("no provider named", "groq") == "groq"


def test_every_reason_says_whether_retrying_or_switching_helps():
    for reason in REASONS:
        info = explain(reason, "gemini", "gemini-3.5-flash-lite")
        assert info.title and info.detail and info.action
    assert explain("quota_exhausted", "gemini", "m").switch_model and not explain("quota_exhausted", "gemini", "m").retry_helps
    assert explain("model_unavailable", "gemini", "m").switch_model
    assert not explain("auth", "groq", "m").switch_model and "GROQ_API_KEY" in explain("auth", "groq", "m").action
    assert "deal-intelligence/.env" in explain("auth", "groq", "m").action
    assert "DEAL_CONCURRENCY" in explain("rate_limited", "gemini", "m").action
    assert explain("rate_limited", "gemini", "m").retry_helps
    assert {r for r in REASONS if explain(r).blocking} == BLOCKING
    assert explain(None) is None and explain("nonsense").reason == "nonsense"


def test_phrases_round_trip_through_reason_of():
    for reason, phrase in PHRASE.items():
        assert reason_of(f"brief: Gemini {phrase} for the model (details)") == reason


def test_quota_is_hard_only_for_limits_that_will_not_clear_in_a_minute():
    assert quota_is_hard("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert quota_is_hard("tokens per day (TPD): Limit 100000")
    assert quota_is_hard("Your credit balance is too low")
    assert quota_is_hard("Quota exceeded, limit: 0")
    assert not quota_is_hard("tokens per minute (TPM)")
    assert not quota_is_hard("", "slow down")


def test_retry_after_is_read_from_both_wordings():
    assert retry_after_seconds("Please retry in 37.5s.") == 37.5
    assert retry_after_seconds("{'retryDelay': '37s'}") == 37
    assert retry_after_seconds("nothing here") is None and retry_after_seconds(None) is None


def test_breaker_trips_only_on_errors_that_would_repeat():
    breaker = StopOnBlocking()
    breaker.record("brief: unusable output: empty response (finish reason: STOP)")
    breaker.record(OVERLOADED)
    breaker.record(None)
    assert not breaker.tripped and breaker.summary(1, 2, "gemini", "m") is None  # one-off and transient errors don't stop a batch
    breaker.record(RETIRED_MODEL)
    assert breaker.tripped and breaker.reason == "model_unavailable"
    breaker.record(QUOTA.message)
    assert breaker.reason == "model_unavailable"  # the first blocking error is kept
    breaker.skipped = 19
    assert "19 of 21 weren't attempted" in breaker.summary(2, 21, "gemini", "gemini-2.5-flash-lite")
    # The page splits a job's warnings on "; " and drops the stop message, so it must never contain one,
    # and the stored summary must still read back as its reason (so the page can explain it).
    for reason in BLOCKING:
        stop = StopOnBlocking()
        stop.reason, stop.message = reason, f"brief: Gemini {reason}"
        text = stop.summary(1, 5, "gemini", "gemini-2.5-flash-lite")
        assert "; " not in text
        assert reason_of(text) == reason and explain_message(text, "gemini", "gemini-2.5-flash-lite")["reason"] == reason


def test_breaker_summary_can_carry_a_custom_action():
    breaker = StopOnBlocking()
    breaker.record(QUOTA.message)
    assert breaker.summary(1, 3, "gemini", "m", action="Switch model.").endswith("Switch model.")


class Inner:
    model = "gemini-3.5-flash-lite"

    def __init__(self) -> None:
        self.outcome: object = None
        self.calls: list[dict] = []

    async def structured(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return LLMResult(output=Verdict(answer="ok"), model=self.model, usage=TokenUsage(input_tokens=3))


def structured(llm: MonitoredLLM):
    return asyncio.run(llm.structured(purpose="brief", output_format=Verdict, system="s", user="u", temperature=0))


def test_monitored_llm_passes_the_call_through_and_exposes_the_model():
    inner = Inner()
    llm = MonitoredLLM(inner, "gemini")
    result = structured(llm)
    assert result.output.answer == "ok" and result.usage.input_tokens == 3 and llm.model == inner.model
    assert inner.calls == [{"purpose": "brief", "output_format": Verdict, "system": "s", "user": "u", "temperature": 0}]
    provider_health.clear()


def test_provider_health_follows_the_last_call():
    provider_health.clear()
    inner = Inner()
    llm = MonitoredLLM(inner, "gemini")
    assert provider_health.view("gemini", inner.model)["state"] == "unknown"

    inner.outcome = LLMError("malformed", "brief: unusable output: empty response", reason="malformed")
    with pytest.raises(LLMError):
        structured(llm)
    assert provider_health.view("gemini", inner.model)["state"] == "unknown"  # about one request, not the model

    inner.outcome = QUOTA
    with pytest.raises(LLMError) as raised:
        structured(llm)
    assert raised.value is QUOTA  # re-raised unchanged
    view = provider_health.view("gemini", inner.model)
    assert view["state"] == "failing" and view["explanation"]["reason"] == "quota_exhausted"
    assert view["explanation"]["switch_model"] is True and view["message"] == QUOTA.message
    assert provider_health.view("gemini", "gemini-3.8-flash")["state"] == "unknown"  # another model is unaffected
    assert provider_health.view("groq", inner.model)["state"] == "unknown"  # and so is another provider

    inner.outcome = None
    structured(llm)
    assert provider_health.view("gemini", inner.model)["state"] == "ok"
    provider_health.clear()


def test_monitored_llm_does_not_swallow_other_exceptions():
    provider_health.clear()
    inner = Inner()
    inner.outcome = RuntimeError("bug")
    with pytest.raises(RuntimeError):
        structured(MonitoredLLM(inner, "gemini"))
    assert provider_health.view("gemini", inner.model)["state"] == "unknown"


def test_llm_error_reads_its_reason_from_the_message_when_none_is_given():
    assert LLMError("api_error", "brief: Gemini quota used up for m").reason == "quota_exhausted"
    assert LLMError("api_error", "something odd").reason == "unknown"
