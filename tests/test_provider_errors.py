"""Provider errors explain themselves: what happened, what to do, whether to switch model. A job
stops after an error every remaining call would repeat. Offline."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from rfp_assistant.providers.base import LLMError, MonitoredLLM
from rfp_assistant.main import app
from rfp_assistant.providers.errors import (
    BLOCKING, REASONS, StopOnBlocking, explain, explain_message, provider_health, reason_of,
)
from rfp_assistant.api.v1 import experiment, projects
from rfp_assistant.api.v1.db import DraftRow, Job
from tests.conftest import docx_bytes
from tests.test_hindsight_lessons import QUERY, library_with_competing_answers
from tests.v1_fakes import FakeLessons, FakeV1LLM, cite_first_past_answer, make_context, req

# Real messages stored by earlier versions of the app (from the live database), before PHRASE existed.
RETIRED_MODEL = ("drafting REQ-001: model not found; check RFP_MODEL (This model models/gemini-2.5-flash-lite is no "
                 "longer available to new users. Please update your code to use models/gemini-3.5-flash-lite.")
OVERLOADED = ("requirement extraction: Gemini API error 503 (This model is currently experiencing high demand. "
              "Spikes in demand are usually temporary. Please try again later.)")
OLD_RATE_LIMIT = "drafting REQ-004: Gemini rate limit or quota reached, even after retries (Resource has been exhausted)."
QUOTA = LLMError("api_error", "drafting REQ-001: Gemini quota used up for gemini-3.5-flash-lite (You exceeded your current quota)",
                 reason="quota_exhausted")


@pytest.mark.parametrize("message, reason", [
    (RETIRED_MODEL, "model_unavailable"),
    (OVERLOADED, "provider_error"),
    (OLD_RATE_LIMIT, "rate_limited"),
    ("requirement extraction: No Gemini API key found. Set GEMINI_API_KEY in .env (see README).", "auth"),
    ("drafting REQ-002: Gemini declined (finish reason: SAFETY)", "refused"),
    ("drafting REQ-003: empty response (finish reason: MAX_TOKENS)", "malformed"),
])
def test_older_stored_messages_are_still_understood(message, reason):
    assert reason_of(message) == reason
    assert explain_message(message, "gemini", "gemini-2.5-flash-lite")["reason"] == reason


@pytest.mark.parametrize("message", [
    "Stopped by you.",
    "No requirements were found in rfp.docx.",
    "rfp.docx has 504 requirements; the limit is 150.",
    "Stopped before the requirements were extracted. Retry extraction to run it again.",
    "Every draft failed, so there is nothing to compare. Check the model and try again.",
    None,
])
def test_errors_that_are_not_model_errors_get_no_provider_explanation(message):
    assert explain_message(message, "gemini", "gemini-3.5-flash-lite") is None


def test_every_reason_says_whether_retrying_or_switching_helps():
    for reason in REASONS:
        info = explain(reason, "gemini", "gemini-3.5-flash-lite")
        assert info.title and info.detail and info.action
    assert explain("quota_exhausted", "gemini", "m").switch_model and not explain("quota_exhausted", "gemini", "m").retry_helps
    assert explain("model_unavailable", "gemini", "m").switch_model
    assert not explain("auth", "groq", "m").switch_model and "GROQ_API_KEY" in explain("auth", "groq", "m").action
    assert explain("rate_limited", "gemini", "m").retry_helps
    assert {r for r in REASONS if explain(r).blocking} == BLOCKING


def test_breaker_trips_only_on_errors_that_would_repeat():
    breaker = StopOnBlocking()
    breaker.record("drafting REQ-001: unusable output: empty response (finish reason: STOP)")
    breaker.record(OVERLOADED)
    breaker.record(None)
    assert not breaker.tripped  # one-off and transient errors don't stop a job
    breaker.record(RETIRED_MODEL)
    assert breaker.tripped and breaker.reason == "model_unavailable"
    breaker.skipped = 19
    assert "19 of 21 weren't attempted" in breaker.summary(2, 21, "gemini", "gemini-2.5-flash-lite")
    # The page splits a job's warnings on "; " and drops the stop message, so it must never contain one,
    # and the stored summary must still read back as its reason (so the page can explain it).
    for reason in BLOCKING:
        stop = StopOnBlocking()
        stop.reason, stop.message = reason, f"drafting: Gemini {reason}"
        text = stop.summary(1, 5, "gemini", "gemini-2.5-flash-lite")
        assert "; " not in text
        assert reason_of(text) == reason and explain_message(text, "gemini", "gemini-2.5-flash-lite")["reason"] == reason


def test_provider_health_follows_the_last_call():
    provider_health.clear()

    class Inner:
        model = "gemini-3.5-flash-lite"
        outcome: object = None

        async def draft_answer(self, *args, **kwargs):
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return "ok"

    inner = Inner()
    llm = MonitoredLLM(inner, "gemini")
    assert provider_health.view("gemini", inner.model)["state"] == "unknown"

    inner.outcome = LLMError("malformed", "drafting REQ-001: unusable output: empty response", reason="malformed")
    with pytest.raises(LLMError):
        asyncio.run(llm.draft_answer())
    assert provider_health.view("gemini", inner.model)["state"] == "unknown"  # about one question, not the model

    inner.outcome = QUOTA
    with pytest.raises(LLMError):
        asyncio.run(llm.draft_answer())
    view = provider_health.view("gemini", inner.model)
    assert view["state"] == "failing" and view["explanation"]["reason"] == "quota_exhausted"
    assert view["explanation"]["switch_model"] is True
    assert provider_health.view("gemini", "gemini-3.8-flash")["state"] == "unknown"  # another model is unaffected

    inner.outcome = None
    asyncio.run(llm.draft_answer())
    assert provider_health.view("gemini", inner.model)["state"] == "ok"
    provider_health.clear()


def _failing_after(n_ok, error):
    """A drafter that answers the first n_ok questions, then raises `error` for every call."""
    calls = {"n": 0}

    def drafter(requirement, past_answers, instructions):
        calls["n"] += 1
        return cite_first_past_answer(requirement, past_answers, instructions) if calls["n"] <= n_ok else error

    return drafter


QUESTIONS = [req(f"{QUERY} Part {i}.") for i in range(1, 9)]


def test_drafting_stops_after_a_quota_error_and_explains_it(tmp_path):
    llm = FakeV1LLM(requirements=QUESTIONS)
    ctx = make_context(tmp_path, llm, lessons=FakeLessons(), draft_concurrency=1)

    async def scenario():
        await library_with_competing_answers(ctx, llm)
        llm.drafter = _failing_after(1, QUOTA)
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("rfp"), name="Quota",
                                               client="Ashford Community Bank", industry="finance")
        await ctx.jobs.wait(job.id)
        llm.drafted.clear()
        draft = projects.start_drafting(ctx, project.id)
        await ctx.jobs.wait(draft.id)
        return project.id, draft.id

    project_id, job_id = asyncio.run(scenario())
    assert len(llm.drafted) == 2  # one answer, one quota error, then no more calls
    with ctx.db.session() as session:
        job = session.get(Job, job_id)
        assert job.status == "completed" and job.warning.startswith("Stopped early, quota used up: Gemini quota used up")
        assert "6 of 8 weren't attempted" in job.warning
        assert len(session.scalars(select(DraftRow)).all()) == 2  # the other 6 are left to draft later

    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as client:
            view = client.get(f"/v1/projects/{project_id}").json()
    finally:
        app.state.v1_factory = None
    failed = [r["draft"] for r in view["requirements"] if r["draft"] and r["draft"]["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["error_info"]["reason"] == "quota_exhausted"
    assert failed[0]["error_info"]["switch_model"] is True and failed[0]["error_info"]["retry_helps"] is False
    assert sum(r["draft"] is None for r in view["requirements"]) == 6
    assert view["job"]["warning"].startswith("Stopped early")


def test_one_off_errors_do_not_stop_drafting(tmp_path):
    malformed = LLMError("malformed", "drafting: unusable output: empty response", reason="malformed")
    llm = FakeV1LLM(requirements=QUESTIONS[:4])
    ctx = make_context(tmp_path, llm, lessons=FakeLessons())

    async def scenario():
        await library_with_competing_answers(ctx, llm)
        llm.drafter = _failing_after(0, malformed)
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("rfp"), name="Flaky",
                                               client="Ashford Community Bank", industry="finance")
        await ctx.jobs.wait(job.id)
        llm.drafted.clear()
        draft = projects.start_drafting(ctx, project.id)
        await ctx.jobs.wait(draft.id)
        with ctx.db.session() as session:
            return session.get(Job, draft.id).warning

    warning = asyncio.run(scenario())
    assert len(llm.drafted) == 4 and not (warning or "").startswith("Stopped early")


def test_a_comparison_stops_early_and_can_be_finished_later(tmp_path):
    retired = LLMError("bad_request", "drafting: Gemini model not available: gemini-2.5-flash-lite", reason="model_unavailable")
    llm = FakeV1LLM(requirements=QUESTIONS[:3])
    ctx = make_context(tmp_path, llm, lessons=FakeLessons(), draft_concurrency=1)

    async def scenario():
        await library_with_competing_answers(ctx, llm)
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("rfp"), name="Compare",
                                               client="Ashford Community Bank", industry="finance")
        await ctx.jobs.wait(job.id)
        llm.drafter = _failing_after(2, retired)
        llm.drafted.clear()
        first = experiment.start_comparison(ctx, project.id)
        await ctx.jobs.wait(first.id)
        stopped_calls = len(llm.drafted)
        llm.drafter = cite_first_past_answer  # e.g. the person picked a working model... under the same name
        second = experiment.start_comparison(ctx, project.id)
        await ctx.jobs.wait(second.id)
        return project.id, first.id, stopped_calls

    project_id, first_id, stopped_calls = asyncio.run(scenario())
    assert stopped_calls == 3  # 2 drafts, 1 error, then stopped instead of 3 more failing calls
    with ctx.db.session() as session:
        first = session.get(Job, first_id)
        assert first.status == "failed" and first.error.startswith("Stopped early")
    result = experiment.latest_comparison(ctx, project_id)["comparison"]
    assert result["status"] == "completed" and result["carried_over"] == 2  # the good drafts were kept
    assert len(llm.drafted) == stopped_calls + 4  # only the 4 missing (1 failed + 3 skipped) were drafted
