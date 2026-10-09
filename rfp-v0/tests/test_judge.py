"""A judge model and a person compare the before/after drafts blind. Offline, with
a scripted judge."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from rfp_assistant.providers.base import LLMError
from rfp_assistant.main import app
from rfp_assistant.errors import PipelineError
from rfp_assistant.api.v1 import experiment, judge, projects
from rfp_assistant.api.v1.db import Job, JudgeVerdict
from tests.conftest import docx_bytes
from tests.test_hindsight_lessons import QUERY, library_with_competing_answers
from tests.v1_fakes import FakeLessons, FakeV1LLM, cite_first_past_answer, draft_text, make_context, req, verdict

SSO = "Do you support single sign-on?"  # nothing in the library: both arms need an expert, identically


def sso_needs_an_expert(requirement, past, instructions):
    if requirement.question == SSO:
        return cite_first_past_answer(requirement, [], instructions)  # the same "ask an expert" in both arms
    return cite_first_past_answer(requirement, past, instructions)


def compared(tmp_path, **llm_kwargs):
    """A project whose before/after comparison is done: the pen-test question differs between arms
    (plain offers the vague lost answer, lessons the specific won one), the SSO question doesn't."""
    llm = FakeV1LLM(requirements=[req(QUERY), req(SSO)], drafter=sso_needs_an_expert, **llm_kwargs)
    ctx = make_context(tmp_path, llm, lessons=FakeLessons(), model="gemini-3.5-flash-lite")

    async def scenario():
        await library_with_competing_answers(ctx, llm, lost_on="2025-06-01", won_on="2025-06-01")
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("rfp"), name="Judge",
                                               client="Ashford Community Bank", industry="finance")
        await ctx.jobs.wait(job.id)
        comparison = experiment.start_comparison(ctx, project.id)
        await ctx.jobs.wait(comparison.id)
        return project.id

    return ctx, llm, asyncio.run(scenario())


def run_judge(ctx, project_id, **kwargs):
    async def go():
        job = judge.start_judging(ctx, project_id, **kwargs)
        await ctx.jobs.wait(job.id)
        return job.id

    return asyncio.run(go())


def test_plan_costs_only_differing_pairs_in_both_orders(tmp_path):
    ctx, _llm, project_id = compared(tmp_path)
    plan = judge.plan(ctx, project_id, "gemini-3.8-flash")
    assert (plan["pairs"], plan["differing"], plan["identical"], plan["calls_needed"]) == (2, 1, 1, 2)
    assert plan["same_model_as_drafter"] is False
    assert judge.plan(ctx, project_id, "gemini-3.5-flash-lite")["same_model_as_drafter"] is True
    assert judge.plan(ctx, project_id, "gemini-3.8-flash", orders=1)["calls_needed"] == 1


def test_judge_is_blind_uses_both_orders_and_unblinds(tmp_path):
    ctx, llm, project_id = compared(tmp_path)
    run_judge(ctx, project_id, judge_model="gemini-3.8-flash")

    assert len(llm.judged) == 2  # one differing pair x two orders; the identical pair cost nothing
    firsts = sorted("Ironbridge" in draft_text(m, "A") for m in llm.judged)
    assert firsts == [False, True]  # each arm was shown first once
    for message in llm.judged:
        # The judge sees the evidence but not who won or lost, nor which arm is which.
        assert "<fact_sheet" in message and "<past_answers>" in message and "Ironbridge" in message
        for leak in ("Pinecrest", "Corvid", "won", "lost", "hindsight", "lessons", "plain"):
            assert leak not in message, leak

    result = judge.results(ctx, project_id)["judge"]
    assert result["judge_model"] == "gemini-3.8-flash" and result["served_models"] == ["fake-model-judge"]
    assert result["tally"] == {"hindsight": 1, "plain": 0, "tie": 0, "depends_on_order": 0, "identical": 1, "not_judged": 0}
    assert result["order_consistency"] == {"both_orders": 1, "agreed": 1}
    assert result["mean_scores"]["hindsight"]["accurate"] == 5 and result["mean_scores"]["plain"]["accurate"] == 3
    assert result["tokens"] == {"input": 200, "output": 40}
    with ctx.db.session() as session:
        rows = session.scalars(select(JudgeVerdict)).all()
        assert {(r.order, r.winner, r.prompt_version) for r in rows} == {
            ("plain_first", "hindsight", judge.JUDGE_PROMPT_VERSION), ("hindsight_first", "hindsight", judge.JUDGE_PROMPT_VERSION)}


def test_a_judge_that_favours_position_counts_as_a_tie(tmp_path):
    ctx, _llm, project_id = compared(tmp_path, judger=lambda message: verdict("A"))
    run_judge(ctx, project_id, judge_model="gemini-3.8-flash")
    result = judge.results(ctx, project_id)["judge"]
    assert result["tally"]["depends_on_order"] == 1 and result["tally"]["hindsight"] == 0 and result["tally"]["plain"] == 0
    assert result["order_consistency"] == {"both_orders": 1, "agreed": 0}


def test_nothing_is_judged_twice_and_a_stopped_run_is_finished(tmp_path):
    quota = LLMError("api_error", "judging a draft pair: Gemini quota used up for gemini-3.8-flash (daily)", reason="quota_exhausted")
    calls = {"n": 0}

    def flaky(message):
        calls["n"] += 1
        return verdict("B") if calls["n"] == 1 else quota

    ctx, llm, project_id = compared(tmp_path, judger=flaky)
    first = run_judge(ctx, project_id, judge_model="gemini-3.8-flash", orders=2)
    with ctx.db.session() as session:
        job = session.get(Job, first)
        assert job.status == "failed" and job.error.startswith("Stopped early, quota used up: Gemini quota used up")
        assert "judge panel" in job.error and "Provider said: judging a draft pair" in job.error
        assert "Redraft" not in job.error  # the judge has no Redraft button
    assert judge.plan(ctx, project_id, "gemini-3.8-flash")["calls_needed"] == 1  # the good verdict was kept
    partial = judge.results(ctx, project_id, "gemini-3.8-flash")["judge"]
    assert partial["tally"]["not_judged"] == 1 and partial["tally"]["plain"] == 0  # one order of two isn't a result

    llm.judger = None  # quota back: finish the run
    run_judge(ctx, project_id, judge_model="gemini-3.8-flash")
    assert len(llm.judged) == 3 and judge.plan(ctx, project_id, "gemini-3.8-flash")["calls_needed"] == 0
    run_judge(ctx, project_id, judge_model="gemini-3.8-flash")
    assert len(llm.judged) == 3  # already judged under the same model and prompt: no new calls
    run_judge(ctx, project_id, judge_model="gemini-3.6-flash", orders=1)
    assert len(llm.judged) == 4  # a different judge model is judged separately


def test_spot_check_hides_the_arms_until_answered_and_measures_agreement(tmp_path):
    ctx, _llm, project_id = compared(tmp_path)
    run_judge(ctx, project_id, judge_model="gemini-3.8-flash")

    check = judge.spot_check(ctx, project_id)["spot_check"]
    assert check["total"] == 1 and check["answered"] == 0  # only the differing pair is shown
    [item] = check["items"]
    assert "reveal" not in item and set(item) >= {"A", "B", "question"}
    specific = "A" if "Ironbridge" in item["A"]["answer"] else "B"

    after = judge.record_spot_check(ctx, project_id, item["requirement_id"], specific, "names the tester")["spot_check"]
    reveal = after["items"][0]["reveal"]
    assert reveal[specific] == "hindsight" and reveal["your_choice"] == "hindsight" and reveal["note"] == "names the tester"
    assert judge.spot_check(ctx, project_id)["spot_check"]["items"][0]["A"] == item["A"]  # the order is stable
    result = judge.results(ctx, project_id)["judge"]
    assert result["human_agreement"] == {"compared": 1, "agreed": 1}

    with pytest.raises(PipelineError):
        judge.record_spot_check(ctx, project_id, item["requirement_id"], "C", None)


def test_judge_api_reports_the_cost_before_spending(tmp_path):
    ctx, llm, project_id = compared(tmp_path)
    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as client:
            before = client.get(f"/v1/projects/{project_id}/comparison/judge?judge_model=gemini-3.8-flash").json()
            assert before["plan"]["calls_needed"] == 2 and before["judge"]["tally"]["not_judged"] == 1
            assert llm.judged == []  # reading the plan costs nothing
            default = client.get(f"/v1/projects/{project_id}/comparison/judge").json()
            assert default["plan"]["judge_model"] == default["judge"]["judge_model"]  # one model, described consistently

            bad = client.post(f"/v1/projects/{project_id}/comparison/judge", json={"judge_model": "Not A Model!"})
            assert bad.status_code == 422
            started = client.post(f"/v1/projects/{project_id}/comparison/judge", json={"judge_model": "gemini-3.8-flash"}).json()
            assert started["planned_calls"] == 2
            spot = client.get(f"/v1/projects/{project_id}/comparison/spot-check").json()["spot_check"]
            assert spot["total"] == 1
            wrong = client.post(f"/v1/projects/{project_id}/comparison/spot-check",
                                json={"requirement_id": spot["items"][0]["requirement_id"], "choice": "maybe"})
            assert wrong.status_code == 422
    finally:
        app.state.v1_factory = None


def test_judging_needs_a_finished_comparison(tmp_path):
    llm = FakeV1LLM(requirements=[req(QUERY)])
    ctx = make_context(tmp_path, llm, lessons=FakeLessons())

    async def scenario():
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("rfp"), name="None yet",
                                               client=None, industry=None)
        await ctx.jobs.wait(job.id)
        return project.id

    project_id = asyncio.run(scenario())
    assert judge.plan(ctx, project_id)["calls_needed"] == 0
    with pytest.raises(PipelineError) as caught:
        judge.start_judging(ctx, project_id)
    assert caught.value.http_status == 409
