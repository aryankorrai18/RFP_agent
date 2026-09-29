"""The learning curve is counted from real reviews, and the before/after comparison changes only the
memory state. Offline, with FakeMemory and FakeLessons."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.main import app
from backend.core import PipelineError
from backend.v1 import experiment, projects
from backend.v1.curve import WEIGHTS, learning_curve, review_bucket
from backend.v1.db import ComparisonDraft
from tests.conftest import docx_bytes
from tests.test_hindsight_lessons import QUERY, library_with_competing_answers
from tests.v1_fakes import FakeLessons, FakeV1LLM, make_context, req


async def new_project(ctx, name):
    project, job = projects.create_project(ctx, filename=f"{name}.docx", data=docx_bytes(name), name=name,
                                           client="Ashford Community Bank", industry="finance")
    await ctx.jobs.wait(job.id)
    return project.id


async def draft_project(ctx, project_id):
    job = projects.start_drafting(ctx, project_id)
    await ctx.jobs.wait(job.id)
    with ctx.db.session() as session:
        from backend.v1.db import RequirementRow

        return session.scalars(select(RequirementRow.id).where(RequirementRow.project_id == project_id)).all()


def test_review_buckets_follow_the_documented_thresholds():
    assert review_bucket("accepted", None) == "accepted"
    assert review_bucket("edited", 0.25) == "light_edit" and review_bucket("edited", 0.26) == "heavy_edit"
    assert review_bucket("rewritten", 0.9) == "heavy_edit" and review_bucket("rejected", None) == "rejected"


def test_curve_counts_real_reviews_project_by_project(tmp_path):
    llm = FakeV1LLM(requirements=[req(QUERY), req("Do you support SSO?")])
    ctx = make_context(tmp_path, llm, lessons=FakeLessons())

    async def scenario():
        await library_with_competing_answers(ctx, llm)
        first = await new_project(ctx, "first")
        first_reqs = await draft_project(ctx, first)
        projects.review(ctx, first_reqs[0], "rejected", None)
        projects.review(ctx, first_reqs[1], "rewritten", "A completely different answer written by hand.")
        second = await new_project(ctx, "second")
        second_reqs = await draft_project(ctx, second)
        projects.review(ctx, second_reqs[0], "accepted", None)
        with ctx.db.session() as session:
            draft = session.scalars(select(ComparisonDraft)).all()
            assert draft == []  # nothing from the comparison table leaks in
            return learning_curve(session)

    curve = asyncio.run(scenario())
    first, second = curve["projects"]
    assert (first["name"], first["reviewed"], first["rejected"], first["heavy_edit"], first["accepted"]) == ("first", 2, 1, 1, 0)
    assert first["score"] == round((WEIGHTS["rejected"] + WEIGHTS["heavy_edit"]) / 2, 3) and first["avg_edit_distance"] > 0.5
    assert (second["reviewed"], second["accepted"], second["score"], second["accepted_rate"]) == (1, 1, 1.0, 1.0)  # 1 of 2 reviewed
    assert second["drafted"] == 2 and curve["score_change"] == round(1.0 - first["score"], 3)
    assert first["retrieval_modes"] == ["hindsight"] and curve["mixed_conditions"] is False and curve["caveat"]


def test_only_the_latest_reviewed_draft_of_a_question_counts(tmp_path):
    llm = FakeV1LLM(requirements=[req(QUERY)])
    ctx = make_context(tmp_path, llm, lessons=FakeLessons())

    async def scenario():
        await library_with_competing_answers(ctx, llm)
        project = await new_project(ctx, "one")
        [requirement] = await draft_project(ctx, project)
        projects.review(ctx, requirement, "rejected", None)
        await projects.regenerate(ctx, requirement, "be specific")
        projects.review(ctx, requirement, "accepted", None)
        with ctx.db.session() as session:
            return learning_curve(session)["projects"][0]

    point = asyncio.run(scenario())
    assert (point["reviewed"], point["accepted"], point["rejected"]) == (1, 1, 0)


def test_before_after_changes_only_the_memory_state(tmp_path):
    lessons = FakeLessons()
    llm = FakeV1LLM(requirements=[req(QUERY)])
    ctx = make_context(tmp_path, llm, lessons=lessons)

    async def scenario():
        # Same month, so freshness can't explain a difference: only the lessons can.
        vague, specific = await library_with_competing_answers(ctx, llm, lost_on="2025-06-01", won_on="2025-06-01")
        project = await new_project(ctx, "compare")
        llm.drafted.clear()
        llm.fact_sets.clear()
        job = experiment.start_comparison(ctx, project)
        await ctx.jobs.wait(job.id)
        return vague, specific, project

    vague, specific, project = asyncio.run(scenario())
    # Same model call for both arms: same facts, no reviewer instructions; only the offered answers differ.
    assert len(llm.drafted) == 2 and llm.fact_sets[0] == llm.fact_sets[1]
    assert all(instructions is None for _id, _offered, instructions in llm.drafted)
    offered_first = {arm_offer[0] for _r, arm_offer, _i in llm.drafted}
    assert offered_first == {vague, specific}  # plain leads with the wording match, hindsight with the won answer

    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as client:
            result = client.get(f"/v1/projects/{project}/comparison").json()["comparison"]
    finally:
        app.state.v1_factory = None
    assert result["status"] == "completed" and result["arms"] == ["plain", "hindsight"] and result["controls"]
    [question] = result["questions"]
    plain, learned = question["arms"]["plain"], question["arms"]["hindsight"]
    assert plain["cited_answers"] == [vague] and learned["cited_answers"] == [specific] and question["changed"] is True
    assert result["questions_changed"] == 1
    assert plain["cited_history"] == {"won": 0, "lost": 1} and learned["cited_history"] == {"won": 1, "lost": 0}
    assert result["summary"]["plain"]["cited_answers_from_lost_proposals"] == 1
    assert result["summary"]["hindsight"]["cited_answers_from_won_proposals"] == 1
    assert learned["retrieved"][0]["lesson_evidence"] and learned["retrieved"][0]["lessons"] > 1
    assert plain["retrieved"][0]["lesson_evidence"] == []

    with ctx.db.session() as session:
        from backend.v1.db import DraftRow

        assert session.scalars(select(DraftRow)).all() == []  # comparison drafts are never review drafts
        assert len(session.scalars(select(ComparisonDraft)).all()) == 2


def _stop_partway(ctx, job_id, drop):
    """Make a finished comparison look like one the user stopped with `drop` drafts still missing."""
    from backend.v1.db import Job

    with ctx.db.session() as session:
        rows = session.scalars(select(ComparisonDraft).where(ComparisonDraft.job_id == job_id)).all()
        for row in rows[:drop]:
            session.delete(row)
        session.get(Job, job_id).status = "cancelled"
        session.commit()


def _comparison(ctx, project):
    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as client:
            return client.get(f"/v1/projects/{project}/comparison").json()["comparison"]
    finally:
        app.state.v1_factory = None


def test_a_stopped_comparison_is_finished_not_restarted(tmp_path):
    llm = FakeV1LLM(requirements=[req(QUERY), req("Do you support SSO?")])
    ctx = make_context(tmp_path, llm, lessons=FakeLessons())

    async def scenario():
        await library_with_competing_answers(ctx, llm)
        project = await new_project(ctx, "resume")
        first = experiment.start_comparison(ctx, project)
        await ctx.jobs.wait(first.id)
        _stop_partway(ctx, first.id, drop=1)  # 3 of 4 drafts made before the stop
        calls = len(llm.drafted)
        second = experiment.start_comparison(ctx, project)
        await ctx.jobs.wait(second.id)
        return project, first.id, calls

    project, first_id, calls = asyncio.run(scenario())
    assert calls == 4 and len(llm.drafted) == 5  # only the missing draft cost a model call
    assert llm.temperatures[-1] == experiment.TEMPERATURE
    result = _comparison(ctx, project)
    assert result["status"] == "completed" and result["done"] == result["total"] == 4
    assert (result["carried_over"], result["carried_over_from"]) == (3, first_id)
    assert all(q["arms"].get("plain") and q["arms"].get("hindsight") for q in result["questions"])


def test_running_again_after_a_finished_comparison_drafts_everything_fresh(tmp_path):
    llm = FakeV1LLM(requirements=[req(QUERY), req("Do you support SSO?")])
    ctx = make_context(tmp_path, llm, lessons=FakeLessons())

    async def scenario():
        await library_with_competing_answers(ctx, llm)
        project = await new_project(ctx, "again")
        for _ in range(2):
            job = experiment.start_comparison(ctx, project)
            await ctx.jobs.wait(job.id)
        return project

    project = asyncio.run(scenario())
    assert len(llm.drafted) == 8  # memory may have changed between runs, so nothing is reused
    assert _comparison(ctx, project)["carried_over"] == 0


def test_a_stopped_run_under_other_conditions_is_not_reused(tmp_path):
    from backend.v1.db import Job

    llm = FakeV1LLM(requirements=[req(QUERY), req("Do you support SSO?")])
    ctx = make_context(tmp_path, llm, lessons=FakeLessons())

    async def scenario():
        await library_with_competing_answers(ctx, llm)
        project = await new_project(ctx, "conditions")
        counts = []
        for payload_change in ({"model": "another-model"}, {"temperature": None}, "legacy"):
            job = experiment.start_comparison(ctx, project)
            await ctx.jobs.wait(job.id)
            _stop_partway(ctx, job.id, drop=1)
            with ctx.db.session() as session:
                row = session.get(Job, job.id)
                row.payload = {"arms": row.payload["arms"]} if payload_change == "legacy" else {**row.payload, **payload_change}
                session.commit()
            before = len(llm.drafted)
            nxt = experiment.start_comparison(ctx, project)
            await ctx.jobs.wait(nxt.id)
            counts.append(len(llm.drafted) - before)
            _stop_partway(ctx, nxt.id, drop=0)  # keep the next round's "previous run" a stopped one
        return counts

    # Different model, a different temperature, or a run from before conditions were recorded:
    # all 4 drafts are made again rather than mixing conditions.
    assert asyncio.run(scenario()) == [4, 4, 4]


def test_comparison_input_checks(tmp_path):
    llm = FakeV1LLM(requirements=[req(QUERY)])
    ctx = make_context(tmp_path, llm, lessons=None)

    async def scenario():
        project = await new_project(ctx, "checks")
        for arms in (["plain"], ["plain", "bogus"]):
            with pytest.raises(PipelineError) as caught:
                experiment.start_comparison(ctx, project, arms)
            assert caught.value.http_status == 422
        with pytest.raises(PipelineError) as caught:
            experiment.start_comparison(ctx, project)  # lessons are off, so "hindsight" can't run
        assert "switched off" in caught.value.message
        with pytest.raises(PipelineError):
            experiment.start_comparison(ctx, 999, ["plain", "outcome"])

    asyncio.run(scenario())
