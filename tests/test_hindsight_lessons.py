"""Hindsight as the learning memory: lessons are built from outcomes and reviews, synced to the
lessons bank, and change which past answer is offered. Offline, with FakeMemory and FakeLessons."""

from __future__ import annotations

import asyncio
from datetime import date

from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.main import app
from backend.v1 import library, projects
from backend.v1.db import Lesson, Pair
from backend.v1.lessons import AnswerLessons, LessonHit, answer_signals, collect_lessons
from backend.v1.retrieval import retrieve
from tests.conftest import docx_bytes
from tests.v1_fakes import FakeLessons, FakeV1LLM, make_context, pair, req

QUERY = "Describe your penetration testing practices."
VAGUE = pair("Describe your security testing and penetration testing practices.", "We perform regular security testing.")
SPECIFIC = pair("Who performs penetration tests?", "Ironbridge Security performs an annual independent penetration test.")


async def import_proposal(ctx, llm, pairs, *, client, industry, submitted_on, result, loss_reason=None, marker="x"):
    llm.pairs = pairs
    proposal, job = library.create_past_proposal(
        ctx, filename=f"{client}.docx", data=docx_bytes(marker, client), client=client, industry=industry,
        submitted_on=submitted_on, result=result, loss_reason=loss_reason,
    )
    await ctx.jobs.wait(job.id)
    with ctx.db.session() as s:
        ids = s.scalars(select(Pair.id).where(Pair.past_proposal_id == proposal.id)).all()
    return library.confirm_pairs(ctx, proposal.id, [library.PairDecision(pair_id=i, decision="kept") for i in ids])


async def library_with_competing_answers(ctx, llm, *, lost_on="2025-02-01", won_on="2025-06-01"):
    [vague] = await import_proposal(ctx, llm, [VAGUE], client="Corvid Logistics", industry="logistics",
                                    submitted_on=date.fromisoformat(lost_on), result="lost",
                                    loss_reason="response quality", marker="a")
    [specific] = await import_proposal(ctx, llm, [SPECIFIC], client="Pinecrest Federal", industry="finance",
                                       submitted_on=date.fromisoformat(won_on), result="won", marker="b")
    await ctx.sync()
    return vague.code, specific.code


def test_proposal_outcomes_become_tagged_lessons(tmp_path):
    lessons = FakeLessons()
    llm = FakeV1LLM()
    ctx = make_context(tmp_path, llm, lessons=lessons)

    async def scenario():
        vague, specific = await library_with_competing_answers(ctx, llm)
        with ctx.db.session() as s:
            rows = {row.key: row for row in s.scalars(select(Lesson))}
        by_answer = {row.answer_id: row for row in rows.values()}
        assert len(rows) == 2 and all(row.hindsight_status == "retained" for row in rows.values())
        lost = next(r for r in rows.values() if f"answer:{vague}" in r.tags)
        won = next(r for r in rows.values() if f"answer:{specific}" in r.tags)
        assert lost.signal == "negative" and "lost on response quality" in lost.text and "February 2025" in lost.text
        assert won.signal == "positive" and "which was won" in won.text
        assert {"client:pinecrest-federal", "industry:finance", "signal:positive"} <= set(won.tags)
        assert len(lessons.items) == 2 and by_answer
        assert collect_lessons(ctx.db) == 0  # idempotent

    asyncio.run(scenario())


def test_lessons_change_which_answer_is_offered(tmp_path):
    lessons = FakeLessons()
    llm = FakeV1LLM()
    ctx = make_context(tmp_path, llm, lessons=lessons)

    async def scenario():
        # Same month, so freshness can't explain the difference: only the lessons can.
        vague, specific = await library_with_competing_answers(ctx, llm, lost_on="2025-06-01", won_on="2025-06-01")
        plain = await retrieve(QUERY, memory=ctx.memory, db=ctx.db, k=2, mode="plain")
        assert [p.id for p in plain.past_answers] == [vague, specific]  # plain search prefers the wording match

        learned = await retrieve(QUERY, memory=ctx.memory, db=ctx.db, k=2, mode="hindsight", lessons=lessons)
        assert [p.id for p in learned.past_answers] == [specific, vague]
        top = learned.retrieved[0]
        assert top["mode"] == "hindsight" and top["lessons"] > 1 and top["lesson_evidence"]
        assert any("Hindsight: 1 positive" in reason for reason in top["reasons"])
        assert learned.warning is None

    asyncio.run(scenario())


def test_active_drafting_can_exclude_all_lost_proposal_answers(tmp_path):
    lessons = FakeLessons()
    llm = FakeV1LLM()
    ctx = make_context(tmp_path, llm, lessons=lessons)

    async def scenario():
        vague, specific = await library_with_competing_answers(ctx, llm)
        found = await retrieve(
            QUERY, memory=ctx.memory, db=ctx.db, k=2, mode="hindsight", lessons=lessons,
            exclude_lost_proposals=True,
        )
        assert [p.id for p in found.past_answers] == [specific]
        assert vague in found.warning
        assert "excluded them from drafting evidence" in found.warning

    asyncio.run(scenario())


def test_hindsight_outage_falls_back_to_local_ranking_with_a_warning(tmp_path):
    lessons = FakeLessons()
    llm = FakeV1LLM()
    ctx = make_context(tmp_path, llm, lessons=lessons)

    async def scenario():
        await library_with_competing_answers(ctx, llm)
        lessons.available = False
        result = await retrieve(QUERY, memory=ctx.memory, db=ctx.db, k=2, mode="hindsight", lessons=lessons)
        assert result.past_answers and result.retrieved[0]["mode"] == "outcome"
        assert "lessons unavailable" in result.warning

    asyncio.run(scenario())


def test_freshness_uses_the_proposal_date_not_the_import_date(tmp_path):
    llm = FakeV1LLM()
    ctx = make_context(tmp_path, llm, lessons=FakeLessons())

    async def scenario():
        vague, specific = await library_with_competing_answers(ctx, llm, lost_on="2023-05-01", won_on="2026-03-01")
        result = await retrieve(QUERY, memory=ctx.memory, db=ctx.db, k=2, mode="plain")
        fresh = {r["id"]: r["freshness"] for r in result.retrieved}
        assert fresh[vague] < 0.5 < fresh[specific]  # imported today, but written in 2023

    asyncio.run(scenario())


def test_a_rejected_review_becomes_a_negative_lesson(tmp_path):
    lessons = FakeLessons()
    llm = FakeV1LLM(requirements=[req(QUERY)])
    ctx = make_context(tmp_path, llm, lessons=lessons)

    async def scenario():
        vague, _specific = await library_with_competing_answers(ctx, llm)
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("rfp"), name="Ashford",
                                               client="Ashford Community Bank", industry="finance")
        await ctx.jobs.wait(job.id)
        draft_job = projects.start_drafting(ctx, project.id)
        await ctx.jobs.wait(draft_job.id)
        with ctx.db.session() as s:
            from backend.v1.db import Project

            requirement = s.get(Project, project.id).requirements[0]
            cited = requirement.drafts[-1].sources
        projects.review(ctx, requirement.id, "rejected", None, reason_tags=["outdated"])
        await ctx.sync()
        with ctx.db.session() as s:
            review_lessons = [r for r in s.scalars(select(Lesson)) if "kind:review" in r.tags]
        assert review_lessons and all(r.signal == "negative" for r in review_lessons)
        assert {f"answer:{code}" for code in cited if code.startswith("ANS-")} <= {t for r in review_lessons for t in r.tags}
        assert all("client:ashford-community-bank" in r.tags for r in review_lessons)

    asyncio.run(scenario())


def test_rejecting_a_facts_only_draft_as_too_vague_is_still_learned(tmp_path):
    """Found live: the drafter answered from the fact sheet and ignored the specific past answers, so
    rejecting it created no lesson at all. Now it becomes a lesson about the question and client,
    and "too vague" becomes a drafting preference for that client."""
    from backend.schemas import DraftClaimOut, DraftResult

    def facts_only(requirement, past, instructions):
        return DraftResult(answer="An independent third party tests the platform every year.",
                           claims=[DraftClaimOut(text="third party tests yearly", source_ids=["FACT-009"])],
                           unsupported_claims=[], needs_sme=False, sme_question=None)

    lessons = FakeLessons()
    llm = FakeV1LLM(requirements=[req(QUERY)])
    ctx = make_context(tmp_path, llm, lessons=lessons)

    async def scenario():
        await library_with_competing_answers(ctx, llm)
        llm.drafter = facts_only
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("rfp"), name="Ashford",
                                               client="Ashford Community Bank", industry="finance")
        await ctx.jobs.wait(job.id)
        await ctx.jobs.wait(projects.start_drafting(ctx, project.id).id)
        with ctx.db.session() as session:
            from backend.v1.db import Project

            requirement = session.get(Project, project.id).requirements[0]
            offered = [r["id"] for r in requirement.drafts[-1].retrieved]
        projects.review(ctx, requirement.id, "rejected", None, reason_tags=["too_vague"])
        await ctx.sync()
        await projects.regenerate(ctx, requirement.id, None)
        return offered

    offered = asyncio.run(scenario())
    with ctx.db.session() as session:
        [lesson] = [r for r in session.scalars(select(Lesson)) if "kind:review" in r.tags]
    assert lesson.signal == "negative" and lesson.answer_id is None
    assert not any(t.startswith("answer:") for t in lesson.tags)  # the unused answers aren't penalised
    assert {"client:ashford-community-bank", "industry:finance", "action:rejected"} <= set(lesson.tags)
    assert "too_vague" in lesson.text and "FACT-009" in lesson.text and all(code in lesson.text for code in offered)
    assert "Be specific" in (llm.drafted[-1][2] or "")  # the next draft for this client is asked for specifics


def test_signals_count_each_lesson_once_and_ignore_other_answers():
    hits = [
        LessonHit("won with Pinecrest", ["signal:positive", "answer:ANS-0002"], "lesson-a", "world", 1),
        LessonHit("same lesson, second fact", ["signal:positive", "answer:ANS-0002"], "lesson-a", "world", 2),
        LessonHit("lost with Corvid", ["signal:negative", "answer:ANS-0001"], "lesson-b", "world", 3),
        LessonHit("unrelated", ["signal:negative", "answer:ANS-0099"], "lesson-c", "world", 4),
    ]
    signals = answer_signals(hits, {"ANS-0001", "ANS-0002"})
    assert signals["ANS-0002"].positive == 1 and signals["ANS-0001"].negative == 1 and "ANS-0099" not in signals
    assert signals["ANS-0002"].factor > 1 > signals["ANS-0001"].factor
    assert AnswerLessons(net=-10).factor == 0.25 and AnswerLessons(net=10).factor == 2.5


def test_brief_and_playbook_endpoints(tmp_path):
    lessons = FakeLessons()
    llm = FakeV1LLM(requirements=[req(QUERY)])
    ctx = make_context(tmp_path, llm, lessons=lessons)

    async def prepare():
        await library_with_competing_answers(ctx, llm)
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("rfp"), name="Ashford",
                                               client="Ashford Community Bank", industry="finance")
        await ctx.jobs.wait(job.id)
        return project.id

    project_id = asyncio.run(prepare())
    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as client:
            assert client.get(f"/v1/projects/{project_id}/brief").json()["brief"] is None  # nothing spent yet
            made = client.post(f"/v1/projects/{project_id}/brief").json()["brief"]
            assert made["text"] and made["based_on"]
            assert lessons.reflect_calls[-1][1] == ["client:ashford-community-bank", "industry:finance"]
            assert client.get(f"/v1/projects/{project_id}/brief").json()["brief"]["text"] == made["text"]  # cached

            assert client.get("/v1/memory/playbook").json()["playbook"] is None
            refreshed = client.post("/v1/memory/playbook/refresh").json()["playbook"]
            assert refreshed["content"].startswith("Playbook from 2 lessons")

            listing = client.get("/v1/memory/lessons").json()
            assert listing["total"] == 2 and listing["pending"] == 0
            status = client.get("/v1/status").json()
            assert status["lessons"]["enabled"] is True and status["retrieval_mode"] == "hindsight"
    finally:
        app.state.v1_factory = None
