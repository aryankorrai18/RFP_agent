"""Stopping background jobs: extraction lands in a retryable state, drafting keeps finished drafts,
and a server shutdown still leaves jobs to resume. Offline, with a model that can be paused."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from backend.core import PipelineError
from backend.v1 import library, projects
from backend.v1.db import DraftRow, Job, PastProposal, Project
from tests.conftest import docx_bytes
from tests.v1_fakes import FakeV1LLM, make_context, pair, req


@dataclass
class PausableLLM(FakeV1LLM):
    """Blocks extraction, or every draft after the first `drafts_before_pause`, until released."""

    block_extraction: bool = False
    drafts_before_pause: int | None = None
    release: asyncio.Event = field(default_factory=asyncio.Event)
    _drafts: int = 0

    async def extract_requirements(self, document):
        if self.block_extraction:
            await self.release.wait()
        return await super().extract_requirements(document)

    async def extract_pairs(self, document):
        if self.block_extraction:
            await self.release.wait()
        return await super().extract_pairs(document)

    async def draft_answer(self, *args, **kwargs):
        self._drafts += 1
        if self.drafts_before_pause is not None and self._drafts > self.drafts_before_pause:
            await self.release.wait()
        return await super().draft_answer(*args, **kwargs)


async def settle():
    for _ in range(20):
        await asyncio.sleep(0)


def test_stopping_extraction_makes_the_project_retryable(tmp_path):
    llm = PausableLLM(requirements=[req("Q1?"), req("Q2?")], block_extraction=True)
    ctx = make_context(tmp_path, llm)

    async def scenario():
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("x"), name="P", client=None, industry=None)
        await settle()
        stopped = ctx.jobs.cancel(job.id)
        await ctx.jobs.wait(job.id)
        with ctx.db.session() as s:
            row = s.get(Project, project.id)
            assert stopped.status == "cancelled" and s.get(Job, job.id).status == "cancelled"
            assert row.state == "failed" and "Stopped" in row.error
        # Retry works exactly as for any failed extraction.
        llm.block_extraction = False
        retry = projects.retry_extraction(ctx, project.id)
        await ctx.jobs.wait(retry.id)
        with ctx.db.session() as s:
            assert s.get(Project, project.id).state == "requirements_extracted"

    asyncio.run(scenario())


def test_stopping_drafting_keeps_finished_drafts_and_the_rest_can_be_drafted(tmp_path):
    llm = PausableLLM(requirements=[req("Q1?"), req("Q2?"), req("Q3?")], drafts_before_pause=1)
    ctx = make_context(tmp_path, llm)

    async def scenario():
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("x"), name="P", client=None, industry=None)
        await ctx.jobs.wait(job.id)
        draft_job = projects.start_drafting(ctx, project.id)
        for _ in range(200):  # until the first draft is saved and the second is paused
            await asyncio.sleep(0)
            with ctx.db.session() as s:
                if s.get(Job, draft_job.id).done >= 1:
                    break
        ctx.jobs.cancel(draft_job.id)
        await ctx.jobs.wait(draft_job.id)
        with ctx.db.session() as s:
            row = s.get(Project, project.id)
            drafted = {d.requirement_id for d in s.query(DraftRow).all()}
            assert row.state == "in_review" and len(drafted) == 1
            missing = [r.id for r in row.requirements if r.id not in drafted]
        assert len(missing) == 2

        llm.release.set()
        rest = projects.start_drafting(ctx, project.id, missing)
        await ctx.jobs.wait(rest.id)
        with ctx.db.session() as s:
            assert {d.requirement_id for d in s.query(DraftRow).all()} == {r.id for r in s.get(Project, project.id).requirements}

    asyncio.run(scenario())


def test_stopping_a_past_proposal_import(tmp_path):
    llm = PausableLLM(pairs=[pair("Q?", "A.")], block_extraction=True)
    ctx = make_context(tmp_path, llm)

    async def scenario():
        proposal, job = library.create_past_proposal(ctx, filename="p.docx", data=docx_bytes("x"), client=None,
                                                     industry=None, submitted_on=None, result="unknown", loss_reason=None)
        await settle()
        ctx.jobs.cancel(job.id)
        await ctx.jobs.wait(job.id)
        with ctx.db.session() as s:
            row = s.get(PastProposal, proposal.id)
            assert row.status == "failed" and "Stopped" in row.error
        assert library.discard_proposal(ctx, proposal.id).status == "discarded"

    asyncio.run(scenario())


def test_finished_jobs_cant_be_stopped_and_shutdown_is_not_a_stop(tmp_path):
    llm = PausableLLM(requirements=[req("Q1?")])
    ctx = make_context(tmp_path, llm)

    async def scenario():
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("x"), name="P", client=None, industry=None)
        await ctx.jobs.wait(job.id)
        with pytest.raises(PipelineError) as error:
            ctx.jobs.cancel(job.id)
        assert error.value.http_status == 409

        # A shutdown cancels the task directly: the job must stay resumable, not "cancelled".
        llm.block_extraction = True
        retry_project, blocked = projects.create_project(ctx, filename="rfp2.docx", data=docx_bytes("y"), name="Q", client=None, industry=None)
        await settle()
        ctx.jobs._tasks[blocked.id].cancel()
        await ctx.jobs.wait(blocked.id)
        with ctx.db.session() as s:
            assert s.get(Job, blocked.id).status == "running"
            assert s.get(Project, retry_project.id).state == "extracting"

    asyncio.run(scenario())
