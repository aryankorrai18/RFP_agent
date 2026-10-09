"""Model usage and storage: every call through ctx.llm is counted with its tokens, and /v1/usage reports it."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from rfp_assistant.api.v1 import usage
from rfp_assistant.main import app
from rfp_assistant.providers.base import LLMError, TokenUsage
from tests.v1_fakes import FakeMemory, make_context


@dataclass
class Reply:
    model: str
    usage: TokenUsage


class CountingLLM:
    model = "fake-model"
    label = "plain attribute"

    async def draft_answer(self) -> Reply:
        return Reply("fake-model-001", TokenUsage(input_tokens=1000, output_tokens=200))

    async def extract_requirements(self) -> Reply:
        return Reply("fake-model-001", TokenUsage(input_tokens=300, output_tokens=50))

    async def broken(self) -> Reply:
        raise LLMError("api_error", "Gemini quota used up")


def run(coro):  # noqa: ANN001, ANN201
    return asyncio.run(coro)


@pytest.fixture
def ctx(tmp_path):  # noqa: ANN001, ANN201
    return make_context(tmp_path, CountingLLM(), FakeMemory())


def test_a_fresh_workspace_has_used_nothing(ctx):
    assert usage.summary(ctx.db) == {"calls": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "by_purpose": {}, "since": None, "last": None}


def test_every_call_is_counted_with_its_tokens_and_what_was_called(ctx):
    run(ctx.llm.draft_answer())
    run(ctx.llm.draft_answer())
    run(ctx.llm.extract_requirements())
    s = usage.summary(ctx.db)
    assert (s["calls"], s["failed"], s["input_tokens"], s["output_tokens"]) == (3, 0, 2300, 450)
    assert s["by_purpose"]["draft_answer"] == {"calls": 2, "failed": 0, "input_tokens": 2000, "output_tokens": 400}
    assert s["by_purpose"]["extract_requirements"]["calls"] == 1 and s["since"] and s["last"]


def test_a_failed_call_is_counted_as_failed_and_still_raises(ctx):
    with pytest.raises(LLMError):
        run(ctx.llm.broken())
    s = usage.summary(ctx.db)
    assert (s["calls"], s["failed"], s["input_tokens"]) == (1, 1, 0)


def test_the_wrapper_is_invisible_to_the_caller(ctx):
    assert ctx.llm.model == "fake-model" and ctx.llm.label == "plain attribute"
    assert run(ctx.llm.draft_answer()).usage.input_tokens == 1000  # the result is returned untouched


def test_if_the_record_cannot_be_written_the_call_still_succeeds(ctx, monkeypatch):
    def broken_session():  # noqa: ANN202
        raise RuntimeError("disk full")

    llm = ctx.llm
    monkeypatch.setattr(ctx.db, "session", broken_session)
    assert run(llm.draft_answer()).usage.output_tokens == 200


def test_the_drafting_pipeline_is_metered_end_to_end(tmp_path):
    from rfp_assistant.api.v1.db import Project, RequirementRow
    from rfp_assistant.api.v1.projects import draft_requirement
    from rfp_assistant.api.v1.storage import store_document
    from rfp_assistant.schemas import DraftResult, Fact
    from tests.v1_fakes import FakeV1LLM

    llm = FakeV1LLM(drafter=lambda req, past, instr: DraftResult(answer="", claims=[], unsupported_claims=[], needs_sme=True, sme_question="?"))
    ctx = make_context(tmp_path, llm, FakeMemory())
    document = store_document(ctx, "rfp", "a.docx", b"a")
    with ctx.db.session() as session:
        project = Project(document_id=document.id, name="P", state="requirements_extracted")
        session.add(project)
        session.flush()
        row = RequirementRow(project_id=project.id, order=1, question="Q?", mandatory=True, word_limit=50)
        session.add(row)
        session.commit()
    run(draft_requirement(ctx, project, row, "Co", [Fact(id="FACT-001", topic="t", statement="s")]))
    s = usage.summary(ctx.db)
    assert s["calls"] == 1 and s["by_purpose"]["draft_answer"]["calls"] == 1


def test_storage_counts_the_database_and_the_uploads(ctx, tmp_path):
    before = usage.storage(ctx)
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "uploads" / "a.pdf").write_bytes(b"x" * 5000)
    (tmp_path / "uploads" / "sub").mkdir()
    (tmp_path / "uploads" / "sub" / "b.txt").write_bytes(b"y" * 700)
    after = usage.storage(ctx)
    assert before["database_bytes"] > 0 and before["uploads_bytes"] == 0
    assert (after["uploads_bytes"], after["uploads_files"]) == (5700, 2)
    assert after["total_bytes"] == after["database_bytes"] + 5700 + after["other_bytes"]


def test_the_usage_route_reports_model_use_and_storage(tmp_path):
    ctx = make_context(tmp_path, CountingLLM(), FakeMemory())
    run(ctx.llm.draft_answer())
    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as client:
            body = client.get("/v1/usage").json()
    finally:
        app.state.v1_factory = None
    assert body["model"]["calls"] == 1 and body["model"]["input_tokens"] == 1000 and body["storage"]["database_bytes"] > 0
    assert set(body["storage"]) == {"database_bytes", "uploads_bytes", "uploads_files", "other_bytes", "total_bytes"}
