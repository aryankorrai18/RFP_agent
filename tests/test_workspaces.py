"""Workspaces: the same engine with different starting memory. The demo seed needs no model calls;
a company starting from scratch sets up its facts in the app. Offline."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import rfp_assistant.main as main
from rfp_assistant import workspaces
from rfp_assistant.config import ROOT, Settings
from rfp_assistant.providers.base import LLMError
from rfp_assistant.errors import PipelineError
from rfp_assistant.api.v1 import demo, experiment, library, projects
from rfp_assistant.api.v1.context import V1Context
from rfp_assistant.api.v1.db import AnswerStats, ComparisonDraft, Database, Lesson, Pair
from tests.conftest import docx_bytes
from tests.test_hindsight_lessons import QUERY, library_with_competing_answers
from tests.v1_fakes import FakeLessons, FakeMemory, FakeV1LLM, make_context, req

NO_MODEL = LLMError("auth", "No model calls expected in this test", reason="auth")


def base(tmp_path) -> Settings:
    return replace(Settings(), db_path=tmp_path / "main" / "rfp.db", uploads_dir=tmp_path / "main" / "uploads",
                   fact_sheet_path=ROOT / "data" / "fact_sheet.json")


def test_without_a_registry_settings_are_untouched(tmp_path):
    settings = base(tmp_path)
    assert workspaces.apply_workspace(settings) == settings and workspaces.active() is None


def test_registry_keeps_existing_data_as_main_and_isolates_new_workspaces(tmp_path):
    settings = base(tmp_path)
    main_space = workspaces.ensure_registry(settings)
    assert (main_space.id, main_space.kind, main_space.fact_sheet_locked) == ("main", "main", True)
    assert workspaces.apply_workspace(settings).db_path == settings.db_path  # nothing moved

    acme = workspaces.create("Acme Corp", "company", settings)
    again = workspaces.create("Acme Corp", "company", settings)
    assert (acme.id, again.id) == ("acme-corp", "acme-corp-2")
    assert (acme.bank, acme.lessons_bank) == ("rfp-library-acme-corp", "rfp-lessons-acme-corp")
    assert workspaces.active().id == "main"  # creating doesn't switch

    workspaces.activate("acme-corp")
    applied = workspaces.apply_workspace(settings)
    assert applied.db_path.parent.name == "acme-corp" and applied.db_path.parent.exists()
    assert applied.fact_sheet_path.name == "fact_sheet.json" and not applied.fact_sheet_path.exists()
    assert applied.hindsight_bank == "rfp-library-acme-corp"
    with pytest.raises(ValueError):
        workspaces.create("  ", "company", settings)
    with pytest.raises(ValueError):
        workspaces.create("X", "other", settings)


def test_remove_protects_main_and_the_active_workspace_and_deletes_its_files(tmp_path):
    settings = base(tmp_path)
    workspaces.ensure_registry(settings)
    acme = workspaces.create("Acme Corp", "company", settings)
    folder = workspaces._abs(acme.db_path).parent
    assert folder.is_dir()

    with pytest.raises(ValueError, match="original workspace"):
        workspaces.remove("main")
    workspaces.activate("acme-corp")
    with pytest.raises(ValueError, match="Switch to another"):
        workspaces.remove("acme-corp")
    with pytest.raises(KeyError):
        workspaces.remove("nope")

    workspaces.activate("main")  # switch away, then removal is allowed
    removed = workspaces.remove("acme-corp")
    assert removed.id == "acme-corp" and not folder.exists()
    assert workspaces.get("acme-corp") is None
    assert [w.id for w in workspaces.load()[1]] == ["main"]


def test_prepared_import_makes_no_model_call_and_credits_outcomes(tmp_path):
    llm = FakeV1LLM(fail_pairs=NO_MODEL)
    ctx = make_context(tmp_path, llm)
    pairs = [{"section": "Security", "reference": "Q1", "question": "Who tests?", "answer": "Ironbridge Security."},
             {"section": "Security", "reference": "Q2", "question": "Blank?", "answer": "  "}]
    created = library.import_prepared_proposal(
        ctx, filename="won.docx", data=docx_bytes("won"), client="Pinecrest", industry="finance",
        submitted_on=None, result="won", loss_reason=None, pairs=pairs,
    )
    assert len(created) == 1  # the blank answer is skipped
    with ctx.db.session() as session:
        assert session.scalars(select(Pair)).one().decision == "kept"
        assert session.get(AnswerStats, created[0].id).outcome_credit > 0
    with pytest.raises(PipelineError) as caught:
        library.import_prepared_proposal(ctx, filename="won.docx", data=docx_bytes("won"), client=None, industry=None,
                                         submitted_on=None, result="won", loss_reason=None, pairs=pairs)
    assert caught.value.code == "already_imported"


def test_demo_seed_loads_the_whole_history_without_model_calls(tmp_path):
    llm = FakeV1LLM(fail_pairs=NO_MODEL, fail_requirements=NO_MODEL)
    lessons = FakeLessons()
    ctx = make_context(tmp_path, llm, lessons=lessons, fact_sheet_path=tmp_path / "facts" / "fact_sheet.json")

    report = demo.seed_demo(ctx)
    assert (report["proposals"], report["answers"], report["already_there"]) == (8, 45, 0)
    assert ctx.settings.fact_sheet_path.exists()  # the demo company's facts, as the workspace's own copy
    assert json.loads(ctx.settings.fact_sheet_path.read_text(encoding="utf-8"))["company"]
    assert {r["file"] for r in report["sample_rfps"]} >= {"2026-12_ashford_community_bank_rfp.docx"}

    asyncio.run(ctx.sync())
    with ctx.db.session() as session:
        signals = {row.signal for row in session.scalars(select(Lesson))}
        # Lost on price or to the incumbent says nothing about the answers: those lessons are neutral.
        assert len(session.scalars(select(Lesson)).all()) == 45 and signals == {"positive", "negative", "neutral"}
    assert demo.seed_demo(ctx)["already_there"] == 8  # running it again adds nothing


def test_virtusa_cyber_pack_loads_prepared_answers_without_model_calls(tmp_path):
    llm = FakeV1LLM(fail_pairs=NO_MODEL, fail_requirements=NO_MODEL)
    ctx = make_context(tmp_path, llm)

    report = demo.seed_virtusa_cyber(ctx)
    assert (report["proposals"], report["answers"], report["already_there"]) == (1, 10, 0)
    assert report["uses_model_calls"] is False
    with ctx.db.session() as session:
        pairs = session.scalars(select(Pair)).all()
        assert len(pairs) == 10
        assert all(pair.decision == "kept" for pair in pairs)
        assert all("End of synthetic" not in pair.answer for pair in pairs)
    again = demo.seed_virtusa_cyber(ctx)
    assert (again["proposals"], again["answers"], again["already_there"]) == (0, 0, 1)


def test_company_facts_are_set_up_in_the_app(tmp_path):
    ctx = make_context(tmp_path, FakeV1LLM(), fact_sheet_path=tmp_path / "company" / "fact_sheet.json")
    main.app.state.v1_factory = lambda: ctx
    try:
        with TestClient(main.app) as client:
            assert client.get("/v1/company").json() == {"company": None, "facts": [], "locked": False, "set_up": False}
            assert client.get("/v1/status").json()["workspace"]["company_set_up"] is False
            saved = client.put("/v1/company", json={"company": "Acme Corp", "facts": [
                {"topic": "SSO", "statement": "Acme supports SAML 2.0 single sign-on."},
                {"id": "FACT-007", "topic": "Hosting", "statement": "Data is hosted in the EU."},
                {"topic": "Blank", "statement": "   "},
            ]}).json()
            assert [f["id"] for f in saved["facts"]] == ["FACT-008", "FACT-007"]  # given IDs kept, new ones after them
            assert client.get("/v1/company").json()["company"] == "Acme Corp"
            assert client.get("/v1/status").json()["workspace"]["company_set_up"] is True
            assert client.put("/v1/company", json={"company": "Acme", "facts": []}).status_code == 422
    finally:
        main.app.state.v1_factory = None


def test_the_bundled_sample_fact_sheet_is_never_edited(tmp_path):
    bundled = ROOT / "data" / "fact_sheet.json"
    before = bundled.read_bytes()
    ctx = make_context(tmp_path, FakeV1LLM(), fact_sheet_path=bundled)
    main.app.state.v1_factory = lambda: ctx
    try:
        with TestClient(main.app) as client:
            assert client.get("/v1/company").json()["locked"] is True
            refused = client.put("/v1/company", json={"company": "X", "facts": [{"statement": "y"}]})
            assert refused.status_code == 409
    finally:
        main.app.state.v1_factory = None
    assert bundled.read_bytes() == before


def test_switching_workspaces_opens_that_workspaces_memory(tmp_path, monkeypatch):
    settings = base(tmp_path)
    workspaces.ensure_registry(settings)
    llm = FakeV1LLM(fail_pairs=NO_MODEL)
    built = []

    def factory():
        s = workspaces.apply_workspace(settings)
        ctx = V1Context(db=Database(s.db_path), memory=FakeMemory(), lessons=FakeLessons(),
                        settings_provider=lambda: s, llm_provider=lambda _s: llm)
        built.append(ctx)
        return ctx

    monkeypatch.setattr(main, "base_settings", lambda: settings)
    main.app.state.v1_factory = factory
    try:
        with TestClient(main.app) as client:
            listed = client.get("/v1/workspaces").json()
            assert listed["active"] == "main" and [w["id"] for w in listed["workspaces"]] == ["main"]

            made = client.post("/v1/workspaces", json={"name": "Acme Corp", "kind": "company"}).json()
            assert made["workspace"]["id"] == "acme-corp" and made["seeded"] is None
            assert main.app.state.v1 is built[-1] and main.app.state.v1.settings.db_path.parent.name == "acme-corp"
            status = client.get("/v1/status").json()["workspace"]
            assert (status["id"], status["company_set_up"], status["projects"]) == ("acme-corp", False, 0)

            demo_made = client.post("/v1/workspaces", json={"kind": "demo"}).json()
            assert demo_made["workspace"]["name"] == "Larkspur Data (demo)" and demo_made["seeded"]["answers"] == 45
            assert client.get("/v1/status").json()["library"]["answers"] == 45

            monkeypatch.setattr(main.app.state.v1.jobs, "busy", lambda: True)
            busy = client.post("/v1/workspaces/main/activate")
            assert busy.status_code == 409 and "still running" in busy.json()["error"]["message"]
            monkeypatch.setattr(main.app.state.v1.jobs, "busy", lambda: False)
            back = client.post("/v1/workspaces/main/activate").json()
            assert back["active"] == "main" and main.app.state.v1.settings.db_path == settings.db_path
            assert client.post("/v1/workspaces/nope/activate").status_code == 404
            assert client.post("/v1/workspaces", json={"name": "", "kind": "company"}).status_code == 422
    finally:
        main.app.state.v1_factory = None


def test_deleting_a_workspace_over_http_removes_it_and_tries_its_hindsight_banks(tmp_path, monkeypatch):
    settings = base(tmp_path)
    workspaces.ensure_registry(settings)
    llm = FakeV1LLM(fail_pairs=NO_MODEL)

    def factory():
        s = workspaces.apply_workspace(settings)
        return V1Context(db=Database(s.db_path), memory=FakeMemory(), lessons=FakeLessons(),
                         settings_provider=lambda: s, llm_provider=lambda _s: llm)

    monkeypatch.setattr(main, "base_settings", lambda: settings)
    deleted_banks = []

    async def fake_delete_banks(hs_settings, space):
        deleted_banks.append((space.id, hs_settings.hindsight_bank, space.bank, space.lessons_bank))

    monkeypatch.setattr(main, "_delete_hindsight_banks", fake_delete_banks)
    main.app.state.v1_factory = factory
    try:
        with TestClient(main.app) as client:
            client.post("/v1/workspaces", json={"name": "Acme Corp", "kind": "company"})
            client.post("/v1/workspaces/main/activate")  # switch away so acme-corp can be removed

            refused = client.delete("/v1/workspaces/main")
            assert refused.status_code == 422 and "original workspace" in refused.json()["error"]["message"]
            assert client.delete("/v1/workspaces/nope").status_code == 404

            gone = client.delete("/v1/workspaces/acme-corp").json()
            assert gone["active"] == "main" and [w["id"] for w in gone["workspaces"]] == ["main"]
            assert deleted_banks == [("acme-corp", settings.hindsight_bank, "rfp-library-acme-corp", "rfp-lessons-acme-corp")]
            assert client.get("/v1/workspaces").json()["workspaces"] == [{
                "id": "main", "name": "Larkspur Data (sample history)", "kind": "main",
                "created_at": workspaces.get("main").created_at, "fact_sheet_locked": True, "active": True,
            }]
    finally:
        main.app.state.v1_factory = None


def test_editing_requirements_after_a_comparison_clears_it(tmp_path):
    llm = FakeV1LLM(requirements=[req(QUERY), req("Do you support SSO?")])
    ctx = make_context(tmp_path, llm, lessons=FakeLessons())

    async def scenario():
        await library_with_competing_answers(ctx, llm)
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("rfp"), name="Edit",
                                               client="Ashford Community Bank", industry="finance")
        await ctx.jobs.wait(job.id)
        comparison = experiment.start_comparison(ctx, project.id)
        await ctx.jobs.wait(comparison.id)
        projects.replace_requirements(ctx, project.id, [projects.RequirementEdit(question="Only this one?")])
        return project.id

    project_id = asyncio.run(scenario())
    with ctx.db.session() as session:
        assert session.scalars(select(ComparisonDraft)).all() == []
    assert experiment.latest_comparison(ctx, project_id)["comparison"]["questions"][0]["question"] == "Only this one?"


def test_too_vague_is_a_review_reason(tmp_path):
    from rfp_assistant.api.v1.learning import validate_feedback

    assert validate_feedback(["too_vague"], None) == ["too_vague"]
