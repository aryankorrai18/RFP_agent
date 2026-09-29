from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from backend.llm import LLMError
from backend.schemas import ExtractionResult
from backend.service import PipelineError, load_fact_sheet, run_pipeline
from tests.conftest import TODAY, FakeLLM, docx_bytes, grounded, needs_sme, requirement

RFP = docx_bytes("3.1 Do you support SSO?", "3.2 Do you support SCIM?")


def run(llm, fact_sheet, settings, data=RFP, filename="rfp.docx"):
    return asyncio.run(
        run_pipeline(filename=filename, data=data, fact_sheet=fact_sheet, llm=llm, settings=settings, today=TODAY)
    )


def two_requirements() -> ExtractionResult:
    return ExtractionResult(
        requirements=[
            requirement("Do you support SSO?", section="Security", mandatory=True, word_limit=100, reference="3.1"),
            requirement("Do you support SCIM?", section="Security", reference="3.2"),
        ]
    )


def test_pipeline_assigns_ids_drafts_every_requirement_and_counts(fact_sheet, settings):
    llm = FakeLLM(
        two_requirements(),
        drafter=lambda r: grounded("We support SAML SSO.", "FACT-001") if "SSO" in r.question else needs_sme("SCIM?"),
    )
    response = run(llm, fact_sheet, settings)

    assert [r.id for r in response.requirements] == ["REQ-001", "REQ-002"]
    assert response.requirements[0].word_limit == 100
    assert [d.status for d in response.drafts] == ["drafted", "needs_sme"]
    assert response.stats.model_dump() == {
        "requirements": 2, "drafted": 1, "grounded": 1, "needs_sme": 1, "failed": 0, "flagged": 0,
    }
    assert response.usage.input_tokens == 100 + 2 * 10
    assert response.usage.cache_read_input_tokens == 16
    assert response.model == "fake-model"


def test_expired_facts_are_not_sent_and_citing_them_is_invalid(fact_sheet, settings):
    llm = FakeLLM(
        ExtractionResult(requirements=[requirement("Q?")]),
        drafter=lambda r: grounded("Old claim.", "FACT-009"),
    )
    response = run(llm, fact_sheet, settings)

    assert {f.id for f in llm.facts_seen[0]} == {"FACT-001", "FACT-002"}
    assert {f.id for f in response.facts} == {"FACT-001", "FACT-002"}
    assert response.drafts[0].invalid_citations == ["FACT-009"]
    assert any("expired" in w for w in response.warnings)


def test_one_failed_draft_does_not_fail_the_run(fact_sheet, settings):
    def drafter(req):
        if req.id == "REQ-001":
            raise LLMError("refused", "drafting REQ-001: the model declined")
        return grounded("ok", "FACT-002")

    response = run(FakeLLM(two_requirements(), drafter), fact_sheet, settings)
    assert [d.status for d in response.drafts] == ["failed", "drafted"]
    assert response.drafts[0].error == "drafting REQ-001: the model declined"
    assert response.stats.failed == 1


def test_a_quota_error_stops_the_remaining_drafts(fact_sheet, settings):
    def drafter(req):
        raise LLMError("api_error", f"drafting {req.id}: Gemini quota used up for fake-model (daily limit)",
                       reason="quota_exhausted")

    many = ExtractionResult(requirements=[requirement(f"Question {i}?") for i in range(1, 7)])
    llm = FakeLLM(many, drafter)
    response = run(llm, fact_sheet, replace(settings, draft_concurrency=1))
    assert len(llm.drafted) == 1  # the other 5 would fail the same way, so they weren't sent
    assert [d.status for d in response.drafts] == ["failed"] * 6
    assert all(d.error.startswith("Not attempted") for d in response.drafts[1:])
    assert any(w.startswith("Stopped early") and "5 of 6 weren't attempted" in w for w in response.warnings)


def test_blank_extracted_questions_are_dropped(fact_sheet, settings):
    llm = FakeLLM(ExtractionResult(requirements=[requirement("  "), requirement("Real question?")]))
    response = run(llm, fact_sheet, settings)
    assert [r.question for r in response.requirements] == ["Real question?"]
    assert response.requirements[0].id == "REQ-001"


def test_no_requirements_is_an_error(fact_sheet, settings):
    with pytest.raises(PipelineError) as error:
        run(FakeLLM(ExtractionResult(requirements=[])), fact_sheet, settings)
    assert (error.value.code, error.value.http_status) == ("no_requirements_found", 422)


def test_too_many_requirements_is_rejected_not_truncated(fact_sheet, settings):
    llm = FakeLLM(ExtractionResult(requirements=[requirement(f"Q{i}?") for i in range(4)]))
    with pytest.raises(PipelineError) as error:
        run(llm, fact_sheet, replace(settings, max_requirements=3))
    assert error.value.code == "too_many_requirements"
    assert llm.drafted == []


@pytest.mark.parametrize(
    ("kind", "code"), [("auth", "llm_auth_failed"), ("refused", "extraction_failed"), ("malformed", "extraction_failed")]
)
def test_extraction_errors_map_to_502(fact_sheet, settings, kind, code):
    with pytest.raises(PipelineError) as error:
        run(FakeLLM(LLMError(kind, "boom")), fact_sheet, settings)
    assert (error.value.code, error.value.http_status) == (code, 502)


def test_all_expired_fact_sheet_is_rejected(fact_sheet, settings):
    only_expired = fact_sheet.model_copy(update={"facts": [fact_sheet.facts[2]]})
    with pytest.raises(PipelineError) as error:
        run(FakeLLM(two_requirements()), only_expired, settings)
    assert error.value.code == "invalid_fact_sheet"


def test_run_is_saved_as_json(fact_sheet, settings):
    response = run(FakeLLM(two_requirements()), fact_sheet, settings)
    saved = json.loads((settings.runs_dir / f"{response.run_id}.json").read_text(encoding="utf-8"))
    assert saved["run_id"] == response.run_id
    assert len(saved["drafts"]) == 2


def test_drafting_respects_the_concurrency_limit(fact_sheet, settings):
    active = 0
    peak = 0

    class SlowLLM(FakeLLM):
        async def draft_answer(self, company, facts, req):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return await super().draft_answer(company, facts, req)

    llm = SlowLLM(ExtractionResult(requirements=[requirement(f"Q{i}?") for i in range(10)]))
    run(llm, fact_sheet, replace(settings, draft_concurrency=3))
    assert peak == 3


def test_load_fact_sheet_reports_problems():
    with pytest.raises(PipelineError) as error:
        load_fact_sheet(b"{not json", "facts.json")
    assert error.value.code == "invalid_fact_sheet"

    bad = {"company": "X", "facts": [{"id": "F1", "statement": "s"}]}
    with pytest.raises(PipelineError) as error:
        load_fact_sheet(json.dumps(bad), "facts.json")
    assert "FACT-001" in error.value.message

    duplicate = {"company": "X", "facts": [{"id": "FACT-1", "statement": "a"}, {"id": "FACT-1", "statement": "b"}]}
    with pytest.raises(PipelineError) as error:
        load_fact_sheet(json.dumps(duplicate), "facts.json")
    assert "duplicate" in error.value.message
