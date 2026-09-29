"""Shared fixtures. Everything runs offline: the AI is replaced by FakeLLM."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import replace
from datetime import date

import docx
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.llm import LLMError, LLMResult, TokenUsage
from backend.main import app, get_llm, get_settings
from backend.parser import ParsedDocument
from backend.schemas import (
    DraftClaimOut,
    DraftResult,
    ExtractedRequirement,
    ExtractionResult,
    Fact,
    FactSheet,
    Requirement,
)

TODAY = date(2026, 9, 28)


def requirement(question: str, **kwargs) -> ExtractedRequirement:
    fields = {"section": None, "mandatory": None, "word_limit": None, "reference": None}
    fields.update(kwargs)
    return ExtractedRequirement(question=question, **fields)


def grounded(answer: str, *fact_ids: str) -> DraftResult:
    return DraftResult(
        answer=answer,
        claims=[DraftClaimOut(text=answer, source_ids=list(fact_ids))],
        unsupported_claims=[],
        needs_sme=False,
        sme_question=None,
    )


def needs_sme(question: str) -> DraftResult:
    return DraftResult(answer="", claims=[], unsupported_claims=[], needs_sme=True, sme_question=question)


class FakeLLM:
    """Scripted stand-in for ClaudeLLM.

    extraction: an ExtractionResult to return, or an LLMError to raise.
    drafter: maps a Requirement to a DraftResult, or raises LLMError.
    """

    model = "fake-model"

    def __init__(
        self,
        extraction: ExtractionResult | LLMError,
        drafter: Callable[[Requirement], DraftResult] | None = None,
    ):
        self.extraction = extraction
        self.drafter = drafter or (lambda req: grounded(f"Answer to {req.id}", "FACT-001"))
        self.documents: list[ParsedDocument] = []
        self.drafted: list[Requirement] = []
        self.facts_seen: list[list[Fact]] = []

    async def extract_requirements(self, document: ParsedDocument) -> LLMResult[ExtractionResult]:
        self.documents.append(document)
        if isinstance(self.extraction, LLMError):
            raise self.extraction
        return LLMResult(output=self.extraction, model=self.model, usage=TokenUsage(input_tokens=100, output_tokens=50))

    async def draft_answer(self, company: str, facts: list[Fact], req: Requirement) -> LLMResult[DraftResult]:
        self.drafted.append(req)
        self.facts_seen.append(facts)
        result = self.drafter(req)
        return LLMResult(
            output=result,
            model=self.model,
            usage=TokenUsage(input_tokens=10, output_tokens=5, cache_read_input_tokens=8),
        )


@pytest.fixture(autouse=True)
def _isolated_workspaces(tmp_path, monkeypatch):
    """Tests never read or write the real workspace registry (data/workspaces.json)."""
    monkeypatch.setenv("RFP_WORKSPACES_FILE", str(tmp_path / "workspaces.json"))


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(Settings(), db_path=tmp_path / "rfp.db", uploads_dir=tmp_path / "uploads")


@pytest.fixture
def fact_sheet() -> FactSheet:
    return FactSheet(
        company="Test Co",
        facts=[
            Fact(id="FACT-001", topic="SSO", statement="Test Co supports SAML 2.0 SSO."),
            Fact(id="FACT-002", topic="Encryption", statement="Data is encrypted at rest with AES-256."),
            Fact(id="FACT-009", topic="Old", statement="An expired fact.", valid_to=date(2025, 1, 1)),
        ],
    )


def docx_bytes(*paragraphs: str) -> bytes:
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def api(settings):
    """A TestClient factory: api(fake_llm, **settings_overrides) -> TestClient."""

    def make(llm: FakeLLM, **overrides) -> TestClient:
        effective = replace(settings, **overrides)
        app.dependency_overrides[get_settings] = lambda: effective
        app.dependency_overrides[get_llm] = lambda: llm
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()
