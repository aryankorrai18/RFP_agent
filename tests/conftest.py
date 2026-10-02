"""Shared fixtures. Everything runs offline: the AI is replaced by fakes (tests/v1_fakes.py)."""

from __future__ import annotations

import io
from dataclasses import replace
from datetime import date

import docx
import pytest

from rfp_assistant.config import Settings
from rfp_assistant.schemas import DraftClaimOut, DraftResult, ExtractedRequirement, Fact, FactSheet


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
