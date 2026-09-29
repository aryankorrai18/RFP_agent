"""All data shapes: the fact sheet, the models the AI must return, and the API response.

The AI output models (ExtractionResult, DraftResult) are passed to the Anthropic SDK as
structured-output formats, so they use plain types only (no numeric or length constraints).
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FACT_ID_PATTERN = re.compile(r"^[A-Z]+-\d+$")


# --- Fact sheet --------------------------------------------------------------------------


class Fact(BaseModel):
    id: str
    topic: str = ""
    statement: str
    valid_from: date | None = None
    valid_to: date | None = None

    def is_live(self, today: date) -> bool:
        return self.valid_to is None or self.valid_to >= today


class FactSheet(BaseModel):
    company: str
    facts: list[Fact]

    @field_validator("company")
    @classmethod
    def _company_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("company must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def _check_facts(self) -> FactSheet:
        if not self.facts:
            raise ValueError("facts must contain at least one fact")
        seen: set[str] = set()
        for fact in self.facts:
            if not FACT_ID_PATTERN.match(fact.id):
                raise ValueError(f"fact id {fact.id!r} must look like FACT-001")
            if fact.id in seen:
                raise ValueError(f"duplicate fact id {fact.id!r}")
            if not fact.statement.strip():
                raise ValueError(f"fact {fact.id} has an empty statement")
            seen.add(fact.id)
        return self


# --- AI call #1: requirement extraction ---------------------------------------------------


class ExtractedRequirement(BaseModel):
    section: str | None = Field(description="Heading or section the item appears under, or null.")
    question: str = Field(description="The buyer's wording of the item, verbatim where possible.")
    mandatory: bool | None = Field(
        description="true if marked mandatory or phrased must/shall/required; "
        "false if marked optional; otherwise null."
    )
    word_limit: int | None = Field(description="Word limit stated for this item, or null.")
    reference: str | None = Field(
        description='Item number or location, e.g. "3.2" or "Sheet Security, Row 5", or null.'
    )


class ExtractionResult(BaseModel):
    requirements: list[ExtractedRequirement]


# --- AI call #2: draft generation ---------------------------------------------------------


class DraftClaimOut(BaseModel):
    text: str = Field(description="One factual statement about the company made in the answer.")
    source_ids: list[str] = Field(description="IDs of the facts that support the whole statement.")


class DraftResult(BaseModel):
    answer: str = Field(description="The answer text, or empty if nothing can be supported.")
    claims: list[DraftClaimOut]
    unsupported_claims: list[str] = Field(
        description="Statements in the answer that no fact supports. Normally empty."
    )
    needs_sme: bool = Field(description="true if the facts don't fully cover the requirement.")
    sme_question: str | None = Field(
        description="Specific question for a subject-matter expert when needs_sme is true, else null."
    )


# --- AI call #3 (V1): past-proposal pair extraction ---------------------------------------


class ExtractedPair(BaseModel):
    section: str | None = Field(description="Heading or section the pair appears under, or null.")
    reference: str | None = Field(description='Item number or location, e.g. "3.2", or null.')
    question: str = Field(description="The buyer's question or requirement, verbatim.")
    answer: str = Field(description="The vendor's answer, copied verbatim from the document. Never rewritten.")


class PairsResult(BaseModel):
    pairs: list[ExtractedPair]


class JudgeScores(BaseModel):
    """1 (poor) to 5 (excellent). Out-of-range values are clamped by the reader, not rejected."""

    accurate: int = Field(description="Every claim is supported by the evidence; nothing contradicts the fact sheet. 1-5.")
    answers_question: int = Field(description="Addresses every part of what the buyer asked. 1-5.")
    specific: int = Field(description="Concrete, verifiable details rather than vague assurances, where supported. 1-5.")


class JudgeResult(BaseModel):
    """AI output of the V4 pairwise judge: which of two blinded drafts a careful proposal manager would submit."""

    winner: Literal["A", "B", "tie"]
    reason: str = Field(description="Why, in at most 60 words, naming the deciding difference.")
    A: JudgeScores
    B: JudgeScores


class PastAnswer(BaseModel):
    """An approved library answer offered to the drafter (V1). Not an AI output."""

    id: str
    question: str
    answer: str
    client: str | None = None
    industry: str | None = None
    approved_on: date | None = None


# --- API response --------------------------------------------------------------------------

DraftStatus = Literal["drafted", "needs_sme", "failed"]
Flag = Literal[
    "unsupported_claims", "invalid_citation", "over_word_limit", "no_claims", "empty_answer",
    "sme_template",
]


class Requirement(BaseModel):
    id: str
    section: str | None
    question: str
    mandatory: bool | None
    word_limit: int | None
    reference: str | None


class Claim(BaseModel):
    text: str
    source_ids: list[str]
    valid: bool
    invalid_source_ids: list[str] = []


class Draft(BaseModel):
    requirement_id: str
    status: DraftStatus
    answer: str = ""
    claims: list[Claim] = []
    sources: list[str] = []
    unsupported_claims: list[str] = []
    invalid_citations: list[str] = []
    needs_sme: bool = False
    sme_question: str | None = None
    word_count: int = 0
    over_word_limit: bool = False
    flags: list[Flag] = []
    served_by_model: str | None = None
    error: str | None = None

    @property
    def grounded(self) -> bool:
        return self.status == "drafted" and not self.flags


class FactOut(BaseModel):
    id: str
    topic: str
    statement: str
