"""Pydantic models for what the AI returns.

AI output models use plain types only (no numeric or length constraints): they go to the providers'
structured-output helpers, and Anthropic's rejects constraints. Anything stricter is checked in code
after the call (see api/v1/briefs.py and signals.py).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---- Signals extraction: one call per uploaded deal ----------------------------------------------

class ObjectionOut(BaseModel):
    type: str  # one of db.OBJECTION_TYPES; anything else is mapped to the closest or dropped in code
    text: str
    status: str  # raised | addressed | unresolved
    first_seen_on: str | None = None  # YYYY-MM-DD
    evidence: list[str] = Field(default_factory=list)  # INT-xxxx ids the objection is read from


class PromiseOut(BaseModel):
    text: str
    owner: str | None = None
    due_on: str | None = None  # YYYY-MM-DD
    status: str = "open"  # open | kept | missed
    evidence: list[str] = Field(default_factory=list)


class StakeholderOut(BaseModel):
    name: str
    title: str | None = None
    stance: str = "neutral"  # champion | supporter | neutral | blocker
    engaged: bool = True
    economic_buyer: bool = False


class DealSignalsResult(BaseModel):
    stage: str | None = None
    objections: list[ObjectionOut] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    discount_requested: bool = False
    pricing_notes: str | None = None
    promises: list[PromiseOut] = Field(default_factory=list)
    stakeholders: list[StakeholderOut] = Field(default_factory=list)
    plays_used: list[str] = Field(default_factory=list)  # PLAY-xx codes from the catalogue only


# ---- The brief: one call per deal and memory mode ------------------------------------------------

class BriefClaim(BaseModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)  # INT-xxxx for this deal; D-xxx for similar deals


class BriefStep(BaseModel):
    play_code: str  # must be one of the offered candidate plays
    rationale: str
    source_ids: list[str] = Field(default_factory=list)  # D-xxx ids the rationale rests on (never lessons)


class DealAnswerResult(BaseModel):
    found: bool  # false when the notes and evidence do not answer the question
    answer: str
    source_ids: list[str] = Field(default_factory=list)  # INT- ids of this deal, or D- ids of past deals


class FollowupResult(BaseModel):
    subject: str  # an email subject, or the call title
    body: str
    source_ids: list[str] = Field(default_factory=list)  # INT- ids this draft relies on


class DealBriefResult(BaseModel):
    summary: str  # where the deal stands, in two or three sentences
    summary_sources: list[str] = Field(default_factory=list)
    this_deal: list[BriefClaim] = Field(default_factory=list)  # risks and facts about this deal; must cite INT- ids
    memory: list[BriefClaim] = Field(default_factory=list)  # what similar past deals say; must cite D- ids
    next_steps: list[BriefStep] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)  # what the AE should find out
