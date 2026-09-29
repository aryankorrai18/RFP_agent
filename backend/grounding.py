"""Deterministic citation-validity and word-limit checks; no model call required."""

from __future__ import annotations

import re

from .schemas import Claim, Draft, DraftResult, Flag, Requirement

_WORD = re.compile(r"\S+")


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def normalise_source_id(raw: str) -> str:
    """Models occasionally wrap IDs as "[FACT-003]"; strip that and whitespace, nothing more."""
    return raw.strip().strip("[]").strip()


def evaluate_draft(
    requirement: Requirement,
    result: DraftResult,
    live_fact_ids: set[str],
    served_by_model: str,
) -> Draft:
    answer = result.answer.strip()

    claims: list[Claim] = []
    invalid_citations: list[str] = []
    sources: list[str] = []
    uncited_claim = False
    for raw_claim in result.claims:
        ids = [normalise_source_id(i) for i in raw_claim.source_ids if normalise_source_id(i)]
        invalid = [i for i in ids if i not in live_fact_ids]
        if not ids:
            uncited_claim = True
        for i in invalid:
            if i not in invalid_citations:
                invalid_citations.append(i)
        for i in ids:
            if i in live_fact_ids and i not in sources:
                sources.append(i)
        claims.append(
            Claim(text=raw_claim.text, source_ids=ids, valid=bool(ids) and not invalid, invalid_source_ids=invalid)
        )

    unsupported = [c.strip() for c in result.unsupported_claims if c.strip()]
    word_count = count_words(answer)
    over_limit = bool(requirement.word_limit) and word_count > requirement.word_limit

    flags: list[Flag] = []
    if unsupported:
        flags.append("unsupported_claims")
    if invalid_citations or uncited_claim:
        flags.append("invalid_citation")
    if over_limit:
        flags.append("over_word_limit")
    if answer and not claims:
        flags.append("no_claims")

    if result.needs_sme:
        status = "needs_sme"
    elif not answer:
        status = "needs_sme"
        flags.append("empty_answer")
    else:
        status = "drafted"

    sme_question = (result.sme_question or "").strip() or None
    if status == "needs_sme" and sme_question is None:
        sme_question = f"Please provide an approved answer for: {requirement.question}"

    return Draft(
        requirement_id=requirement.id,
        status=status,
        answer=answer,
        claims=claims,
        sources=sources,
        unsupported_claims=unsupported,
        invalid_citations=invalid_citations,
        needs_sme=status == "needs_sme",
        sme_question=sme_question if status == "needs_sme" else None,
        word_count=word_count,
        over_word_limit=over_limit,
        flags=flags,
        served_by_model=served_by_model,
    )
