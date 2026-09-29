"""Approved-answer retrieval with optional outcome and lesson ranking.

Hindsight finds candidates by meaning; SQLite decides which of them may be used. The order
Hindsight returns *is* the relevance ranking: no scores are recomputed and no threshold is
applied (the spike showed the scores aren't calibrated for this content; §13).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select

from ..schemas import PastAnswer
from .db import Answer, AnswerStats, Database, PastProposal, ProjectOutcome, parse_answer_code
from .lessons import AnswerLessons, LessonsMemory, answer_signals, answer_tag
from .memory import Memory, MemoryUnavailable, RecallHit
from .ranking import Scored, order_candidates, rank_factors

# Ask Hindsight for more than k, because stale copies (deleted or superseded in SQLite) are dropped.
OVERFETCH = 4


@dataclass
class Retrieval:
    past_answers: list[PastAnswer] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)  # scores and ranks, stored with each draft
    warning: str | None = None


async def retrieve(
    question: str,
    *,
    memory: Memory,
    db: Database,
    k: int,
    mode: str = "plain",
    client: str | None = None,
    industry: str | None = None,
    freshness_half_life_days: int = 730,
    lessons: LessonsMemory | None = None,
    relevance: str = "gated",
    min_share: float = 0.01,
    exclude_lost_proposals: bool = False,
) -> Retrieval:
    if mode == "none":
        return Retrieval()
    try:
        hits = await memory.recall(question, limit=max(k * OVERFETCH, 12))
    except MemoryUnavailable as exc:
        return Retrieval(warning=f"Library search unavailable ({exc}); drafted from the fact sheet only.")

    live, stats, written, lost_answers = _live_answers(db, hits)
    candidates = [(hit, live[hit.answer_code]) for hit in hits if hit.answer_code in live]
    warning_parts: list[str] = []
    if exclude_lost_proposals:
        excluded = [answer for _hit, answer in candidates if answer.id in lost_answers]
        candidates = [(hit, answer) for hit, answer in candidates if answer.id not in lost_answers]
        if excluded:
            details = ", ".join(f"{answer.code} ({lost_answers[answer.id]})" for answer in excluded)
            warning_parts.append(
                f"Kept lost-proposal answer(s) for learning but excluded them from drafting evidence: {details}."
            )

    # Hindsight mode: recall what the lessons bank remembers about these candidate answers.
    lesson_signals: dict[str, AnswerLessons] | None = None
    warning = None
    if mode == "hindsight" and candidates:
        if lessons is None:  # turned off with RFP_LESSONS=false: rank by local signals, no warning
            mode = "outcome"
        else:
            codes = {answer.code for _hit, answer in candidates}
            try:
                lesson_hits = await lessons.recall(question, tags=sorted(answer_tag(c) for c in codes))
                lesson_signals = answer_signals(lesson_hits, codes)
            except MemoryUnavailable as exc:
                mode = "outcome"
                warning_parts.append(f"Hindsight lessons unavailable ({exc}); ranked by local signals.")

    factors = {
        answer.code: rank_factors(
            answer, stats.get(answer.id), recall_rank=hit.rank, client=client, industry=industry,
            half_life_days=freshness_half_life_days, written_on=written.get(answer.id),
            lessons=(lesson_signals.get(answer.code, AnswerLessons()) if lesson_signals is not None else None),
        )
        for hit, answer in candidates
    }
    if mode in ("outcome", "hindsight"):
        # Memory reorders within groups of about equally relevant answers (ranking.order_candidates).
        candidates = order_candidates(
            [Scored((hit, answer), factors[answer.code].score, factors[answer.code].relevance, hit.final, hit.rank)
             for hit, answer in candidates],
            policy=relevance, min_share=min_share,
        )

    past: list[PastAnswer] = []
    retrieved: list[dict] = []
    for hit, answer in candidates:
        past.append(
            PastAnswer(
                id=answer.code,
                question=answer.question,
                answer=answer.answer,
                client=answer.client,
                industry=answer.industry,
                approved_on=answer.updated_at.date(),
            )
        )
        factor = factors[answer.code]
        retrieved.append(hit.as_dict() | {
            "position": len(past),
            "mode": mode,
            "relevance_policy": relevance if mode in ("outcome", "hindsight") else None,
            "outcome_score": factor.score if mode in ("outcome", "hindsight") else None,
            "relevance": factor.relevance,
            "quality": factor.quality,
            "freshness": factor.freshness,
            "context": factor.context,
            "lessons": factor.lessons if mode == "hindsight" else None,
            "lesson_evidence": (lesson_signals.get(answer.code, AnswerLessons()).evidence
                                if lesson_signals is not None else []),
            "reasons": factor.reasons if mode in ("outcome", "hindsight") else [f"semantic rank #{hit.rank}"],
        })
        if len(past) == k:
            break
    warning = " ".join(warning_parts) or warning
    return Retrieval(past_answers=past, retrieved=retrieved, warning=warning)


def _live_answers(
    db: Database, hits: list[RecallHit]
) -> tuple[dict[str, Answer], dict[int, AnswerStats], dict[int, date], dict[int, str]]:
    ids = [i for i in (parse_answer_code(h.answer_code) for h in hits) if i is not None]
    if not ids:
        return {}, {}, {}, {}
    with db.session() as session:
        rows = session.scalars(select(Answer).where(Answer.id.in_(ids))).all()
        stats = session.scalars(select(AnswerStats).where(AnswerStats.answer_id.in_(ids))).all()
        written = dict(session.execute(
            select(Answer.id, PastProposal.submitted_on)
            .join(PastProposal, Answer.past_proposal_id == PastProposal.id)
            .where(Answer.id.in_(ids), PastProposal.submitted_on.is_not(None))
        ).all())
        source_outcomes = list(session.execute(
            select(Answer.id, PastProposal.result, PastProposal.loss_reason)
            .join(PastProposal, Answer.past_proposal_id == PastProposal.id)
            .where(Answer.id.in_(ids))
        ).all())
        source_outcomes += list(session.execute(
            select(Answer.id, ProjectOutcome.result, ProjectOutcome.loss_reason)
            .join(ProjectOutcome, Answer.project_id == ProjectOutcome.project_id)
            .where(Answer.id.in_(ids))
        ).all())
    lost_answers = {
        answer_id: (reason or "proposal lost")
        for answer_id, result, reason in source_outcomes
        if result == "lost"
    }
    return (
        {row.code: row for row in rows if row.live},
        {row.answer_id: row for row in stats},
        written,
        lost_answers,
    )
