"""V2's deterministic, explainable outcome ranking.

Hindsight remains the candidate generator. This module combines its rank with exact signals from
SQLite. Keeping this pure makes the V4 ablations cheap and completely offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any

from .db import Answer, AnswerStats

# How relevance limits what memory can do to the order:
#   rank   the V2 pre-fix rule: score = 1/Hindsight rank x memory x freshness x context, sorted by
#          score. A strong enough lesson could lift an off-topic answer above a relevant one.
#   gated  memory reorders only answers that are about equally relevant. Candidates are grouped by
#          Hindsight's relevance score: the first group is every candidate scoring at least
#          `min_share` of the best one, the next group the same over what's left, and so on. Inside a
#          group the usual score decides; no answer ever moves above a group that ranked above it.
RELEVANCE_POLICIES = ("rank", "gated")


@dataclass(frozen=True)
class Scored:
    item: Any
    score: float  # rank_factors().score
    relevance: float  # rank_factors().relevance, the part of the score that comes from Hindsight's order
    final: float | None  # Hindsight's relevance score for the candidate
    rank: int  # Hindsight's order


def order_candidates(scored: list[Scored], *, policy: str = "gated", min_share: float = 0.01,
                     within: str = "rank") -> list[Any]:
    """Order candidates for drafting. `within` = "rank" keeps Hindsight's order as part of the score
    inside a group; "flat" lets memory, freshness and context alone decide inside a group."""
    if policy == "rank":
        return [s.item for s in sorted(scored, key=lambda s: (-s.score, s.rank))]
    if policy != "gated":
        raise ValueError(f"relevance policy must be one of {RELEVANCE_POLICIES}")

    def key(s: Scored):  # noqa: ANN202
        value = s.score / s.relevance if within == "flat" and s.relevance else s.score
        return (-value, s.rank)

    remaining = sorted(scored, key=lambda s: s.rank)
    ordered: list[Any] = []
    while remaining:
        best = max((s.final or 0.0) for s in remaining)
        group = [s for s in remaining if best <= 0 or (s.final or 0.0) >= min_share * best]
        ordered += [s.item for s in sorted(group, key=key)]
        remaining = [s for s in remaining if s not in group]
    return ordered


@dataclass(frozen=True)
class RankFactors:
    relevance: float
    quality: float
    freshness: float
    context: float
    score: float
    reasons: list[str]
    lessons: float = 1.0  # Hindsight lesson factor (hindsight mode only)


def quality_score(stats: AnswerStats | None) -> float:
    """A neutral-prior, smoothed score. New answers are neither promoted nor buried."""
    if stats is None:
        return 0.5
    positive = (stats.accepted or 0) + (stats.length_edits or 0) + 0.5 * (stats.light_edits or 0)
    negative = 0.25 * (stats.heavy_edits or 0) + 0.5 * (stats.rewritten or 0) + (stats.rejected or 0)
    ratings = (((stats.rating_total or 0) / stats.rating_count) - 3) * 0.1 if stats.rating_count else 0.0
    raw = (2.0 + positive - negative + ratings + (stats.debrief_credit or 0) + (stats.outcome_credit or 0)) / (
        4.0 + (stats.times_used or 0)
    )
    return round(max(0.05, min(1.25, raw)), 6)


def freshness_score(
    answer: Answer, *, now: datetime | None = None, half_life_days: int = 730, written_on: date | None = None
) -> float:
    """Age decay from when the answer was written: the past proposal's submission date when known,
    otherwise when it was last approved. (Import date would make a 2023 answer look new.)"""
    if not answer.live:
        return 0.0
    now = now or datetime.now(UTC)
    updated = datetime.combine(written_on, time(12, 0), tzinfo=UTC) if written_on else answer.updated_at
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    age_days = max(0.0, (now - updated).total_seconds() / 86400)
    return round(0.5 ** (age_days / half_life_days), 6)


def rank_factors(
    answer: Answer,
    stats: AnswerStats | None,
    *,
    recall_rank: int,
    client: str | None,
    industry: str | None,
    now: datetime | None = None,
    half_life_days: int = 730,
    written_on: date | None = None,
    lessons=None,  # lessons.AnswerLessons, hindsight mode only  # noqa: ANN001
) -> RankFactors:
    relevance = 1.0 / max(1, recall_rank)
    quality = quality_score(stats)
    freshness = freshness_score(answer, now=now, half_life_days=half_life_days, written_on=written_on)
    context = 1.0
    reasons = [f"semantic rank #{recall_rank}"]
    if client and answer.client and client.strip().lower() == answer.client.strip().lower():
        context *= 1.10
        reasons.append("same client")
    if industry and answer.industry and industry.strip().lower() == answer.industry.strip().lower():
        context *= 1.05
        reasons.append("same industry")
    if stats and stats.times_used:
        reasons.append(
            f"{stats.accepted or 0} accepted, "
            f"{(stats.light_edits or 0) + (stats.length_edits or 0) + (stats.heavy_edits or 0)} edited, "
            f"{stats.rewritten or 0} rewritten, {stats.rejected or 0} rejected"
        )
    else:
        reasons.append("no review history yet")
    if stats and stats.suggested_supersede:
        quality *= 0.1
        reasons.append("flagged outdated")
    if written_on:
        reasons.append(f"written {written_on.isoformat()}")
    lesson_factor = 1.0
    if lessons is not None:
        # Hindsight mode: what the lessons bank remembers replaces the SQLite-only quality score.
        lesson_factor = lessons.factor
        score = relevance * lesson_factor * freshness * context
        if lessons.positive or lessons.negative or lessons.neutral:
            reasons.append(f"Hindsight: {lessons.positive} positive, {lessons.negative} negative lessons")
        else:
            reasons.append("Hindsight: no lessons yet")
    else:
        score = relevance * quality * freshness * context
    return RankFactors(
        relevance=round(relevance, 6), quality=round(quality, 6), freshness=freshness,
        context=round(context, 6), score=round(score, 8), reasons=reasons, lessons=round(lesson_factor, 4),
    )
