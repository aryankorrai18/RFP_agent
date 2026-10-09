"""Which closed deals resemble this open deal, and what did they teach?

Hindsight finds candidate closed deals by meaning; SQLite decides which of them may be used. A
candidate counts as similar only if it passes the structural gate in ranking.py (shared problem keys,
or the same segment and industry); Hindsight's rank is only the tiebreak and the lessons bank only
breaks ties between plays. The recall query is built from the open deal's extracted signals, never
from the account name.

Modes (config.RETRIEVAL_MODES):
  none       nothing from other deals.
  longctx    every closed deal as a SimilarDeal, no plays or warnings: the brief layer puts all the
             summaries in the prompt.
  similar    recall + SQLite validation + gate; plays and warnings from contrast. No lessons.
  hindsight  as similar, plus lesson evidence from the lessons bank as a tiebreak.

Hindsight being down never fails a brief: similar deals are then found from SQLite alone with the
same gate, and `degraded` says so. `source="sqlite"` asks for that SQLite-only path on purpose (no
Hindsight, no lessons bank) and is not a degradation. In similar and hindsight mode `avoid` lists plays
tried in several similar deals that never won one; they are removed from `plays`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from ...config import RETRIEVAL_MODES
from ...errors import PipelineError
from .contracts import Recommendations, SimilarDeal
from .db import Deal, Interaction, Lesson, Play, interaction_code, parse_interaction_code
from .lessons import LessonsMemory, lesson_document_id, play_signals, play_tag
from .memory import DEAL_SUMMARY_TAG, INTERACTION_TAG, MemoryUnavailable, RecallHit, deal_summary_item
from .ranking import (
    DealFacts, LessonEvidence, avoid_plays, contrast_candidates, deal_facts, gate_and_order, rank_plays,
    recall_query, shared_keys, warnings,
)

if TYPE_CHECKING:
    from .context import V1Context
    from .db import Database

# Ask Hindsight for more than k: hits are dropped when SQLite no longer has the deal, it is not closed,
# or the gate rejects it.
OVERFETCH = 4
MAX_PLAYS = 8
AVOID_MIN_USED = 3  # similar closed deals that must have used a play, with no win, before it is flagged
UNSYNCED_RANK = 10_000  # rank given to a closed deal that is not in Hindsight yet


@dataclass
class _Closed:
    facts: DealFacts
    summary: str
    retained: bool  # whether Hindsight has its summary


def n_closed(db: Database) -> int:
    with db.session() as session:
        return session.scalar(
            select(func.count(Deal.id)).where(Deal.status == "active", Deal.result.in_(("won", "lost")))
        ) or 0


def _load(
    db: Database, deal_id: int, exclude: frozenset[int] = frozenset()
) -> tuple[DealFacts, dict[str, _Closed], dict[str, str], dict[str, frozenset[str]], int]:
    """The target's facts, the other closed deals, play names and addresses, and how many closed deals
    are in memory for this question. The target may itself be closed (leave-one-out): it is then never
    its own neighbour, and its own plays are not "already done" because they are what is being predicted."""
    with db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        plays = list(session.scalars(select(Play)))
        names = {p.code: p.name for p in plays}
        addresses = {p.code: frozenset(p.addresses or []) for p in plays}
        open_facts = deal_facts(deal, deal.signals, list(deal.stakeholders))
        if deal.result in ("won", "lost"):
            open_facts = replace(open_facts, plays_used=())
        closed: dict[str, _Closed] = {}
        rows = session.scalars(
            select(Deal).where(Deal.status == "active", Deal.result.in_(("won", "lost"))).order_by(Deal.id)
        ).all()
        total = len(rows)
        for row in rows:
            if row.id == deal_id or row.id in exclude:
                total -= 1
                continue
            closed[row.code] = _Closed(
                facts=deal_facts(row, row.signals, list(row.stakeholders)),
                summary=deal_summary_item(row, row.signals, list(row.stakeholders), names).content,
                retained=row.hindsight_status == "retained",
            )
    return open_facts, closed, names, addresses, total


def closed_facts(db: Database) -> list[DealFacts]:
    """Facts of every active closed deal (used by the leave-one-out evaluation and the Memory page)."""
    with db.session() as session:
        rows = session.scalars(
            select(Deal).where(Deal.status == "active", Deal.result.in_(("won", "lost"))).order_by(Deal.id)
        ).all()
        return [deal_facts(r, r.signals, list(r.stakeholders)) for r in rows]


def memory_state(rec: Recommendations) -> str:
    payload = {
        "similar": sorted(s.code for s in rec.similar),
        "plays": sorted([p.play_code, p.used_in_similar, p.won_in_similar, p.lost_quality_in_similar] for p in rec.plays),
        "avoid": sorted([a.play_code, a.used_in_similar, a.won_in_similar] for a in rec.avoid),
        "lessons": sorted({doc for p in rec.plays for doc in p.lesson_evidence}),
        "n_closed": rec.n_closed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _similar_deal(closed: _Closed, rank: int, hit: RecallHit | None, keys: list[str]) -> SimilarDeal:
    facts = closed.facts
    relevance = (hit.final if hit.final is not None else hit.semantic) if hit is not None else None
    return SimilarDeal(
        deal_id=facts.deal_id, code=facts.code, account=facts.account, result=facts.result,
        loss_reason=facts.loss_reason, rank=rank, relevance=float(relevance or 0.0), shared_keys=keys,
        plays_used=list(facts.plays_used), summary=closed.summary,
    )


async def build_recommendations(
    ctx: V1Context, deal_id: int, mode: str | None = None, *, source: str = "memory",
    exclude_deal_ids: frozenset[int] = frozenset(),
) -> Recommendations:
    """`source="sqlite"` skips Hindsight and the lessons bank and uses the structural path over the
    local database, without marking the result degraded. `exclude_deal_ids` removes closed deals from
    the candidates (leave-one-out)."""
    settings = ctx.settings
    mode = mode or settings.retrieval_mode
    if mode not in RETRIEVAL_MODES:
        raise PipelineError("invalid_request", f"mode must be one of {', '.join(RETRIEVAL_MODES)}.", 422)
    if source not in ("memory", "sqlite"):
        raise PipelineError("invalid_request", "source must be 'memory' or 'sqlite'.", 422)
    open_facts, closed, names, addresses, total = _load(ctx.db, deal_id, frozenset(exclude_deal_ids))
    rec = Recommendations(deal_id=deal_id, mode=mode, n_closed=total)

    if mode == "none":
        rec.memory_state = memory_state(rec)
        return rec
    if mode == "longctx":
        rec.similar = [
            _similar_deal(c, rank, None, shared_keys(open_facts, c.facts))
            for rank, c in enumerate(sorted(closed.values(), key=lambda c: c.facts.deal_id), start=1)
        ]
        rec.memory_state = memory_state(rec)
        return rec

    k = settings.similar_deals_k
    candidates: list[tuple[DealFacts, int]] = []
    hit_by_code: dict[str, RecallHit] = {}
    hits: list[RecallHit] | None
    if source == "sqlite":
        hits = None
    else:
        try:
            hits = await ctx.memory.recall(recall_query(open_facts), [DEAL_SUMMARY_TAG], limit=max(k * OVERFETCH, 12))
        except MemoryUnavailable as exc:
            hits = None
            rec.degraded = (f"Hindsight is unavailable ({exc}); similar deals were found from the recorded "
                            "deal signals in the local database instead.")
    if hits is None:
        candidates = [(c.facts, rank) for rank, c in enumerate(closed.values(), start=1)]
    else:
        for hit in hits:
            if hit.code in hit_by_code or hit.code not in closed:  # duplicate, deleted, open or this deal
                continue
            hit_by_code[hit.code] = hit
            candidates.append((closed[hit.code].facts, hit.rank))
        # A deal closed a moment ago may not be in Hindsight yet; SQLite already knows it.
        candidates += [(c.facts, UNSYNCED_RANK + c.facts.deal_id) for code, c in closed.items()
                       if not c.retained and code not in hit_by_code]

    ordered = gate_and_order(open_facts, candidates, settings.min_shared_keys)[:k]
    rec.similar = [
        _similar_deal(closed[facts.code], hit_by_code[facts.code].rank if facts.code in hit_by_code else position,
                      hit_by_code.get(facts.code), keys)
        for position, (facts, _rank, keys) in enumerate(ordered, start=1)
    ]
    similar_facts = [facts for facts, _rank, _keys in ordered]
    rec.avoid = avoid_plays(open_facts, similar_facts, names, AVOID_MIN_USED)
    avoided = {a.play_code for a in rec.avoid}
    contrast = [c for c in contrast_candidates(open_facts, similar_facts) if c.play_code not in avoided]
    hit_ranks = {code: hit.rank for code, hit in hit_by_code.items()}

    evidence: dict[str, LessonEvidence] = {}
    if mode == "hindsight" and source == "memory":
        evidence = await _lesson_evidence(ctx, rec, open_facts, {c.play_code for c in contrast}, set(closed))
    open_objections = frozenset(t for t, status in open_facts.objections if status != "addressed")
    plays = rank_plays(contrast, names, lessons=evidence,
                       hit_ranks=hit_ranks, addresses=addresses, open_objections=open_objections)[:MAX_PLAYS]
    rec.plays = plays
    rec.warnings = warnings(open_facts, similar_facts)
    rec.memory_state = memory_state(rec)
    return rec


async def _lesson_evidence(
    ctx: V1Context, rec: Recommendations, open_facts: DealFacts, codes: set[str], closed_codes: set[str]
) -> dict[str, LessonEvidence]:
    lessons: LessonsMemory | None = ctx.lessons
    if lessons is None or not ctx.settings.lessons_enabled:
        _note(rec, "The lessons bank is turned off; plays were ranked from similar deals only.")
        return {}
    if not codes:
        return {}
    try:
        hits = await lessons.recall(recall_query(open_facts), tags=sorted(play_tag(c) for c in codes),
                                    limit=max(len(codes) * 4, 20))
    except MemoryUnavailable as exc:
        _note(rec, f"Hindsight lessons are unavailable ({exc}); plays were ranked from similar deals only.")
        return {}
    with ctx.db.session() as session:
        documents = {lesson_document_id(key) for key in session.scalars(select(Lesson.key))}
    rec.lessons_used = True
    return play_signals(hits, codes, valid_deals=closed_codes, valid_documents=documents)


def _note(rec: Recommendations, text: str) -> None:
    rec.degraded = f"{rec.degraded} {text}" if rec.degraded else text


async def deal_timeline(ctx: V1Context, deal_id: int) -> dict:
    """The deal's own interactions as Hindsight recalls them (tag deal:<code>), each checked against
    SQLite, oldest first. If Hindsight is down, SQLite's list is returned with a note."""
    with ctx.db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        code = deal.code
        rows = {interaction_code(i.id): i for i in session.scalars(select(Interaction).where(Interaction.deal_id == deal_id))}
    degraded = None
    ranks: dict[str, int] = {}
    try:
        hits = await ctx.memory.recall(
            "timeline of emails, calls and meetings for this deal", [INTERACTION_TAG, f"deal:{code}"], limit=200
        )
        for hit in hits:
            if parse_interaction_code(hit.code) is not None and hit.code in rows and hit.code not in ranks:
                ranks[hit.code] = hit.rank
    except MemoryUnavailable as exc:
        degraded = f"Hindsight is unavailable ({exc}); the timeline is read from the local database."
        ranks = {c: n for n, c in enumerate(rows, start=1)}
    items = [
        {"code": c, "kind": rows[c].kind, "occurred_on": rows[c].occurred_on.isoformat(), "author": rows[c].author,
         "subject": rows[c].subject, "rank": rank}
        for c, rank in ranks.items()
    ]
    items.sort(key=lambda item: (item["occurred_on"], item["code"]))
    return {"deal_id": deal_id, "code": code, "items": items, "degraded": degraded}

