"""Closing a deal: the outcome, the plays that were really used, and the credit those plays earn.

Credit is deliberately small and only breaks ties: a win gives +0.1 to the plays that were ticked, a
loss on something a play could have changed (QUALITY_LOSSES) gives -0.1, and any other loss (price,
no decision, timing, champion left, a plain competitor loss) gives none either way. Exact counts live
beside it in PlayStats and are what recommendations show.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import select

from ...errors import PipelineError
from .db import (
    LOSS_REASONS, QUALITY_LOSSES, Database, Deal, DealSignals, MemoryEvent, Play, PlayStats, deal_code, utcnow,
)
from .lessons import collect_lessons
from .sync import touch

WIN_CREDIT = 0.1


def outcome_credit(result: str, loss_reason: str | None) -> float:
    if result == "won":
        return WIN_CREDIT
    if result == "lost" and loss_reason in QUALITY_LOSSES:
        return -WIN_CREDIT
    return 0.0


def _effect(result: str, loss_reason: str | None) -> tuple[int, int, int, float]:
    """(won, lost_quality, lost_other, credit) one play gains from one deal."""
    if result == "won":
        return 1, 0, 0, outcome_credit(result, loss_reason)
    if result == "lost" and loss_reason in QUALITY_LOSSES:
        return 0, 1, 0, outcome_credit(result, loss_reason)
    if result == "lost":
        return 0, 0, 1, 0.0
    return 0, 0, 0, 0.0


def _apply(session, play_code: str, result: str, loss_reason: str | None, sign: int) -> float:  # noqa: ANN001
    stats = session.get(PlayStats, play_code)
    if stats is None:
        stats = PlayStats(play_code=play_code, times_used=0, won=0, lost_quality=0, lost_other=0, outcome_credit=0.0)
        session.add(stats)
    won, lost_quality, lost_other, credit = _effect(result, loss_reason)
    stats.times_used = (stats.times_used or 0) + sign
    stats.won = (stats.won or 0) + sign * won
    stats.lost_quality = (stats.lost_quality or 0) + sign * lost_quality
    stats.lost_other = (stats.lost_other or 0) + sign * lost_other
    stats.outcome_credit = round((stats.outcome_credit or 0.0) + sign * credit, 4)
    stats.updated_at = utcnow()
    return sign * credit


def _describe(result: str, loss_reason: str | None) -> str:
    return result + (f" ({loss_reason})" if loss_reason else "")


def record_outcome(
    ctx,  # noqa: ANN001
    deal_id: int,
    result: str,
    loss_reason: str | None = None,
    plays_used: list[str] | None = None,
    closed_on: date | None = None,
) -> dict:
    """Close a deal as won or lost. Re-recording a different outcome applies only the difference to
    the play statistics. `plays_used` is what the team actually did; None keeps the plays already
    recorded on the deal."""
    if result not in ("won", "lost"):
        raise PipelineError("invalid_request", "result must be won or lost.", 422)
    reason = (loss_reason or "").strip() or None
    if result == "lost":
        if reason is None:
            raise PipelineError("invalid_request", "A lost deal needs a loss reason.", 422)
        if reason not in LOSS_REASONS:
            raise PipelineError("invalid_request", f"loss_reason must be one of {', '.join(LOSS_REASONS)}.", 422)
    else:
        reason = None

    with ctx.db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        signals = session.get(DealSignals, deal_id)
        previous_plays = list(signals.plays_used or []) if signals else []
        plays = list(dict.fromkeys(plays_used)) if plays_used is not None else previous_plays
        known = set(session.scalars(select(Play.code)))
        unknown = [p for p in plays if p not in known]
        if unknown:
            raise PipelineError("invalid_request", f"Unknown play(s): {', '.join(unknown)}.", 422)

        previous = (deal.result, deal.loss_reason) if deal.result in ("won", "lost") else None
        delta: dict[str, float] = {}
        if previous is not None:  # take back what the earlier outcome gave, then give the new credit
            for play in previous_plays:
                delta[play] = delta.get(play, 0.0) + _apply(session, play, previous[0], previous[1], -1)
        for play in plays:
            delta[play] = delta.get(play, 0.0) + _apply(session, play, result, reason, +1)

        deal.result, deal.loss_reason, deal.stage = result, reason, "closed"
        deal.closed_on = closed_on or date.today()
        if signals is None:
            signals = DealSignals(deal_id=deal_id, objections=[], competitors=[], pricing={}, promises=[], source="manual")
            session.add(signals)
        signals.plays_used = plays
        touch(signals)
        deal.hindsight_status, deal.hindsight_attempts, deal.hindsight_error = "pending", 0, None  # re-retain the summary
        touch(deal)

        detail = f"{deal.code} recorded as {_describe(result, reason)}"
        if previous is not None and previous != (result, reason):
            detail += f", previously {_describe(*previous)}"
        session.add(MemoryEvent(kind="outcome_recorded", deal_id=deal_id, detail=detail + ". Plays used: "
                                + (", ".join(plays) or "none") + "."))
        for play, amount in sorted(delta.items()):
            if round(amount, 4):
                session.add(MemoryEvent(
                    kind="play_credit", deal_id=deal_id, play_code=play,
                    detail=f"{play} {amount:+.1f} credit after {deal.code} was {_describe(result, reason)}.",
                ))
        session.commit()
        summary = {
            "deal_id": deal_id, "code": deal.code, "result": result, "loss_reason": reason,
            "closed_on": deal.closed_on.isoformat(), "plays_used": plays,
            "credit": outcome_credit(result, reason), "credit_applied": {p: round(a, 4) for p, a in delta.items() if round(a, 4)},
            "previous": {"result": previous[0], "loss_reason": previous[1]} if previous else None,
        }
    summary["lessons_changed"] = collect_lessons(ctx.db)
    ctx.schedule_sync()
    return summary


def rebuild_play_stats(db: Database) -> int:
    """Recompute every play's statistics from scratch from the closed deals. Returns the number of
    closed deals counted. Run after seeding, or to check that incremental updates agree."""
    with db.session() as session:
        for play in session.scalars(select(Play)):
            stats = session.get(PlayStats, play.code)
            if stats is None:
                session.add(PlayStats(play_code=play.code))
            else:
                stats.times_used = stats.won = stats.lost_quality = stats.lost_other = 0
                stats.outcome_credit = 0.0
        session.flush()
        deals = session.scalars(select(Deal).where(Deal.status == "active", Deal.result.in_(("won", "lost")))).all()
        for deal in deals:
            signals = session.get(DealSignals, deal.id)
            for play in dict.fromkeys(signals.plays_used if signals else []):
                if session.get(Play, play) is not None:
                    _apply(session, play, deal.result, deal.loss_reason, +1)
        session.commit()
        return len(deals)


def memory_journal(db: Database, limit: int = 50) -> list[dict]:
    """The learning journal, newest first."""
    with db.session() as session:
        events = session.scalars(select(MemoryEvent).order_by(MemoryEvent.id.desc()).limit(limit)).all()
        return [
            {"id": e.id, "kind": e.kind, "deal_id": e.deal_id, "deal_code": deal_code(e.deal_id) if e.deal_id else None,
             "play_code": e.play_code, "detail": e.detail, "created_at": e.created_at.isoformat()}
            for e in events
        ]
