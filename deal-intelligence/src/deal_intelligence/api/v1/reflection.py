"""What Hindsight's Reflect says about deals like this one. Cached, so reading it is free; refreshing
asks the lessons bank again (it spends Hindsight credits on the Cloud backend, nothing on local)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...errors import PipelineError
from .db import Deal, DealReflection, utcnow
from .lessons import reflect_deal_type
from .memory import MemoryUnavailable
from .ranking import deal_facts

if TYPE_CHECKING:
    from .context import V1Context


def _view(row: DealReflection | None, backend: str) -> dict:
    if row is None:
        return {"state": "missing", "text": None, "based_on": [], "created_at": None, "backend": backend}
    return {"state": "ready", "text": row.text, "based_on": row.based_on, "created_at": row.created_at, "backend": row.backend}


def get_reflection(ctx: V1Context, deal_id: int) -> dict:
    with ctx.db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        return _view(session.get(DealReflection, deal_id), ctx.memory_backend)


async def refresh_reflection(ctx: V1Context, deal_id: int) -> dict:
    if ctx.lessons is None:
        raise PipelineError("lessons_disabled", "The lessons bank is turned off (DEAL_LESSONS).", 409)
    with ctx.db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        facts = deal_facts(deal, deal.signals, list(deal.stakeholders))
    await ctx.sync_lessons()
    try:
        brief = await reflect_deal_type(ctx.lessons, facts)
    except MemoryUnavailable as exc:
        raise PipelineError("hindsight_unavailable", str(exc), 503) from exc
    with ctx.db.session() as session:
        row = session.get(DealReflection, deal_id)
        if row is None:
            row = DealReflection(deal_id=deal_id, text=brief.text, based_on=brief.based_on, backend=ctx.memory_backend)
            session.add(row)
        else:
            row.text, row.based_on, row.backend, row.created_at = brief.text, brief.based_on, ctx.memory_backend, utcnow()
        session.commit()
        return _view(row, ctx.memory_backend)
