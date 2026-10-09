"""A memory self-check the viewer can trust: leave one closed deal out, rebuild the recommendations
from the others using only the local database (no Hindsight, no model), and see whether they would
have pointed at what actually happened. The sample is small and the result is stated with its size."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from .db import Deal
from .retrieval import build_recommendations

if TYPE_CHECKING:
    from .context import V1Context

TOP = 3


async def memory_quality(ctx: V1Context) -> dict:
    with ctx.db.session() as session:
        closed = [
            (d.id, d.code, d.result, set(d.signals.plays_used if d.signals else []),
             {o["type"] for o in (d.signals.objections if d.signals else []) if o.get("status") == "unresolved"})
            for d in session.scalars(select(Deal).where(Deal.status == "active", Deal.result != "open").order_by(Deal.id))
            if d.signals is not None
        ]
    won = {"n": 0, "covered": 0, "hit": 0}
    lost = {"n": 0, "warned": 0, "avoid_hit": 0, "had_unresolved": 0}
    for deal_id, _code, result, plays, unresolved in closed:
        rec = await build_recommendations(ctx, deal_id, "similar", source="sqlite", exclude_deal_ids=frozenset({deal_id}))
        if result == "won":
            won["n"] += 1
            top = [p.play_code for p in rec.plays[:TOP]]
            if top:
                won["covered"] += 1
                won["hit"] += bool(plays & set(top))
        else:
            lost["n"] += 1
            if unresolved:
                lost["had_unresolved"] += 1
                lost["warned"] += any(w.objection_type in unresolved for w in rec.warnings)
            lost["avoid_hit"] += bool(plays & {a.play_code for a in rec.avoid})
    return {
        "n_closed": len(closed), "won": won, "lost": lost, "top": TOP,
        "method": "Leave one closed deal out, rebuild the recommendations from the other deals using only the local "
                  "database, and compare with what really happened. No model and no Hindsight call.",
        "caveat": f"Only {len(closed)} closed deals, so treat these as a sanity check, not a benchmark.",
    }
