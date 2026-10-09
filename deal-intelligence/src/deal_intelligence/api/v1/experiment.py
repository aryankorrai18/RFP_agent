"""Before / after: brief the same deal under different memory states, then compare.

A fair comparison changes one thing. Every arm uses the same deal, model, prompt and temperature;
only the evidence block differs (none = this deal's own records, longctx = every closed-deal summary
pasted into the prompt, hindsight = similar deals recalled from Hindsight and ranked). `diff_briefs`
is the within-subject version: the same deal briefed before and after an outcome was recorded.

The comparison briefs are ordinary Brief rows tied to the comparison job, so nothing here touches
the learning signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from ...errors import PipelineError
from ...providers.base import LLMError
from ...providers.errors import explain_message
from .db import Brief, Deal, Job
from .jobs import JobFailed

if TYPE_CHECKING:
    from .context import V1Context

KIND = "compare_memory"
ARMS = ("none", "longctx", "hindsight")


def start_comparison(ctx: V1Context, deal_id: int, arms: list[str] | None = None) -> Job:
    chosen = list(dict.fromkeys(arms or ARMS))
    unknown = [a for a in chosen if a not in ARMS]
    if unknown or not chosen:
        raise PipelineError("invalid_request", f"arms must be some of {', '.join(ARMS)}.", 422)
    with ctx.db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        if deal.signals_status != "ready":
            raise PipelineError("signals_missing", "Read this deal first so the briefs can use its objections and promises.", 409)
    job = ctx.jobs.submit(KIND, deal_id, {"arms": chosen, "model": ctx.settings.model})
    with ctx.db.session() as session:
        row = session.get(Job, job.id)
        row.total = len(chosen)
        session.commit()
    return job


async def run_comparison(ctx: V1Context, job_id: int) -> None:
    from . import briefs, retrieval

    with ctx.db.session() as session:
        job = session.get(Job, job_id)
        deal_id, arms = job.target_id, list((job.payload or {}).get("arms") or ARMS)
        job.total = len(arms)
        session.commit()
    for arm in arms:
        with ctx.db.session() as session:
            have = session.scalar(
                select(Brief.id).where(Brief.job_id == job_id, Brief.mode == arm, Brief.status == "ready")
            )
        if have is None:
            try:
                recommendations = await retrieval.build_recommendations(ctx, deal_id, arm)
                brief = await briefs.generate_brief(ctx, deal_id, arm, recommendations=recommendations, job_id=job_id)
            except LLMError as exc:
                info = explain_message(exc.message, ctx.settings.provider, ctx.settings.model)
                detail = f"{info['title']}: {info['detail']}" if info else exc.message
                raise JobFailed(f"Stopped on the {arm} brief. {detail}") from exc
            if brief.status != "ready":
                raise JobFailed(f"The {arm} brief failed: {brief.error or 'unknown error'}")
        with ctx.db.session() as session:
            row = session.get(Job, job_id)
            row.done = arms.index(arm) + 1
            session.commit()


def _play_codes(brief: Brief) -> list[str]:
    return [step.get("play_code") for step in (brief.content or {}).get("next_steps", []) if step.get("play_code")]


def _cited_deals(brief: Brief) -> list[str]:
    content = brief.content or {}
    cited: list[str] = []
    for claim in content.get("memory", []):
        cited += [i for i in claim.get("source_ids", []) if i.startswith("D-")]
    for step in content.get("next_steps", []):
        cited += [i for i in step.get("source_ids", []) if i.startswith("D-")]
    return list(dict.fromkeys(cited))


def _avoided_play_recommended_by(by_arm: dict[str, Brief], plays: dict[str, list[str]]) -> list[str]:
    """Arms whose next steps include a play the hindsight arm's evidence says to avoid."""
    hindsight = by_arm.get("hindsight")
    if hindsight is None:
        return []
    avoided = {a.get("play_code") for a in (hindsight.evidence or {}).get("avoid", []) if a.get("play_code")}
    return [arm for arm in ARMS if avoided & set(plays.get(arm, []))]


def latest_comparison(ctx: V1Context, deal_id: int) -> dict[str, Any]:
    from .briefs import brief_view
    from .retrieval import n_closed

    with ctx.db.session() as session:
        job = session.scalars(
            select(Job).where(Job.kind == KIND, Job.target_id == deal_id).order_by(Job.id.desc()).limit(1)
        ).first()
        rows = session.scalars(select(Brief).where(Brief.job_id == job.id).order_by(Brief.id)).all() if job else []
        by_arm = {row.mode: row for row in rows if row.status == "ready"}
        job_view = None if job is None else {
            "id": job.id, "status": job.status, "done": job.done, "total": job.total, "error": job.error,
            "arms": (job.payload or {}).get("arms", list(ARMS)), "created_at": job.created_at,
        }
        plays = {arm: _play_codes(row) for arm, row in by_arm.items()}
        hashes = {row.prompt_hash for row in by_arm.values()}
        top_hindsight = plays.get("hindsight", [None])[:1]
        top_longctx = plays.get("longctx", [None])[:1]
        return {
            "job": job_view,
            "n_closed": n_closed(ctx.db),
            "arms": {arm: brief_view(row) for arm, row in by_arm.items()},
            "prompt_hash_same": len(hashes) <= 1,
            "differs": {
                "play_codes": plays,
                "same_play_as_longctx": bool(top_hindsight and top_longctx and top_hindsight == top_longctx),
                "cited_deals": _cited_deals(by_arm["hindsight"]) if "hindsight" in by_arm else [],
                "avoided_play_recommended_by": _avoided_play_recommended_by(by_arm, plays),
                "tokens": {arm: {"input": row.input_tokens, "output": row.output_tokens} for arm, row in by_arm.items()},
            },
        }


def _warning_key(w: dict) -> tuple[str, str | None]:
    return w.get("kind"), w.get("objection_type")


def diff_briefs(ctx: V1Context, from_id: int, to_id: int) -> dict[str, Any]:
    """What changed between two briefs of the same deal because the memory behind them changed."""
    with ctx.db.session() as session:
        a, b = session.get(Brief, from_id), session.get(Brief, to_id)
        if a is None or b is None:
            raise PipelineError("not_found", "One of the briefs does not exist.", 404)
        if a.deal_id != b.deal_id:
            raise PipelineError("invalid_request", "The two briefs belong to different deals.", 422)
        ea, eb = a.evidence or {}, b.evidence or {}
        wa = {_warning_key(w): w for w in ea.get("warnings", [])}
        wb = {_warning_key(w): w for w in eb.get("warnings", [])}
        changed = [
            {"kind": key[0], "objection_type": key[1],
             "from": {"similar_lost": wa[key].get("similar_lost"), "similar_total": wa[key].get("similar_total")},
             "to": {"similar_lost": wb[key].get("similar_lost"), "similar_total": wb[key].get("similar_total")}}
            for key in wb if key in wa and (wa[key].get("similar_lost"), wa[key].get("similar_total"))
            != (wb[key].get("similar_lost"), wb[key].get("similar_total"))
        ]
        sa = {s["code"] for s in ea.get("similar", [])}
        sb = {s["code"] for s in eb.get("similar", [])}
        pa, pb = set(_play_codes(a)), set(_play_codes(b))
        ra = {p["play_code"]: p for p in ea.get("plays", [])}
        rb = {p["play_code"]: p for p in eb.get("plays", [])}
        changed_plays = [
            {"play_code": code,
             "from": {"used": ra[code].get("used_in_similar"), "won": ra[code].get("won_in_similar")},
             "to": {"used": rb[code].get("used_in_similar"), "won": rb[code].get("won_in_similar")}}
            for code in rb if code in ra and (ra[code].get("used_in_similar"), ra[code].get("won_in_similar"))
            != (rb[code].get("used_in_similar"), rb[code].get("won_in_similar"))
        ]
        aa = {p["play_code"]: p for p in ea.get("avoid", [])}
        ab = {p["play_code"]: p for p in eb.get("avoid", [])}
        return {
            "from": from_id, "to": to_id,
            "added_warnings": [wb[k] for k in wb if k not in wa],
            "removed_warnings": [wa[k] for k in wa if k not in wb],
            "changed_warnings": changed,
            "added_similar": sorted(sb - sa),
            "removed_similar": sorted(sa - sb),
            "added_plays": sorted(pb - pa),
            "removed_plays": sorted(pa - pb),
            "changed_plays": changed_plays,
            "added_avoid": [ab[c] for c in ab if c not in aa],
            "removed_avoid": [aa[c] for c in aa if c not in ab],
            "changed_avoid": [
                {"play_code": code,
                 "from": {"used": aa[code].get("used_in_similar"), "won": aa[code].get("won_in_similar")},
                 "to": {"used": ab[code].get("used_in_similar"), "won": ab[code].get("won_in_similar")}}
                for code in ab if code in aa and (aa[code].get("used_in_similar"), aa[code].get("won_in_similar"))
                != (ab[code].get("used_in_similar"), ab[code].get("won_in_similar"))
            ],
            "memory_state": {"from": a.memory_state, "to": b.memory_state},
        }

