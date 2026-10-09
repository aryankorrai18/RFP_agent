"""HTTP API for deals, briefs, outcomes, memory and jobs. Contract: docs/API.md."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select

from ...config import PROVIDER_KEY_VARS, RETRIEVAL_MODES, has_key
from ...errors import PipelineError
from ...providers.errors import explain_message, provider_health
from . import assist, briefs, portfolio, deals, experiment, outcomes, quality, reflection, retrieval, signals
from .context import V1Context
from .db import (
    INTERACTION_KINDS, LOSS_REASONS, Brief, Deal, DealSignals, Interaction, Job, Lesson, MemoryEvent, Play, PlayStats,
    parse_deal_code,
)
from .memory import MemoryUnavailable

router = APIRouter(prefix="/v1", tags=["v1"])


async def get_v1(request: Request, x_workspace: str | None = Header(None)) -> V1Context:
    """The services for this request: the workspace named in the X-Workspace header (so a caller such as the hub
    can work in its own workspace without switching the one the app's screens show), else the active one."""
    if x_workspace and x_workspace.strip():
        return await request.app.state.context_for(x_workspace.strip())
    ctx = getattr(request.app.state, "v1", None)
    if ctx is None:
        raise PipelineError("v1_unavailable", "The deal service hasn't started.", 503)
    return ctx


# --- request bodies ---------------------------------------------------------------------------------


class NoteIn(BaseModel):
    kind: str = "email"
    text: str
    occurred_on: date | None = None
    author: str | None = None
    subject: str | None = None


class DealPatch(BaseModel):
    name: str | None = None
    account: str | None = None
    industry: str | None = None
    segment: str | None = None
    amount: int | None = None
    owner: str | None = None


class BriefIn(BaseModel):
    mode: str | None = None


class OutcomeIn(BaseModel):
    result: str
    loss_reason: str | None = None
    plays_used: list[str] | None = None  # omitted keeps the plays already recorded on the deal; [] clears them
    closed_on: date | None = None


class AskIn(BaseModel):
    question: str


class FollowupIn(BaseModel):
    kind: str = "email"
    play_code: str | None = None


class ComparisonIn(BaseModel):
    arms: list[str] | None = None


# --- helpers ----------------------------------------------------------------------------------------


def _job_view(job: Job | None, provider: str | None = None, model: str | None = None) -> dict | None:
    if job is None:
        return None
    used = (job.payload or {}).get("model") or model
    return {
        "id": job.id, "kind": job.kind, "target_id": job.target_id, "status": job.status, "done": job.done,
        "total": job.total, "error": job.error, "warning": job.warning, "payload": job.payload,
        "error_info": explain_message(job.error, provider, used),
    }


def _failing(view: dict) -> dict | None:
    """The explanation of a failing model, or None while it is fine or untested."""
    return view.get("explanation") if view.get("state") == "failing" else None


def _mode(ctx: V1Context, mode: str | None) -> str:
    chosen = mode or ctx.settings.retrieval_mode
    if chosen not in RETRIEVAL_MODES:
        raise PipelineError("invalid_request", f"mode must be one of {', '.join(RETRIEVAL_MODES)}.", 422)
    return chosen


async def _read_uploads(ctx: V1Context, files: list[UploadFile] | None) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    for upload in files or []:
        data = await upload.read(ctx.settings.max_upload_bytes + 1)
        if len(data) > ctx.settings.max_upload_bytes:
            raise PipelineError("file_too_large", f"{upload.filename} is over {ctx.settings.max_upload_mb} MB.", 413)
        if not data:
            raise PipelineError("empty_file", f"{upload.filename} is empty.", 422)
        out.append((upload.filename or "upload.txt", data))
    return out


def _detail(ctx: V1Context, deal_id: int) -> dict:
    detail = deals.deal_detail(ctx, deal_id)
    detail["flags"] = [flag.to_dict() for flag in signals.compute_flags(ctx.db, deal_id)] if detail["signals"] else []
    return detail


def _require_signals(ctx: V1Context, deal_id: int) -> None:
    with ctx.db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        if deal.signals_status != "ready":
            raise PipelineError("signals_missing", "Read this deal first so the brief can use its objections and promises.", 409)


# --- status and sync --------------------------------------------------------------------------------


@router.get("/usage")
async def usage(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """What this workspace has used: model calls and tokens (counted since tracking began) and the space its data takes."""
    from .usage import storage, summary

    return {"workspace": ctx.workspace_id, "model": summary(ctx.db), "storage": storage(ctx)}


@router.get("/status")
async def status(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    settings = ctx.settings
    with ctx.db.session() as session:
        open_deals = session.scalar(select(func.count()).select_from(Deal).where(Deal.status == "active", Deal.result == "open")) or 0
        won = session.scalar(select(func.count()).select_from(Deal).where(Deal.status == "active", Deal.result == "won")) or 0
        lost = session.scalar(select(func.count()).select_from(Deal).where(Deal.status == "active", Deal.result == "lost")) or 0
        interactions = session.scalar(select(func.count()).select_from(Interaction)) or 0
        plays = session.scalar(select(func.count()).select_from(Play)) or 0
        pending_deals = session.scalar(select(func.count()).select_from(Deal).where(Deal.hindsight_status.in_(("pending", "pending_delete")))) or 0
        pending_inter = session.scalar(
            select(func.count()).select_from(Interaction).where(Interaction.hindsight_status.in_(("pending", "pending_delete")))
        ) or 0
        lesson_counts = dict(session.execute(select(Lesson.hindsight_status, func.count()).group_by(Lesson.hindsight_status)).all())
        failed_lessons = session.scalar(select(func.count()).select_from(Lesson).where(Lesson.hindsight_error.is_not(None))) or 0
    cloud = "vectorize.io" in settings.hindsight_url
    if ctx.memory_backend == "local":
        cloud, healthy, mode = False, True, "local"
    else:
        healthy = await ctx.memory.healthy()
        can_read_bank = healthy and (bool(settings.hindsight_api_key) or not cloud)  # Cloud refuses bank reads without a key
        mode = await ctx.memory.extraction_mode() if can_read_bank else None
    return {
        "model": {
            "provider": settings.provider, "name": settings.model,
            "credentials": "env" if has_key(settings.provider) else "not_found_in_env",
            "key_variable": PROVIDER_KEY_VARS[settings.provider][0],
            "health": _failing(provider_health.view(settings.provider, settings.model)),
        },
        "retrieval_mode": settings.retrieval_mode,
        "hindsight": {
            "url": settings.hindsight_url, "cloud": cloud, "api_key_set": bool(settings.hindsight_api_key),
            "bank": settings.hindsight_bank, "lessons_bank": settings.hindsight_lessons_bank,
            "healthy": healthy, "mode": mode, "backend": ctx.memory_backend,
        },
        "lessons": {
            "enabled": ctx.lessons is not None,
            "pending": lesson_counts.get("pending", 0), "retained": lesson_counts.get("retained", 0),
            "failed": failed_lessons,
            "last_error": (ctx.last_lesson_sync.errors[-1] if ctx.last_lesson_sync and getattr(ctx.last_lesson_sync, "errors", None) else None),
        },
        "counts": {"open": open_deals, "closed": won + lost, "won": won, "lost": lost, "interactions": interactions, "plays": plays},
        "sync": {"pending": pending_deals + pending_inter},
    }


@router.post("/sync")
async def sync_now(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    report = await ctx.sync()
    lessons = ctx.last_lesson_sync
    return {
        "retained": report.retained, "deleted": report.deleted, "failed": report.failed, "pending": report.pending,
        "lessons": None if lessons is None else {
            "retained": getattr(lessons, "retained", 0), "failed": getattr(lessons, "failed", 0),
            "pending": getattr(lessons, "pending", 0),
        },
    }


# --- deals ------------------------------------------------------------------------------------------


@router.get("/plays")
async def plays(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    return {"plays": deals.list_plays(ctx)}


class PlayIn(BaseModel):
    code: str | None = None
    name: str
    description: str = ""
    category: str = "process"
    addresses: list[str] = []


class PlaysIn(BaseModel):
    plays: list[PlayIn]


@router.post("/plays")
async def save_plays(body: PlaysIn, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Add or update the company's own plays (no model call). Unknown objection types are reported, not stored."""
    if not body.plays or len(body.plays) > 100:
        raise PipelineError("invalid_request", "Send between 1 and 100 plays.", 422)
    return deals.save_plays(ctx, [p.model_dump() for p in body.plays])


@router.get("/deals")
async def list_deals(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    return {"deals": deals.list_deals(ctx)}


@router.post("/deals", status_code=201)
async def create_deal(
    name: str = Form(...),
    account: str = Form(...),
    industry: str | None = Form(None),
    segment: str | None = Form(None),
    amount: int | None = Form(None),
    owner: str | None = Form(None),
    stage: str = Form("discovery"),
    files: list[UploadFile] | None = File(None),
    ctx: V1Context = Depends(get_v1),
) -> dict[str, Any]:
    uploads = await _read_uploads(ctx, files)
    deal = deals.create_deal(
        ctx, name=name, account=account, industry=industry, segment=segment or None, amount=amount, owner=owner,
        stage=stage, files=uploads,
    )
    return _detail(ctx, deal.id)


@router.get("/deals/{deal_id}")
async def get_deal(deal_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    return _detail(ctx, deal_id)


@router.patch("/deals/{deal_id}")
async def update_deal(deal_id: int, body: DealPatch, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Free: change a deal's own details. Fields left out stay as they are."""
    deals.update_deal(ctx, deal_id, body.model_dump(exclude_none=True))
    return _detail(ctx, deal_id)


@router.delete("/deals/{deal_id}")
async def delete_deal(deal_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, bool]:
    deals.delete_deal(ctx, deal_id)
    return {"ok": True}


@router.post("/deals/{deal_id}/files")
async def add_files(deal_id: int, files: list[UploadFile] = File(...), ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    deals.add_files(ctx, deal_id, await _read_uploads(ctx, files))
    return _detail(ctx, deal_id)


@router.post("/deals/{deal_id}/notes")
async def add_note(deal_id: int, body: NoteIn, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    if body.kind not in INTERACTION_KINDS:
        raise PipelineError("invalid_request", f"kind must be one of {', '.join(INTERACTION_KINDS)}.", 422)
    deals.add_note(
        ctx, deal_id, kind=body.kind, text=body.text, occurred_on=body.occurred_on, author=body.author, subject=body.subject,
    )
    return _detail(ctx, deal_id)


@router.post("/deals/{deal_id}/signals", status_code=202)
async def read_deal(deal_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    job = signals.start_extraction(ctx, deal_id)
    return {"job_id": job.id}


# --- briefs and memory evidence ---------------------------------------------------------------------


@router.post("/deals/{deal_id}/brief", status_code=202)
async def start_brief(deal_id: int, body: BriefIn | None = None, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    mode = _mode(ctx, body.mode if body else None)
    _require_signals(ctx, deal_id)
    job = ctx.jobs.submit("brief", deal_id, {"mode": mode, "model": ctx.settings.model})
    return {"job_id": job.id}


@router.post("/deals/{deal_id}/ask")
async def ask_deal(deal_id: int, body: AskIn, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """One model call: answer a question from this deal's notes and the similar closed deals. Nothing is stored."""
    return await assist.ask_deal(ctx, deal_id, body.question)


@router.post("/portfolio/ask")
async def ask_portfolio(body: AskIn, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """One model call: answer a question about the whole pipeline from counts computed in code and the deal summaries.
    Nothing is stored."""
    return await portfolio.ask_portfolio(ctx, body.question)


@router.post("/deals/{deal_id}/followup")
async def draft_followup(deal_id: int, body: FollowupIn | None = None, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """One model call: draft an email or call agenda for the latest brief's next step. Nothing is stored."""
    body = body or FollowupIn()
    return await assist.draft_followup(ctx, deal_id, body.kind, body.play_code)


@router.get("/deals/{deal_id}/brief")
async def get_brief(deal_id: int, mode: str | None = Query(None), ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    if mode is not None:
        _mode(ctx, mode)
    brief = briefs.latest_brief(ctx.db, deal_id, mode)
    if brief is None:
        raise PipelineError("no_brief", "No brief has been made for this deal yet.", 404)
    return briefs.brief_view(brief)


@router.get("/deals/{deal_id}/briefs")
async def list_briefs(deal_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    with ctx.db.session() as session:
        compare_jobs = select(Job.id).where(Job.kind == experiment.KIND)
        rows = session.scalars(
            select(Brief).where(Brief.deal_id == deal_id, Brief.status == "ready", Brief.job_id.not_in(compare_jobs) | Brief.job_id.is_(None))
            .order_by(Brief.id.desc())
        ).all()
        items = []
        for row in rows:
            view = briefs.brief_view(row)
            view.pop("content", None)
            view["n_similar"] = len((row.evidence or {}).get("similar", []))
            items.append(view)
    return {"briefs": items}


@router.get("/deals/{deal_id}/recommendations")
async def get_recommendations(deal_id: int, mode: str | None = Query(None), ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    _require_signals(ctx, deal_id)
    recommendations = await retrieval.build_recommendations(ctx, deal_id, _mode(ctx, mode))
    return recommendations.to_dict()


@router.get("/deals/{deal_id}/brief-diff")
async def brief_diff(
    deal_id: int, from_: int = Query(..., alias="from"), to: int = Query(...), ctx: V1Context = Depends(get_v1)
) -> dict[str, Any]:
    return experiment.diff_briefs(ctx, from_, to)


@router.get("/deals/{deal_id}/memory-says")
async def get_memory_says(deal_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """The cached Reflect text about deals like this one. Reading it costs nothing."""
    return reflection.get_reflection(ctx, deal_id)


@router.post("/deals/{deal_id}/memory-says")
async def refresh_memory_says(deal_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Ask the lessons bank again (spends Hindsight credits on the Cloud backend, nothing on local)."""
    return await reflection.refresh_reflection(ctx, deal_id)


# --- outcome (the learning loop) --------------------------------------------------------------------


@router.put("/deals/{deal_id}/outcome")
async def put_outcome(deal_id: int, body: OutcomeIn, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    if body.result == "lost" and body.loss_reason not in LOSS_REASONS:
        raise PipelineError("invalid_request", f"loss_reason must be one of {', '.join(LOSS_REASONS)}.", 422)
    recorded = outcomes.record_outcome(
        ctx, deal_id, body.result, body.loss_reason, body.plays_used, body.closed_on or date.today()
    )
    with ctx.db.session() as session:
        stats = {s.play_code: s for s in session.scalars(select(PlayStats)).all()}
        events = session.scalars(select(MemoryEvent).where(MemoryEvent.deal_id == deal_id).order_by(MemoryEvent.id.desc()).limit(6)).all()
    credit = [
        {"play_code": code, "delta": delta, "times_used": stats[code].times_used, "won": stats[code].won,
         "lost_quality": stats[code].lost_quality, "lost_other": stats[code].lost_other}
        for code, delta in recorded["credit_applied"].items() if code in stats
    ]
    note = ("Quality-related loss: the plays used lose credit." if recorded["credit"] < 0 else
            "Won: the plays used gain credit." if recorded["credit"] > 0 else
            "This reason is not about the plays used, so they get no credit either way.")
    return {
        "deal": deals.deal_row_by_id(ctx, deal_id), "credit": credit, "events": [e.detail for e in reversed(events)],
        "lessons_added": recorded["lessons_changed"], "memory_note": note, "plays_used": recorded["plays_used"],
    }


# --- before / after ---------------------------------------------------------------------------------


@router.post("/deals/{deal_id}/comparison", status_code=202)
async def start_comparison(deal_id: int, body: ComparisonIn | None = None, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    job = experiment.start_comparison(ctx, deal_id, body.arms if body else None)
    return {"job_id": job.id, "planned_calls": (job.payload or {}).get("arms") and len(job.payload["arms"]) or 3}


@router.get("/deals/{deal_id}/comparison")
async def get_comparison(deal_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    return experiment.latest_comparison(ctx, deal_id)


# --- memory page ------------------------------------------------------------------------------------


@router.get("/memory/journal")
async def memory_journal(limit: int = Query(80, ge=1, le=500), ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    with ctx.db.session() as session:
        rows = session.scalars(select(MemoryEvent).order_by(MemoryEvent.id.desc()).limit(limit)).all()
        codes = {d.id: d.code for d in session.scalars(select(Deal)).all()}
        return {"events": [
            {"id": r.id, "kind": r.kind, "deal_code": codes.get(r.deal_id), "play_code": r.play_code, "detail": r.detail,
             "created_at": r.created_at}
            for r in rows
        ]}


@router.get("/memory/lessons")
async def memory_lessons(limit: int = Query(120, ge=1, le=500), ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    with ctx.db.session() as session:
        rows = session.scalars(select(Lesson).order_by(Lesson.happened_at.desc(), Lesson.id.desc()).limit(limit)).all()
        counts = dict(session.execute(select(Lesson.hindsight_status, func.count()).group_by(Lesson.hindsight_status)).all())
        codes = {d.id: d.code for d in session.scalars(select(Deal)).all()}
        return {
            "lessons": [
                {"key": r.key, "signal": r.signal, "text": r.text, "tags": r.tags, "deal_code": codes.get(r.deal_id),
                 "hindsight_status": r.hindsight_status, "happened_at": r.happened_at}
                for r in rows
            ],
            "counts": {"pending": counts.get("pending", 0), "retained": counts.get("retained", 0),
                       "failed": sum(1 for r in rows if r.hindsight_error)},
        }


@router.get("/memory/quality")
async def memory_quality(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Leave-one-out self-check on the closed deals. No model and no Hindsight call."""
    return await quality.memory_quality(ctx)


@router.get("/memory/play-stats")
async def memory_play_stats(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    with ctx.db.session() as session:
        stats = {s.play_code: s for s in session.scalars(select(PlayStats)).all()}
        return {"plays": [
            {"code": p.code, "name": p.name, "category": p.category,
             "times_used": stats[p.code].times_used if p.code in stats else 0,
             "won": stats[p.code].won if p.code in stats else 0,
             "lost_quality": stats[p.code].lost_quality if p.code in stats else 0,
             "lost_other": stats[p.code].lost_other if p.code in stats else 0,
             "outcome_credit": round(stats[p.code].outcome_credit, 2) if p.code in stats else 0.0}
            for p in session.scalars(select(Play).order_by(Play.code)).all()
        ]}


@router.get("/memory/history")
async def memory_history(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    from .memory import deal_summary_item

    with ctx.db.session() as session:
        plays_by_code = {p.code: p.name for p in session.scalars(select(Play)).all()}
        closed = session.scalars(
            select(Deal).where(Deal.status == "active", Deal.result != "open").order_by(Deal.closed_on.desc(), Deal.id.desc())
        ).all()
        items = []
        for deal in closed:
            signals_row: DealSignals | None = deal.signals
            items.append({
                **deals.deal_row(deal),
                "summary": deal_summary_item(deal, signals_row, deal.stakeholders, plays_by_code).content,
                "plays_used": [{"code": c, "name": plays_by_code.get(c, c)} for c in (signals_row.plays_used if signals_row else [])],
                "objections": [{"type": o.get("type"), "status": o.get("status")} for o in (signals_row.objections if signals_row else [])],
            })
        return {"deals": items}


@router.get("/memory/playbook")
async def memory_playbook(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """The Hindsight mental model "what wins, what loses". Reading uses no Hindsight credits."""
    if ctx.lessons is None:
        return {"content": None, "last_refreshed_at": None, "is_stale": False, "state": "unavailable"}
    try:
        playbook = await ctx.lessons.playbook(refresh=False)
    except MemoryUnavailable:
        return {"content": None, "last_refreshed_at": None, "is_stale": False, "state": "unavailable"}
    if not playbook:
        return {"content": None, "last_refreshed_at": None, "is_stale": False, "state": "missing"}
    return {**playbook, "state": "ready"}


@router.post("/memory/playbook/refresh")
async def refresh_playbook(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Create or refresh the playbook mental model (uses Hindsight credits)."""
    if ctx.lessons is None:
        raise PipelineError("lessons_disabled", "The Hindsight lessons bank is turned off (DEAL_LESSONS).", 409)
    await ctx.sync_lessons()
    try:
        playbook = await ctx.lessons.playbook(refresh=True)
    except MemoryUnavailable as exc:
        raise PipelineError("hindsight_unavailable", str(exc), 503) from exc
    return {**(playbook or {"content": None, "last_refreshed_at": None, "is_stale": False}),
            "state": "ready" if playbook else "missing"}


# --- jobs -------------------------------------------------------------------------------------------


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    return _job_view(ctx.jobs.cancel(job_id), ctx.settings.provider, ctx.settings.model)


@router.get("/jobs/{job_id}")
async def get_job(job_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    with ctx.db.session() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise PipelineError("not_found", f"No job {job_id}.", 404)
        return _job_view(job, ctx.settings.provider, ctx.settings.model)


__all__ = ["router", "parse_deal_code"]
