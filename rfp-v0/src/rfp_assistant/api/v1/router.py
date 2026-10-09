"""HTTP API for projects, the answer library, memory, review, and export."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import func, select

from ... import workspaces
from ...config import PROVIDER_KEY_VARS, ROOT, has_key
from ...providers.errors import explain_message, provider_health
from ...schemas import FACT_ID_PATTERN, FactOut
from ...errors import PipelineError, load_fact_sheet
from . import demo, library, outcomes, projects
from .context import V1Context
from .db import (
    Answer, AnswerStats, Debrief, Job, MemoryEvent, PastProposal, Project, ProjectOutcome,
    RequirementRow, ReviewFeedback, parse_answer_code, Lesson, ProjectBrief, utcnow,
)
from .export import FinalAnswer, export_docx, export_xlsx
from . import experiment, judge
from ...providers.model_choice import MODEL_NAME
from .curve import learning_curve
from .lessons import brief_query, brief_tags, pending_lessons
from .memory import MemoryUnavailable
from .sync import pending_count

router = APIRouter(prefix="/v1", tags=["v1"])


def get_v1(request: Request) -> V1Context:
    ctx = getattr(request.app.state, "v1", None)
    if ctx is None:
        raise PipelineError("v1_unavailable", "The library service hasn't started.", 503)
    return ctx


# --- views ------------------------------------------------------------------------------------


class JobView(BaseModel):
    id: int
    kind: str
    target_id: int
    status: str
    done: int
    total: int
    error: str | None
    warning: str | None
    error_info: dict | None = None  # what a model error means and what to do (provider_errors.explain)


class PairView(BaseModel):
    id: int
    order: int
    section: str | None
    reference: str | None
    question: str
    answer: str
    decision: str


class ProposalView(BaseModel):
    id: int
    filename: str
    client: str | None
    industry: str | None
    submitted_on: date | None
    result: str
    loss_reason: str | None
    status: str
    error: str | None
    created_at: datetime
    job: JobView | None = None
    pairs: list[PairView] = []
    error_info: dict | None = None


class AnswerView(BaseModel):
    id: str
    question: str
    answer: str
    source: str
    client: str | None
    industry: str | None
    updated_at: datetime
    status: str
    hindsight_status: str
    hindsight_error: str | None
    retrieval: dict | None = None
    track_record: dict | None = None
    outcome: dict | None = None  # {"result": "won"|"lost"|..., "loss_reason": str|None} of the source proposal/project


class ReviewView(BaseModel):
    action: str
    final_text: str | None
    edit_distance: float | None
    created_at: datetime
    reason_tags: list[str] = []
    rating: int | None = None


class RetrievedView(BaseModel):
    id: str
    position: int | None = None
    rank: int | None = None
    final: float | None = None
    semantic: float | None = None
    reranker: float | None = None
    outcome_score: float | None = None
    relevance: float | None = None
    quality: float | None = None
    freshness: float | None = None
    context: float | None = None
    mode: str | None = None
    lessons: float | None = None  # Hindsight lesson factor (hindsight mode)
    lesson_evidence: list[str] = []  # what the lessons bank recalled about this answer
    reasons: list[str] = []
    question: str | None = None
    answer: str | None = None
    client: str | None = None
    live: bool = False


class DraftView(BaseModel):
    id: int
    version: int
    status: str
    answer: str
    claims: list[dict]
    sources: list[str]
    unsupported_claims: list[str]
    invalid_citations: list[str]
    flags: list[str]
    sme_question: str | None
    word_count: int
    retrieved: list[RetrievedView]
    retrieval_warning: str | None
    instructions: str | None
    prompt_version: str
    model: str | None
    error: str | None
    created_at: datetime
    reviews: list[ReviewView]
    error_info: dict | None = None


class RequirementView(BaseModel):
    id: int
    code: str
    order: int
    section: str | None
    reference: str | None
    question: str
    mandatory: bool | None
    word_limit: int | None
    final: bool
    final_text: str | None
    draft: DraftView | None
    draft_count: int


class ProjectStats(BaseModel):
    requirements: int
    drafted: int
    final: int
    needs_sme: int
    flagged: int
    failed: int


class ProjectView(BaseModel):
    id: int
    name: str
    client: str | None
    industry: str | None
    state: str
    error: str | None
    filename: str
    file_kind: str
    error_info: dict | None = None
    company: str | None = None
    fact_sheet_filename: str | None = None
    created_at: datetime
    job: JobView | None
    stats: ProjectStats
    requirements: list[RequirementView]
    facts: list[FactOut]
    outcome: dict | None = None


class ProjectSummary(BaseModel):
    id: int
    name: str
    client: str | None
    state: str
    filename: str
    created_at: datetime
    requirement_count: int
    job: JobView | None = None


class PairDecisionIn(BaseModel):
    id: int
    decision: str
    question: str | None = None
    answer: str | None = None


class ConfirmIn(BaseModel):
    pairs: list[PairDecisionIn]


class RequirementIn(BaseModel):
    question: str
    section: str | None = None
    reference: str | None = None
    mandatory: bool | None = None
    word_limit: int | None = None


class RequirementsIn(BaseModel):
    requirements: list[RequirementIn]


class DraftIn(BaseModel):
    requirement_ids: list[int] | None = None


class ReviewIn(BaseModel):
    action: str
    final_text: str | None = None
    reason_tags: list[str] = []
    rating: int | None = None


class RegenerateIn(BaseModel):
    instructions: str | None = None


class SupersedeIn(BaseModel):
    replacement_id: str


class OutcomeIn(BaseModel):
    result: str
    loss_reason: str | None = None
    decided_at: date | None = None


class DebriefItemIn(BaseModel):
    section: str | None = None
    score: float | None = None
    comment: str | None = None


class DebriefIn(BaseModel):
    items: list[DebriefItemIn]


# --- helpers ----------------------------------------------------------------------------------


def _job_view(job: Job | None, provider: str | None = None, model: str | None = None) -> JobView | None:
    if job is None:
        return None
    used = (job.payload or {}).get("judge_model") or (job.payload or {}).get("model") or model
    return JobView(
        id=job.id, kind=job.kind, target_id=job.target_id, status=job.status,
        done=job.done, total=job.total, error=job.error, warning=job.warning,
        error_info=explain_message(job.error, provider, used),
    )


def _latest_job(session, kind: str, target_id: int) -> Job | None:  # noqa: ANN001
    return session.scalars(
        select(Job).where(Job.kind == kind, Job.target_id == target_id).order_by(Job.id.desc())
    ).first()


def _stats_view(stats: AnswerStats | None) -> dict | None:
    if stats is None:
        return None
    return {
        "times_used": stats.times_used, "accepted": stats.accepted,
        "light_edits": stats.light_edits, "length_edits": stats.length_edits,
        "heavy_edits": stats.heavy_edits, "rewritten": stats.rewritten, "rejected": stats.rejected,
        "average_rating": round(stats.rating_total / stats.rating_count, 2) if stats.rating_count else None,
        "debrief_credit": stats.debrief_credit, "outcome_credit": stats.outcome_credit,
        "suggested_supersede": stats.suggested_supersede,
    }


def _answer_view(answer: Answer, retrieval: dict | None = None, stats: AnswerStats | None = None,
                 outcome: dict | None = None) -> AnswerView:
    return AnswerView(
        id=answer.code, question=answer.question, answer=answer.answer, source=answer.source,
        client=answer.client, industry=answer.industry, updated_at=answer.updated_at,
        status=answer.status, hindsight_status=answer.hindsight_status,
        hindsight_error=answer.hindsight_error, retrieval=retrieval, track_record=_stats_view(stats),
        outcome=outcome,
    )


def _answer_outcomes(session, answers: list[Answer]) -> dict[int, dict]:  # noqa: ANN001
    """The won/lost result of each answer's source: its past proposal, or its project's recorded
    outcome. So a reviewer browsing the library can see which answers came from a win.

    Several answers usually share one past proposal (or project), so this maps proposal/project id
    to its result first, then looks each answer's own id up in that — not the other way round, which
    would let one answer's result silently overwrite another's for the same proposal."""
    proposal_ids = {a.past_proposal_id for a in answers if a.past_proposal_id}
    project_ids = {a.project_id for a in answers if a.project_id}
    proposal_result: dict[int, dict] = {}
    if proposal_ids:
        for proposal in session.scalars(select(PastProposal).where(PastProposal.id.in_(proposal_ids))):
            if proposal.result != "unknown":
                proposal_result[proposal.id] = {"result": proposal.result, "loss_reason": proposal.loss_reason}
    project_result: dict[int, dict] = {}
    if project_ids:
        for outcome in session.scalars(select(ProjectOutcome).where(ProjectOutcome.project_id.in_(project_ids))):
            project_result[outcome.project_id] = {"result": outcome.result, "loss_reason": outcome.loss_reason}
    out: dict[int, dict] = {}
    for a in answers:
        if a.past_proposal_id in proposal_result:
            out[a.id] = proposal_result[a.past_proposal_id]
        elif a.project_id in project_result:
            out[a.id] = project_result[a.project_id]
    return out


def _proposal_view(session, proposal: PastProposal, with_pairs: bool = False,  # noqa: ANN001
                   provider: str | None = None, model: str | None = None) -> ProposalView:
    return ProposalView(
        id=proposal.id, filename=proposal.document.filename, client=proposal.client,
        industry=proposal.industry, submitted_on=proposal.submitted_on, result=proposal.result,
        loss_reason=proposal.loss_reason, status=proposal.status, error=proposal.error,
        created_at=proposal.created_at,
        job=_job_view(_latest_job(session, "extract_pairs", proposal.id), provider, model),
        error_info=explain_message(proposal.error, provider, model),
        pairs=[
            PairView(id=p.id, order=p.order, section=p.section, reference=p.reference,
                     question=p.question, answer=p.answer, decision=p.decision)
            for p in proposal.pairs
        ] if with_pairs else [],
    )


def _retrieved_views(session, retrieved: list[dict]) -> list[RetrievedView]:  # noqa: ANN001
    ids = [parse_answer_code(r.get("id", "")) for r in retrieved]
    answers = {a.id: a for a in session.scalars(select(Answer).where(Answer.id.in_([i for i in ids if i])))}
    views = []
    for item, answer_id in zip(retrieved, ids):
        answer = answers.get(answer_id)
        views.append(
            RetrievedView(
                id=item.get("id", ""), position=item.get("position"), rank=item.get("rank"),
                final=item.get("final"), semantic=item.get("semantic"), reranker=item.get("reranker"),
                outcome_score=item.get("outcome_score"), relevance=item.get("relevance"),
                quality=item.get("quality"), freshness=item.get("freshness"), context=item.get("context"),
                mode=item.get("mode"), lessons=item.get("lessons"),
                lesson_evidence=item.get("lesson_evidence") or [],
                reasons=item.get("reasons") or [],
                question=answer.question if answer else None, answer=answer.answer if answer else None,
                client=answer.client if answer else None, live=bool(answer and answer.live),
            )
        )
    return views


def _requirement_view(session, requirement: RequirementRow, provider: str | None = None) -> RequirementView:  # noqa: ANN001
    draft = requirement.drafts[-1] if requirement.drafts else None
    draft_view = None
    if draft is not None:
        feedback = {
            review.id: session.get(ReviewFeedback, review.id) for review in draft.reviews
        }
        draft_view = DraftView(
            id=draft.id, version=draft.version, status=draft.status, answer=draft.answer,
            claims=draft.claims or [], sources=draft.sources or [],
            unsupported_claims=draft.unsupported_claims or [], invalid_citations=draft.invalid_citations or [],
            flags=draft.flags or [], sme_question=draft.sme_question, word_count=draft.word_count,
            retrieved=_retrieved_views(session, draft.retrieved or []),
            retrieval_warning=draft.retrieval_warning, instructions=draft.instructions,
            prompt_version=draft.prompt_version, model=draft.model, error=draft.error,
            error_info=explain_message(draft.error, provider, draft.model),
            created_at=draft.created_at,
            reviews=[ReviewView(
                action=r.action, final_text=r.final_text, edit_distance=r.edit_distance, created_at=r.created_at,
                reason_tags=feedback[r.id].reason_tags if feedback[r.id] else [],
                rating=feedback[r.id].rating if feedback[r.id] else None,
            ) for r in draft.reviews],
        )
    return RequirementView(
        id=requirement.id, code=requirement.code, order=requirement.order, section=requirement.section,
        reference=requirement.reference, question=requirement.question, mandatory=requirement.mandatory,
        word_limit=requirement.word_limit, final=projects.is_final(requirement),
        final_text=projects.final_text(requirement), draft=draft_view, draft_count=len(requirement.drafts),
    )


def _project_view(ctx: V1Context, project_id: int) -> ProjectView:
    with ctx.db.session() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise PipelineError("not_found", f"No project {project_id}.", 404)
        provider, model = ctx.settings.provider, ctx.settings.model
        requirements = [_requirement_view(session, r, provider) for r in project.requirements]
        job = _latest_job(session, "draft_all", project.id) or _latest_job(session, "extract_requirements", project.id)
        drafts = [r.draft for r in requirements if r.draft]
        stats = ProjectStats(
            requirements=len(requirements),
            drafted=len(drafts),
            final=sum(r.final for r in requirements),
            needs_sme=sum(d.status == "needs_sme" for d in drafts),
            flagged=sum(bool(d.flags) for d in drafts),
            failed=sum(d.status == "failed" for d in drafts),
        )
        filename = project.document.filename
        outcome = session.get(ProjectOutcome, project.id)
        view = ProjectView(
            id=project.id, name=project.name, client=project.client, industry=project.industry,
            state=project.state, error=project.error, filename=filename,
            error_info=explain_message(project.error, provider, model),
            file_kind=filename.rsplit(".", 1)[-1].lower(), created_at=project.created_at,
            job=_job_view(job, provider, model), stats=stats, requirements=requirements, facts=[],
            outcome={
                "result": outcome.result, "loss_reason": outcome.loss_reason,
                "decided_at": outcome.decided_at.isoformat() if outcome.decided_at else None,
            } if outcome else None,
        )
    try:
        sheet, fact_sheet_filename = projects.project_fact_sheet(ctx, project_id)
        today = date.today()
        view.company = sheet.company
        view.fact_sheet_filename = fact_sheet_filename
        view.facts = [FactOut(id=f.id, topic=f.topic, statement=f.statement) for f in sheet.facts if f.is_live(today)]
    except PipelineError:
        view.facts = []
    return view


async def _read_upload(ctx: V1Context, upload: UploadFile) -> bytes:
    limit = ctx.settings.max_upload_bytes
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise PipelineError("file_too_large", f"The file is larger than the {ctx.settings.max_upload_mb} MB limit.", 413)
    return data


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PipelineError("invalid_request", "submitted_on must be a date like 2025-11-30.", 422) from exc


# --- status -----------------------------------------------------------------------------------


def _workspace_summary(ctx: V1Context) -> dict[str, Any]:
    """Which workspace is open and how far it has got: what the page needs for its first-run steps."""
    with ctx.db.session() as session:
        answers = session.scalar(select(func.count()).select_from(Answer).where(Answer.status == "approved")) or 0
        project_count = session.scalar(select(func.count()).select_from(Project)) or 0
        proposal_count = session.scalar(
            select(func.count()).select_from(PastProposal).where(PastProposal.status != "discarded")
        ) or 0
    space = workspaces.active()
    return {
        "id": space.id if space else None, "name": space.name if space else None, "kind": space.kind if space else None,
        "company_set_up": ctx.settings.fact_sheet_path.exists(), "fact_sheet_locked": _fact_sheet_locked(ctx),
        "projects": project_count, "past_proposals": proposal_count, "answers": answers,
    }


@router.get("/workspace")
async def workspace_summary(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """The open workspace, without the Hindsight checks /status makes."""
    return _workspace_summary(ctx)


@router.get("/status")
async def status(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    settings = ctx.settings
    with ctx.db.session() as session:
        answers = session.scalar(select(func.count()).select_from(Answer).where(Answer.status == "approved")) or 0
    cloud = "vectorize.io" in settings.hindsight_url
    healthy = await ctx.memory.healthy()
    # Without a key, Cloud answers the version check but refuses bank reads (after retries), so don't ask.
    can_read_bank = healthy and (bool(settings.hindsight_api_key) or not cloud)
    mode = await ctx.memory.extraction_mode() if can_read_bank else None
    return {
        "product_version": "v4-offline",
        "provider": settings.provider,
        "model": settings.model,
        "retrieval_mode": settings.retrieval_mode,
        "evidence_check": "lexical" if settings.evidence_check else "off",
        "llm_credentials": "env" if has_key(settings.provider) else "not_found_in_env",
        "key_variable": PROVIDER_KEY_VARS[settings.provider][0],
        "llm_health": provider_health.view(settings.provider, settings.model),
        "hindsight": {
            "url": settings.hindsight_url,
            "cloud": cloud,
            "api_key_set": bool(settings.hindsight_api_key),
            "bank": settings.hindsight_bank,
            "healthy": healthy,
            "extraction_mode": mode,
            "baseline_ok": mode == "chunks",  # V1-D7: no LLM extraction
        },
        "library": {"answers": answers, "pending_sync": pending_count(ctx.db)},
        "workspace": _workspace_summary(ctx),
        "lessons": {
            "enabled": ctx.lessons is not None,
            "bank": settings.hindsight_lessons_bank,
            "pending": pending_lessons(ctx.db) if ctx.lessons is not None else 0,
            "last_error": (ctx.last_lesson_sync.errors[-1]
                           if ctx.last_lesson_sync and ctx.last_lesson_sync.errors else None),
        },
    }


@router.post("/sync")
async def sync_now(ctx: V1Context = Depends(get_v1)) -> dict[str, int]:
    report = await ctx.sync()
    return {"retained": report.retained, "deleted": report.deleted, "failed": report.failed, "pending": report.pending}


# --- library ----------------------------------------------------------------------------------


@router.post("/library/demo-packs/virtusa-cyber")
async def load_virtusa_cyber_demo(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Load prepared synthetic evidence into the active workspace without a Gemini call."""
    return demo.seed_virtusa_cyber(ctx)


@router.post("/library", response_model=ProposalView, status_code=202)
async def upload_past_proposal(
    file: UploadFile = File(...),
    client: str | None = Form(None),
    industry: str | None = Form(None),
    submitted_on: str | None = Form(None),
    result: str = Form("unknown"),
    loss_reason: str | None = Form(None),
    ctx: V1Context = Depends(get_v1),
) -> ProposalView:
    data = await _read_upload(ctx, file)
    proposal, _job = library.create_past_proposal(
        ctx, filename=file.filename or "upload", data=data, client=client, industry=industry,
        submitted_on=_parse_date(submitted_on), result=result, loss_reason=loss_reason,
    )
    with ctx.db.session() as session:
        return _proposal_view(session, session.get(PastProposal, proposal.id))


@router.get("/library/proposals", response_model=list[ProposalView])
async def list_proposals(ctx: V1Context = Depends(get_v1)) -> list[ProposalView]:
    with ctx.db.session() as session:
        rows = session.scalars(
            select(PastProposal).where(PastProposal.status != "discarded").order_by(PastProposal.id.desc())
        ).all()
        return [_proposal_view(session, p) for p in rows]


@router.get("/library/proposals/{proposal_id}", response_model=ProposalView)
async def get_proposal(proposal_id: int, ctx: V1Context = Depends(get_v1)) -> ProposalView:
    with ctx.db.session() as session:
        proposal = session.get(PastProposal, proposal_id)
        if proposal is None:
            raise PipelineError("not_found", f"No past proposal {proposal_id}.", 404)
        return _proposal_view(session, proposal, with_pairs=True)


@router.post("/library/proposals/{proposal_id}/discard", response_model=ProposalView)
async def discard(proposal_id: int, ctx: V1Context = Depends(get_v1)) -> ProposalView:
    library.discard_proposal(ctx, proposal_id)
    with ctx.db.session() as session:
        return _proposal_view(session, session.get(PastProposal, proposal_id))


@router.post("/library/proposals/{proposal_id}/confirm")
async def confirm(proposal_id: int, body: ConfirmIn, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    created = library.confirm_pairs(
        ctx, proposal_id,
        [library.PairDecision(pair_id=p.id, decision=p.decision, question=p.question, answer=p.answer) for p in body.pairs],
    )
    return {"created": [a.code for a in created]}


@router.get("/library/answers")
async def list_answers(
    query: str | None = Query(None), limit: int = Query(50, ge=1, le=200), ctx: V1Context = Depends(get_v1)
) -> dict[str, Any]:
    rows, retrieved, warning = await library.search_answers(ctx, query, limit)
    by_code = {r["id"]: r for r in retrieved}
    with ctx.db.session() as session:
        stats = {
            row.answer_id: row for row in session.scalars(
                select(AnswerStats).where(AnswerStats.answer_id.in_([a.id for a in rows]))
            )
        }
        outcomes_by_id = _answer_outcomes(session, rows)
    return {
        "answers": [_answer_view(a, by_code.get(a.code), stats.get(a.id), outcomes_by_id.get(a.id)) for a in rows],
        "warning": warning,
    }


@router.delete("/library/answers/{code}")
async def remove_answer(code: str, ctx: V1Context = Depends(get_v1)) -> dict[str, str]:
    answer = library.delete_answer(ctx, code)
    return {"deleted": answer.code}


@router.post("/library/answers/{code}/supersede")
async def supersede(code: str, body: SupersedeIn, ctx: V1Context = Depends(get_v1)) -> dict[str, str]:
    answer = outcomes.supersede_answer(ctx, code, body.replacement_id)
    return {"superseded": answer.code, "replacement": body.replacement_id}


@router.get("/memory/learned")
async def memory_learned(
    limit: int = Query(100, ge=1, le=500), ctx: V1Context = Depends(get_v1)
) -> dict[str, Any]:
    with ctx.db.session() as session:
        events = session.scalars(select(MemoryEvent).order_by(MemoryEvent.id.desc()).limit(limit)).all()
    return {"events": [
        {
            "id": event.id, "kind": event.kind,
            "answer_id": f"ANS-{event.answer_id:04d}" if event.answer_id else None,
            "project_id": event.project_id, "detail": event.detail, "created_at": event.created_at,
        }
        for event in events
    ]}


# --- company facts (the workspace's fact sheet) ---------------------------------------------------

BUNDLED_FACT_SHEET = ROOT / "data" / "fact_sheet.json"


class FactIn(BaseModel):
    id: str | None = None  # kept when given, so drafts that cited FACT-003 keep meaning the same fact
    topic: str = ""
    statement: str
    valid_to: date | None = None


class CompanyIn(BaseModel):
    company: str
    facts: list[FactIn]


def _fact_sheet_locked(ctx: V1Context) -> bool:
    """The bundled sample workspace is read-only; company workspaces own editable fact sheets."""
    return ctx.settings.fact_sheet_path.resolve() == BUNDLED_FACT_SHEET.resolve()


@router.get("/company")
async def get_company(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    path = ctx.settings.fact_sheet_path
    locked = _fact_sheet_locked(ctx)
    if not path.exists():
        return {"company": None, "facts": [], "locked": locked, "set_up": False}
    sheet = load_fact_sheet(path.read_bytes(), path.name)
    return {"company": sheet.company, "facts": [f.model_dump(mode="json") for f in sheet.facts], "locked": locked, "set_up": True}


@router.put("/company")
async def put_company(body: CompanyIn, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Save the company name and official facts. Drafts cite these as FACT-001, FACT-002..."""
    if _fact_sheet_locked(ctx):
        raise PipelineError(
            "fact_sheet_locked",
            "This workspace uses the bundled read-only sample fact sheet. "
            "Start a new workspace to set up your own company.", 409,
        )
    facts = [f for f in body.facts if f.statement.strip()]
    used = {f.id for f in facts if f.id and FACT_ID_PATTERN.match(f.id)}
    next_number = max((int(i.split("-")[1]) for i in used), default=0) + 1
    seen: set[str] = set()
    out = []
    for fact in facts:
        fact_id = fact.id if fact.id and FACT_ID_PATTERN.match(fact.id) and fact.id not in seen else None
        if fact_id is None:
            fact_id, next_number = f"FACT-{next_number:03d}", next_number + 1
        seen.add(fact_id)
        out.append({"id": fact_id, "topic": fact.topic.strip(), "statement": fact.statement.strip(),
                    **({"valid_to": fact.valid_to.isoformat()} if fact.valid_to else {})})
    raw = json.dumps({"company": body.company, "facts": out}, indent=2)
    sheet = load_fact_sheet(raw, "company facts")  # same validation as an uploaded fact sheet
    path = ctx.settings.fact_sheet_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(raw, encoding="utf-8")
    tmp.replace(path)
    return {"company": sheet.company, "facts": [f.model_dump(mode="json") for f in sheet.facts], "locked": False, "set_up": True}


@router.get("/memory/learning-curve")
async def memory_learning_curve(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Per-project review outcomes, oldest first, counted from real review history."""
    with ctx.db.session() as session:
        return learning_curve(session)


class ComparisonIn(BaseModel):
    arms: list[str] | None = None  # default: plain (no lessons) vs hindsight (with lessons)


@router.post("/projects/{project_id}/comparison", status_code=202)
async def start_memory_comparison(project_id: int, body: ComparisonIn | None = None, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Draft every question once per memory state (costs one model call per question per arm)."""
    job = experiment.start_comparison(ctx, project_id, body.arms if body else None)
    return {"job_id": job.id, "status": job.status}


@router.get("/projects/{project_id}/comparison")
async def get_memory_comparison(project_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    return experiment.latest_comparison(ctx, project_id)


class JudgeIn(BaseModel):
    judge_model: str | None = None  # default: the current model (flagged if it also drafted)
    orders: int = 2  # 2 = judge each pair both ways round (recommended); 1 = half the calls


@router.get("/projects/{project_id}/comparison/judge")
async def get_judge(
    project_id: int, judge_model: str | None = Query(None), orders: int = Query(2, ge=1, le=2),
    ctx: V1Context = Depends(get_v1),
) -> dict[str, Any]:
    """The judge's verdicts so far, and what judging the rest would cost (no model calls). Without a
    judge_model, both describe the model the last judging run used."""
    found = judge.results(ctx, project_id, judge_model)
    model = judge_model or (found["judge"] or {}).get("judge_model")
    return {**found, "plan": judge.plan(ctx, project_id, model, orders)}


@router.post("/projects/{project_id}/comparison/judge", status_code=202)
async def start_judge(project_id: int, body: JudgeIn | None = None, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Ask a judge model to compare the before/after drafts blind. Costs plan.calls_needed model calls."""
    body = body or JudgeIn()
    if body.judge_model and not MODEL_NAME.match(body.judge_model):
        raise PipelineError("invalid_model", f"{body.judge_model!r} isn't a valid model name.", 422)
    planned = judge.plan(ctx, project_id, body.judge_model, body.orders)["calls_needed"]
    job = judge.start_judging(ctx, project_id, body.judge_model, body.orders)
    return {"job_id": job.id, "status": job.status, "planned_calls": planned}


class SpotCheckIn(BaseModel):
    requirement_id: int
    choice: str  # A | B | tie
    note: str | None = None


@router.get("/projects/{project_id}/comparison/spot-check")
async def get_spot_check(project_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    return judge.spot_check(ctx, project_id)


@router.post("/projects/{project_id}/comparison/spot-check")
async def post_spot_check(project_id: int, body: SpotCheckIn, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    return judge.record_spot_check(ctx, project_id, body.requirement_id, body.choice, body.note)


@router.get("/memory/lessons")
async def memory_lessons(
    limit: int = Query(100, ge=1, le=500), ctx: V1Context = Depends(get_v1)
) -> dict[str, Any]:
    """What has been sent to (or is waiting for) the Hindsight lessons bank, newest first."""
    with ctx.db.session() as session:
        rows = session.scalars(select(Lesson).order_by(Lesson.happened_at.desc(), Lesson.id.desc()).limit(limit)).all()
        total = session.scalar(select(func.count()).select_from(Lesson)) or 0
    return {"total": total, "pending": pending_lessons(ctx.db), "lessons": [
        {"key": row.key, "signal": row.signal, "text": row.text, "tags": row.tags, "happened_at": row.happened_at,
         "status": row.hindsight_status, "error": row.hindsight_error}
        for row in rows
    ]}


@router.get("/memory/playbook")
async def memory_playbook(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """The Hindsight mental model 'what wins, what loses'. Reading uses no Hindsight credits."""
    if ctx.lessons is None:
        return {"enabled": False, "playbook": None}
    try:
        return {"enabled": True, "playbook": await ctx.lessons.playbook(refresh=False)}
    except MemoryUnavailable as exc:
        raise PipelineError("hindsight_unavailable", str(exc), 503) from exc


@router.post("/memory/playbook/refresh")
async def refresh_playbook(ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Create or refresh the playbook mental model (uses Hindsight credits)."""
    if ctx.lessons is None:
        raise PipelineError("lessons_disabled", "The Hindsight lessons bank is turned off (RFP_LESSONS).", 409)
    await ctx.sync_lessons()
    try:
        return {"enabled": True, "playbook": await ctx.lessons.playbook(refresh=True)}
    except MemoryUnavailable as exc:
        raise PipelineError("hindsight_unavailable", str(exc), 503) from exc


# --- projects ---------------------------------------------------------------------------------


@router.post("/projects", response_model=ProjectSummary, status_code=202)
async def create_project(
    file: UploadFile = File(...),
    fact_sheet: UploadFile | None = File(None),
    name: str | None = Form(None),
    client: str | None = Form(None),
    industry: str | None = Form(None),
    ctx: V1Context = Depends(get_v1),
) -> ProjectSummary:
    data = await _read_upload(ctx, file)
    fact_sheet_data = await _read_upload(ctx, fact_sheet) if fact_sheet is not None else None
    project, job = projects.create_project(
        ctx, filename=file.filename or "upload", data=data, name=name, client=client, industry=industry,
        fact_sheet_filename=(fact_sheet.filename or "facts.json") if fact_sheet is not None else None,
        fact_sheet_data=fact_sheet_data,
    )
    return ProjectSummary(
        id=project.id, name=project.name, client=project.client, state=project.state,
        filename=file.filename or "upload", created_at=project.created_at, requirement_count=0, job=_job_view(job),
    )


@router.get("/projects", response_model=list[ProjectSummary])
async def list_projects(ctx: V1Context = Depends(get_v1)) -> list[ProjectSummary]:
    with ctx.db.session() as session:
        rows = session.scalars(select(Project).order_by(Project.id.desc())).all()
        return [
            ProjectSummary(
                id=p.id, name=p.name, client=p.client, state=p.state, filename=p.document.filename,
                created_at=p.created_at, requirement_count=len(p.requirements),
            )
            for p in rows
        ]


@router.get("/projects/{project_id}", response_model=ProjectView)
async def get_project(project_id: int, ctx: V1Context = Depends(get_v1)) -> ProjectView:
    return _project_view(ctx, project_id)


def _brief_view(row: ProjectBrief | None) -> dict[str, Any]:
    if row is None:
        return {"brief": None}
    return {"brief": {"text": row.text, "based_on": row.based_on, "created_at": row.created_at}}


@router.get("/projects/{project_id}/brief")
async def get_brief(project_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """The cached Hindsight client brief (no Hindsight call)."""
    with ctx.db.session() as session:
        if session.get(Project, project_id) is None:
            raise PipelineError("not_found", f"No project {project_id}.", 404)
        return {"enabled": ctx.lessons is not None, **_brief_view(session.get(ProjectBrief, project_id))}


@router.post("/projects/{project_id}/brief")
async def make_brief(project_id: int, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    """Ask Hindsight Reflect what it remembers about this client and industry (uses Hindsight credits)."""
    if ctx.lessons is None:
        raise PipelineError("lessons_disabled", "The Hindsight lessons bank is turned off (RFP_LESSONS).", 409)
    with ctx.db.session() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise PipelineError("not_found", f"No project {project_id}.", 404)
        client, industry = project.client, project.industry
    await ctx.sync_lessons()
    try:
        brief = await ctx.lessons.reflect(brief_query(client, industry), tags=brief_tags(client, industry))
    except MemoryUnavailable as exc:
        raise PipelineError("hindsight_unavailable", str(exc), 503) from exc
    with ctx.db.session() as session:
        row = session.get(ProjectBrief, project_id)
        if row is None:
            row = ProjectBrief(project_id=project_id, text=brief.text, based_on=brief.based_on)
            session.add(row)
        else:
            row.text, row.based_on, row.created_at = brief.text, brief.based_on, utcnow()
        session.commit()
        return {"enabled": True, **_brief_view(row)}


@router.post("/projects/{project_id}/retry", response_model=JobView, status_code=202)
async def retry_project(project_id: int, ctx: V1Context = Depends(get_v1)) -> JobView:
    return _job_view(projects.retry_extraction(ctx, project_id))


@router.put("/projects/{project_id}/fact-sheet", response_model=ProjectView)
async def put_project_fact_sheet(
    project_id: int,
    file: UploadFile = File(...),
    ctx: V1Context = Depends(get_v1),
) -> ProjectView:
    data = await _read_upload(ctx, file)
    projects.set_fact_sheet(ctx, project_id, file.filename or "facts.json", data)
    return _project_view(ctx, project_id)


@router.put("/projects/{project_id}/requirements", response_model=ProjectView)
async def put_requirements(project_id: int, body: RequirementsIn, ctx: V1Context = Depends(get_v1)) -> ProjectView:
    projects.replace_requirements(
        ctx, project_id,
        [projects.RequirementEdit(**item.model_dump()) for item in body.requirements],
    )
    return _project_view(ctx, project_id)


@router.post("/projects/{project_id}/draft", response_model=JobView, status_code=202)
async def draft(project_id: int, body: DraftIn | None = None, ctx: V1Context = Depends(get_v1)) -> JobView:
    job = projects.start_drafting(ctx, project_id, body.requirement_ids if body else None)
    return _job_view(job)


@router.post("/jobs/{job_id}/cancel", response_model=JobView)
async def cancel_job(job_id: int, ctx: V1Context = Depends(get_v1)) -> JobView:
    """Stop a queued or running background job (extraction or drafting)."""
    return _job_view(ctx.jobs.cancel(job_id))


@router.get("/jobs/{job_id}", response_model=JobView)
async def get_job(job_id: int, ctx: V1Context = Depends(get_v1)) -> JobView:
    with ctx.db.session() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise PipelineError("not_found", f"No job {job_id}.", 404)
        return _job_view(job, ctx.settings.provider, ctx.settings.model)


@router.post("/requirements/{requirement_id}/review", response_model=RequirementView)
async def post_review(requirement_id: int, body: ReviewIn, ctx: V1Context = Depends(get_v1)) -> RequirementView:
    projects.review(ctx, requirement_id, body.action, body.final_text, body.reason_tags, body.rating)
    with ctx.db.session() as session:
        return _requirement_view(session, session.get(RequirementRow, requirement_id))


@router.post("/requirements/{requirement_id}/regenerate", response_model=RequirementView)
async def post_regenerate(requirement_id: int, body: RegenerateIn, ctx: V1Context = Depends(get_v1)) -> RequirementView:
    await projects.regenerate(ctx, requirement_id, body.instructions)
    with ctx.db.session() as session:
        return _requirement_view(session, session.get(RequirementRow, requirement_id))


@router.put("/projects/{project_id}/outcome")
async def put_outcome(project_id: int, body: OutcomeIn, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    row = outcomes.record_outcome(
        ctx, project_id, result=body.result, loss_reason=body.loss_reason, decided_at=body.decided_at
    )
    return {
        "project_id": row.project_id, "result": row.result,
        "loss_reason": row.loss_reason, "decided_at": row.decided_at,
    }


@router.post("/projects/{project_id}/debrief")
async def post_debrief(project_id: int, body: DebriefIn, ctx: V1Context = Depends(get_v1)) -> dict[str, Any]:
    rows = outcomes.add_debrief(
        ctx, project_id,
        [outcomes.DebriefItem(section=i.section, score=i.score, comment=i.comment) for i in body.items],
    )
    return {"created": len(rows)}


@router.get("/projects/{project_id}/export")
async def export(project_id: int, format: str = Query("docx", pattern="^(docx|xlsx)$"), ctx: V1Context = Depends(get_v1)) -> Response:
    with ctx.db.session() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise PipelineError("not_found", f"No project {project_id}.", 404)
        pending = [r.code for r in project.requirements if not projects.is_final(r)]
        if not project.requirements or pending:
            raise PipelineError(
                "not_final",
                f"Every requirement must be accepted, edited or rewritten before export ({len(pending)} still open).",
                409,
            )
        answers = [
            FinalAnswer(code=r.code, section=r.section, reference=r.reference, question=r.question,
                        answer=projects.final_text(r) or "")
            for r in project.requirements
        ]
        name, client, original = project.name, project.client, project.document.stored_path
        filename = project.document.filename

    if format == "xlsx":
        if not filename.lower().endswith(".xlsx"):
            raise PipelineError("invalid_request", "Excel export needs an Excel RFP; export this one as Word.", 422)
        from pathlib import Path

        content, unplaced = export_xlsx(Path(original), answers)
        media, out_name = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", f"{name} - response.xlsx"
        headers = {"X-Unplaced-Answers": ",".join(unplaced)} if unplaced else {}
    else:
        subtitle = f"Response prepared for {client}" if client else None
        content = export_docx(f"{name}: response", subtitle, answers)
        media, out_name, headers = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            f"{name} - response.docx",
            {},
        )
    projects.mark_exported(ctx, project_id)
    headers["Content-Disposition"] = f'attachment; filename="{out_name}"'
    return Response(content=content, media_type=media, headers=headers)
