"""Projects: an RFP → confirmed requirement list → drafts from facts + past answers → review → export.

The steps are fixed (no agent). Drafting is resumable: each (job, requirement) gets at most one
draft, so a job resumed after a crash only drafts what's missing (design §12.1).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError

from ..grounding import count_words, evaluate_draft
from ..llm import LLMError
from ..parser import ParseError, parse_document
from ..prompts import DRAFT_PROMPT_VERSION
from ..provider_errors import StopOnBlocking
from ..schemas import Draft, Fact, Requirement
from ..core import PipelineError, load_default_fact_sheet, load_fact_sheet
from .db import Answer, ComparisonDraft, DraftRow, HumanVerdict, Job, JudgeVerdict, Project, RequirementRow, Review
from .evidence import check_claims
from .jobs import JobFailed, JobRunner
from .learning import apply_review_signals, client_instructions
from .retrieval import retrieve
from .storage import check_extension, store_document, store_fact_sheet
from .sync import touch

if TYPE_CHECKING:
    from .context import V1Context

FINAL_ACTIONS = ("accepted", "edited", "rewritten")
REVIEW_ACTIONS = (*FINAL_ACTIONS, "rejected")
EDITABLE_STATES = ("requirements_extracted",)
DRAFTABLE_STATES = ("requirements_extracted", "in_review", "approved", "exported")


def register(jobs: JobRunner) -> None:
    jobs.register("extract_requirements", run_extract_requirements, on_cancel=_extraction_stopped)
    jobs.register("draft_all", run_draft_all, on_cancel=_drafting_stopped)


def _job_target(ctx: V1Context, job_id: int) -> int:
    with ctx.db.session() as session:
        return session.get(Job, job_id).target_id


def _extraction_stopped(ctx: V1Context, job_id: int) -> None:
    """A stopped extraction lands in 'failed', where Retry extraction already works."""
    with ctx.db.session() as session:
        project = session.get(Project, _job_target(ctx, job_id))
        if project is not None and project.state == "extracting":
            project.state = "failed"
            project.error = "Stopped before the requirements were extracted. Retry extraction to run it again."
            session.commit()


def _drafting_stopped(ctx: V1Context, job_id: int) -> None:
    """Drafts already written are kept; the rest can be drafted later from the review page."""
    project_id = _job_target(ctx, job_id)
    with ctx.db.session() as session:
        project = session.get(Project, project_id)
        if project is not None and project.state == "drafting":
            project.state = "in_review"
            session.commit()
    _refresh_state(ctx, project_id)


# --- creation and requirement extraction -------------------------------------------------------


def create_project(
    ctx: V1Context, *, filename: str, data: bytes, name: str | None, client: str | None, industry: str | None,
    fact_sheet_filename: str | None = None, fact_sheet_data: bytes | None = None,
) -> tuple[Project, Job]:
    check_extension(filename)
    fact_sheet_document = None
    if fact_sheet_data is not None:
        fact_sheet_filename = fact_sheet_filename or "facts.json"
        load_fact_sheet(fact_sheet_data, fact_sheet_filename)
        fact_sheet_document = store_fact_sheet(ctx, fact_sheet_filename, fact_sheet_data)
    document = store_document(ctx, "rfp", filename, data)
    with ctx.db.session() as session:
        project = Project(
            document_id=document.id,
            fact_sheet_document_id=fact_sheet_document.id if fact_sheet_document else None,
            name=(name or "").strip() or Path(filename).stem,
            client=(client or "").strip() or None,
            industry=(industry or "").strip() or None,
            state="extracting",
        )
        session.add(project)
        session.commit()
    job = ctx.jobs.submit("extract_requirements", project.id)
    return project, job


def retry_extraction(ctx: V1Context, project_id: int) -> Job:
    """Re-run extraction before drafting without re-uploading the source RFP."""
    with ctx.db.session() as session:
        project = _project(session, project_id)
        if project.state not in ("failed", "requirements_extracted"):
            raise PipelineError(
                "invalid_state",
                "Requirements can only be re-extracted before drafting starts.",
                409,
            )
        project.state, project.error = "extracting", None
        session.commit()
    return ctx.jobs.submit("extract_requirements", project_id)


def set_fact_sheet(ctx: V1Context, project_id: int, filename: str, data: bytes) -> Project:
    """Attach or replace a project fact sheet before drafting starts."""
    sheet = load_fact_sheet(data, filename)
    if not any(fact.is_live(date.today()) for fact in sheet.facts):
        raise PipelineError("invalid_fact_sheet", "Every fact in the project fact sheet has expired.", 422)
    with ctx.db.session() as session:
        project = _project(session, project_id)
        if project.state not in ("failed", "requirements_extracted"):
            raise PipelineError(
                "invalid_state",
                "The project fact sheet can only be changed before drafting starts.",
                409,
            )
    document = store_fact_sheet(ctx, filename, data)
    with ctx.db.session() as session:
        project = _project(session, project_id)
        if project.state not in ("failed", "requirements_extracted"):
            raise PipelineError("invalid_state", "Drafting started before the fact sheet could be saved.", 409)
        project.fact_sheet_document_id = document.id
        session.commit()
        return project


def project_fact_sheet(ctx: V1Context, project_id: int):
    with ctx.db.session() as session:
        project = _project(session, project_id)
        document = project.fact_sheet_document
        if document is None:
            return load_default_fact_sheet(ctx.settings), None
        filename, path = document.filename, Path(document.stored_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PipelineError("invalid_fact_sheet", f"Project fact sheet not found at {path}", 500) from exc
    return load_fact_sheet(raw, filename), filename


async def run_extract_requirements(ctx: V1Context, job_id: int) -> None:
    with ctx.db.session() as session:
        job = session.get(Job, job_id)
        project = session.get(Project, job.target_id)
        filename, path = project.document.filename, Path(project.document.stored_path)

    settings = ctx.settings
    try:
        document = parse_document(filename, path.read_bytes(), settings.max_document_chars)
        result = await ctx.llm.extract_requirements(document)
    except (ParseError, LLMError) as exc:
        _fail_project(ctx, project.id, exc.message)
        raise JobFailed(exc.message) from exc

    items = [i for i in result.output.requirements if i.question.strip()]
    if not items:
        _fail_project(ctx, project.id, f"No requirements were found in {filename}.")
        raise JobFailed(f"No requirements were found in {filename}.")
    if len(items) > settings.max_requirements:
        message = f"{filename} has {len(items)} requirements; the limit is {settings.max_requirements}."
        _fail_project(ctx, project.id, message)
        raise JobFailed(message)

    with ctx.db.session() as session:
        _clear_requirements(session, project.id)
        for order, item in enumerate(items, start=1):
            session.add(
                RequirementRow(
                    project_id=project.id,
                    order=order,
                    section=(item.section or "").strip() or None,
                    reference=(item.reference or "").strip() or None,
                    question=item.question.strip(),
                    mandatory=item.mandatory,
                    word_limit=item.word_limit if item.word_limit and item.word_limit > 0 else None,
                )
            )
        row = session.get(Project, project.id)
        row.state, row.error = "requirements_extracted", None
        job = session.get(Job, job_id)
        job.total = job.done = len(items)
        session.commit()


def _clear_requirements(session, project_id: int) -> None:  # noqa: ANN001
    """Remove a project's requirements before a new list replaces them (edits, re-extraction). A
    before/after comparison and its judge and spot-check verdicts were about the old list, so they go
    too (only possible before drafting: drafts belong to requirements and block replacing them)."""
    old = select(RequirementRow.id).where(RequirementRow.project_id == project_id)
    for model in (HumanVerdict, JudgeVerdict, ComparisonDraft):
        session.execute(delete(model).where(model.requirement_id.in_(old)))
    session.execute(delete(RequirementRow).where(RequirementRow.project_id == project_id))


def _fail_project(ctx: V1Context, project_id: int, message: str) -> None:
    with ctx.db.session() as session:
        row = session.get(Project, project_id)
        row.state, row.error = "failed", message
        session.commit()


# --- editing the requirement list ---------------------------------------------------------------


@dataclass
class RequirementEdit:
    question: str
    section: str | None = None
    reference: str | None = None
    mandatory: bool | None = None
    word_limit: int | None = None


def replace_requirements(ctx: V1Context, project_id: int, items: list[RequirementEdit]) -> None:
    """The person's corrected list (edits, merges, splits, deletions). Only before drafting."""
    cleaned = [i for i in items if i.question and i.question.strip()]
    if not cleaned:
        raise PipelineError("invalid_request", "The requirement list can't be empty.", 422)
    with ctx.db.session() as session:
        project = _project(session, project_id)
        if project.state not in EDITABLE_STATES:
            raise PipelineError("invalid_state", "Requirements can only be edited before drafting starts.", 409)
        _clear_requirements(session, project_id)
        for order, item in enumerate(cleaned, start=1):
            session.add(
                RequirementRow(
                    project_id=project_id,
                    order=order,
                    section=(item.section or "").strip() or None,
                    reference=(item.reference or "").strip() or None,
                    question=item.question.strip(),
                    mandatory=item.mandatory,
                    word_limit=item.word_limit if item.word_limit and item.word_limit > 0 else None,
                )
            )
        session.commit()


# --- drafting ---------------------------------------------------------------------------------


def start_drafting(ctx: V1Context, project_id: int, requirement_ids: list[int] | None = None) -> Job:
    with ctx.db.session() as session:
        project = _project(session, project_id)
        if project.state not in DRAFTABLE_STATES:
            raise PipelineError("invalid_state", f"A '{project.state}' project can't be drafted.", 409)
        valid = {r.id for r in project.requirements}
        if requirement_ids is not None and not set(requirement_ids) <= valid:
            raise PipelineError("invalid_request", "Some requirement IDs don't belong to this project.", 422)
        project.state = "drafting"
        session.commit()
    payload = {"requirement_ids": requirement_ids} if requirement_ids is not None else {}
    return ctx.jobs.submit("draft_all", project_id, payload)


async def run_draft_all(ctx: V1Context, job_id: int) -> None:
    settings = ctx.settings
    with ctx.db.session() as session:
        job = session.get(Job, job_id)
        project = session.get(Project, job.target_id)
        wanted = set(job.payload.get("requirement_ids") or [r.id for r in project.requirements])
        requirements = [r for r in project.requirements if r.id in wanted]
        drafted = set(session.scalars(select(DraftRow.requirement_id).where(DraftRow.job_id == job_id)))
        job.total = len(requirements)
        job.done = len(drafted & wanted)
        session.commit()
        todo = [r for r in requirements if r.id not in drafted]  # resume: skip what's already done

    company, facts = _live_facts(ctx, project.id)
    semaphore = asyncio.Semaphore(settings.draft_concurrency)
    warnings: set[str] = set()
    breaker = StopOnBlocking()

    async def one(requirement: RequirementRow) -> None:
        async with semaphore:
            if breaker.tripped:  # left undrafted: the project page offers to draft the missing ones
                breaker.skipped += 1
                return
            draft = await draft_requirement(ctx, project, requirement, company, facts, job_id=job_id)
        if draft is not None:
            breaker.record(draft.error)
            if draft.retrieval_warning:
                warnings.add(draft.retrieval_warning)
        with ctx.db.session() as session:
            session.execute(update(Job).where(Job.id == job_id).values(done=Job.done + 1))
            session.commit()

    await asyncio.gather(*(one(r) for r in todo))

    stopped = breaker.summary(len(todo) - breaker.skipped, len(requirements), settings.provider, settings.model)
    with ctx.db.session() as session:
        row = session.get(Project, project.id)
        row.state = "in_review"
        job = session.get(Job, job_id)
        job.warning = "; ".join(([stopped] if stopped else []) + sorted(warnings)) or None
        session.commit()
    _refresh_state(ctx, project.id)


async def draft_requirement(
    ctx: V1Context,
    project: Project,
    requirement: RequirementRow,
    company: str,
    facts: list[Fact],
    *,
    job_id: int | None = None,
    instructions: str | None = None,
) -> DraftRow | None:
    """Retrieve past answers, draft, run check 1, and store a new draft version."""
    settings = ctx.settings
    found = await retrieve(
        requirement.question,
        memory=ctx.memory,
        db=ctx.db,
        k=settings.retrieval_top_k,
        mode=settings.retrieval_mode,
        client=project.client,
        industry=project.industry,
        freshness_half_life_days=settings.retrieval_freshness_half_life_days,
        lessons=ctx.lessons, relevance=settings.retrieval_relevance,
        min_share=settings.retrieval_relevance_min_share,
        exclude_lost_proposals=True,
    )
    spec = Requirement(
        id=requirement.code,
        section=requirement.section,
        question=requirement.question,
        mandatory=requirement.mandatory,
        word_limit=requirement.word_limit,
        reference=requirement.reference,
    )
    # A draft may cite only the facts and the past answers it was actually shown.
    valid_ids = {f.id for f in facts} | {p.id for p in found.past_answers}
    llm = ctx.llm
    with ctx.db.session() as session:
        learned = client_instructions(session, project.client)
    drafting_instructions = " ".join(part for part in (learned, instructions) if part) or None
    try:
        result = await llm.draft_answer(company, facts, spec, found.past_answers, drafting_instructions)
        evaluated = evaluate_draft(spec, result.output, valid_ids, result.model)
        evaluated = _prepare_sme_handoff(spec, evaluated)
        model, error = result.model, None
    except LLMError as exc:
        evaluated, model, error = None, llm.model, exc.message

    with ctx.db.session() as session:
        version = (session.scalar(select(func.max(DraftRow.version)).where(DraftRow.requirement_id == requirement.id)) or 0) + 1
        claims = [c.model_dump() for c in evaluated.claims] if evaluated else []
        flags = list(evaluated.flags) if evaluated else []
        if evaluated and settings.evidence_check:
            source_text = {f.id: f.statement for f in facts} | {p.id: p.answer for p in found.past_answers}
            claims, support_statuses = check_claims(claims, source_text)
            if "unsupported" in support_statuses:
                flags.append("evidence_unsupported")
            elif "partial" in support_statuses:
                flags.append("evidence_partial")
        draft = DraftRow(
            requirement_id=requirement.id,
            job_id=job_id,
            version=version,
            status=evaluated.status if evaluated else "failed",
            answer=evaluated.answer if evaluated else "",
            claims=claims,
            sources=evaluated.sources if evaluated else [],
            unsupported_claims=evaluated.unsupported_claims if evaluated else [],
            invalid_citations=evaluated.invalid_citations if evaluated else [],
            flags=list(dict.fromkeys(flags)),
            sme_question=evaluated.sme_question if evaluated else None,
            word_count=evaluated.word_count if evaluated else 0,
            retrieved=found.retrieved,
            retrieval_warning=found.warning,
            instructions=drafting_instructions,
            prompt_version=DRAFT_PROMPT_VERSION,
            model=model,
            error=error,
        )
        session.add(draft)
        try:
            session.commit()
        except IntegrityError:  # a resumed job raced itself: the draft already exists
            session.rollback()
            return None
    return draft


def _sme_fields(requirement: Requirement) -> list[str]:
    """Return a short, requirement-specific checklist without inventing company facts."""
    text = requirement.question.lower()
    def contains(*terms: str) -> bool:
        return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)

    # Check the most specific evidence type first. A requirement can mention cost or payment while
    # primarily asking for references, personnel, or a delivery methodology.
    if contains("personnel", "cv", "cvs", "curricula", "qualification", "qualifications", "team member"):
        return [
            "proposed name, role, and availability",
            "relevant qualifications, certifications, and years of experience",
            "two comparable engagements and the person's responsibilities",
        ]
    if contains("experience", "reference", "references", "last 7", "past project", "past projects"):
        return [
            "client and project name approved for disclosure",
            "relevant scope, contract value, and delivery dates",
            "measurable result and reference contact or verification route",
        ]
    if contains("awareness", "training", "curriculum"):
        return [
            "audience, learning objectives, and delivery format",
            "week-by-week curriculum, facilitators, and training materials",
            "attendance, assessment, and effectiveness measures",
        ]
    if contains("payment gateway", "assessment", "reviewing the it architecture"):
        return [
            "systems, interfaces, data flows, and environments in scope",
            "assessment standards, tools, tests, and risk-rating method",
            "deliverables, remediation priorities, owners, and acceptance criteria",
        ]
    if contains("methodology", "timeline", "milestone", "milestones", "team roles"):
        return [
            "phases, activities, and decision gates",
            "timeline, dependencies, and named delivery roles",
            "deliverables, acceptance criteria, and payment milestones",
        ]
    if contains("strategy", "programme", "program", "roadmap"):
        return [
            "current-state assessment and target framework",
            "governance, workstreams, priorities, and accountable owners",
            "sequenced roadmap, deliverables, measures, and decision gates",
        ]
    if contains("price", "pricing", "rate", "rates", "cost", "fee", "fees"):
        return [
            "currency, pricing model, and applicable taxes",
            "fees or hourly rates by role and deliverable",
            "commercial assumptions, exclusions, and payment terms",
        ]
    return [
        "the direct company-specific response",
        "the proposed approach, roles, and deliverables",
        "approved evidence, assumptions, and measurable commitments",
    ]


def _prepare_sme_handoff(requirement: Requirement, draft: Draft) -> Draft:
    """Turn an evidence gap into an actionable handoff instead of an empty answer.

    Bracketed fields are instructions, not claims. Reviews refuse to approve the text until every
    marker is replaced, so the convenience cannot accidentally become proposal content.
    """
    if draft.status != "needs_sme":
        return draft
    if "SME COMPLETION TEMPLATE" in draft.answer:
        return draft
    parts = ["SME COMPLETION TEMPLATE - replace every bracketed field before approval."]
    if draft.answer:
        parts.extend(["", "Supported starting point:", draft.answer])
    parts.extend(["", "Complete with:"])
    parts.extend(f"- [SME input required: {field}]" for field in _sme_fields(requirement))
    answer = "\n".join(parts)
    flags = [flag for flag in draft.flags if flag not in ("empty_answer", "no_claims", "over_word_limit")]
    flags.append("sme_template")
    word_count = count_words(answer)
    over_limit = bool(requirement.word_limit) and word_count > requirement.word_limit
    if over_limit:
        flags.append("over_word_limit")
    return draft.model_copy(update={
        "answer": answer,
        "word_count": word_count,
        "over_word_limit": over_limit,
        "flags": list(dict.fromkeys(flags)),
    })


async def regenerate(ctx: V1Context, requirement_id: int, instructions: str | None) -> DraftRow:
    with ctx.db.session() as session:
        requirement = session.get(RequirementRow, requirement_id)
        if requirement is None:
            raise PipelineError("not_found", f"No requirement {requirement_id}.", 404)
        project = requirement.project
        if project.state in ("extracting", "requirements_extracted", "drafting", "failed"):
            raise PipelineError("invalid_state", "Draft the project before regenerating single answers.", 409)
    company, facts = _live_facts(ctx, project.id)
    draft = await draft_requirement(ctx, project, requirement, company, facts, instructions=instructions)
    _refresh_state(ctx, project.id)
    return draft


def _live_facts(ctx: V1Context, project_id: int) -> tuple[str, list[Fact]]:
    sheet, _filename = project_fact_sheet(ctx, project_id)
    today = date.today()
    live = [f for f in sheet.facts if f.is_live(today)]
    if not live:
        raise PipelineError("invalid_fact_sheet", "Every fact in the project fact sheet has expired.", 422)
    return sheet.company, live


# --- review -----------------------------------------------------------------------------------


def edit_distance(draft: str, final: str) -> float:
    """0 = unchanged, 1 = completely different (1 − difflib similarity ratio)."""
    return round(1 - SequenceMatcher(None, draft or "", final or "").ratio(), 3)


def review(
    ctx: V1Context,
    requirement_id: int,
    action: str,
    final_text: str | None,
    reason_tags: list[str] | None = None,
    rating: int | None = None,
) -> Review:
    if action not in REVIEW_ACTIONS:
        raise PipelineError("invalid_request", f"action must be one of {', '.join(REVIEW_ACTIONS)}.", 422)
    with ctx.db.session() as session:
        requirement = session.get(RequirementRow, requirement_id)
        if requirement is None:
            raise PipelineError("not_found", f"No requirement {requirement_id}.", 404)
        draft = requirement.drafts[-1] if requirement.drafts else None
        if draft is None:
            raise PipelineError("invalid_state", "There is no draft to review yet.", 409)

        if action == "accepted":
            if not draft.answer.strip():
                raise PipelineError("invalid_request", "This draft has no answer to accept; rewrite it instead.", 422)
            final = draft.answer
        elif action in ("edited", "rewritten"):
            final = (final_text or "").strip()
            if not final:
                raise PipelineError("invalid_request", "Provide the final text for an edit or rewrite.", 422)
        else:
            final = None

        if final is not None and (
            "SME COMPLETION TEMPLATE" in final or "[SME input required:" in final
        ):
            raise PipelineError(
                "invalid_request",
                "Replace every SME placeholder before accepting, editing, or rewriting this answer.",
                422,
            )

        record = Review(
            draft_id=draft.id,
            action=action,
            final_text=final,
            edit_distance=edit_distance(draft.answer, final) if final is not None else None,
        )
        session.add(record)
        apply_review_signals(
            session, project=requirement.project, draft=draft, review=record,
            reason_tags=reason_tags, rating=rating,
        )
        _update_library(session, requirement, final)
        session.commit()
        project_id = requirement.project_id
    ctx.schedule_sync()
    _refresh_state(ctx, project_id)
    return record


def _update_library(session, requirement: RequirementRow, final: str | None) -> None:  # noqa: ANN001
    """One library answer per requirement: created on the first approval, updated on later ones,
    removed if the answer is rejected. The outbox pushes each change to Hindsight."""
    existing = session.scalars(
        select(Answer).where(Answer.requirement_id == requirement.id, Answer.status == "approved")
    ).first()
    if final is None:
        if existing is not None:
            existing.status = "deleted"
            existing.hindsight_status = "pending_delete"
            touch(existing)
        return
    if existing is not None:
        existing.answer = final
        existing.hindsight_status = "pending"
        touch(existing)
        return
    project = requirement.project
    session.add(
        Answer(
            question=requirement.question,
            answer=final,
            source="project",
            project_id=project.id,
            requirement_id=requirement.id,
            client=project.client,
            industry=project.industry,
            hindsight_status="pending",
        )
    )


def is_final(requirement: RequirementRow) -> bool:
    if not requirement.drafts:
        return False
    reviews = requirement.drafts[-1].reviews
    return bool(reviews) and reviews[-1].action in FINAL_ACTIONS


def final_text(requirement: RequirementRow) -> str | None:
    if not is_final(requirement):
        return None
    return requirement.drafts[-1].reviews[-1].final_text


def _refresh_state(ctx: V1Context, project_id: int) -> None:
    with ctx.db.session() as session:
        project = session.get(Project, project_id)
        if project is None or project.state in ("extracting", "requirements_extracted", "drafting", "failed"):
            return
        all_final = bool(project.requirements) and all(is_final(r) for r in project.requirements)
        if all_final and project.state != "exported":
            project.state = "approved"
        elif not all_final:
            project.state = "in_review"
        session.commit()


def mark_exported(ctx: V1Context, project_id: int) -> None:
    with ctx.db.session() as session:
        project = session.get(Project, project_id)
        project.state = "exported"
        session.commit()


def _project(session, project_id: int) -> Project:  # noqa: ANN001
    project = session.get(Project, project_id)
    if project is None:
        raise PipelineError("not_found", f"No project {project_id}.", 404)
    return project
