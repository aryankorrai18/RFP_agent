"""Before / after: draft the same questions under different memory states, then compare.

A fair comparison changes one thing. Every arm uses the same project, model, prompt, fact sheet and
question, and no reviewer instructions; only the retrieval mode differs (plain = relevance only,
hindsight = ranked with the accumulated lessons). Lessons never enter the prompt: they only decide
which past answers the drafter is shown, so any difference comes from that choice.

These drafts live in their own table. They are not reviewed, exported or counted as learning.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from ..config import RETRIEVAL_MODES
from ..grounding import evaluate_draft
from ..llm import LLMError
from ..provider_errors import StopOnBlocking, explain_message
from ..schemas import Requirement
from ..core import PipelineError
from .db import Answer, ComparisonDraft, Job, Lesson, Project, RequirementRow
from .evidence import check_claims
from .jobs import JobFailed
from .retrieval import retrieve

if TYPE_CHECKING:
    from .context import V1Context

KIND = "compare_memory"
DEFAULT_ARMS = ("plain", "hindsight")
MAX_QUESTIONS = 30  # each question costs one model call per arm
# Pins the model's own sampling so the arms differ only in what memory offered. Gemini and Groq
# honour it; Claude ignores it (extended thinking needs temperature 1). Even at 0 a provider may
# not be perfectly deterministic, so one run is evidence, not proof.
TEMPERATURE = 0


def start_comparison(ctx: V1Context, project_id: int, arms: list[str] | None = None) -> Job:
    from .projects import DRAFTABLE_STATES

    arms = list(dict.fromkeys(arms or DEFAULT_ARMS))
    unknown = [a for a in arms if a not in RETRIEVAL_MODES]
    if unknown or len(arms) < 2:
        raise PipelineError("invalid_request", f"Choose at least two of: {', '.join(RETRIEVAL_MODES)}.", 422)
    if "hindsight" in arms and ctx.lessons is None:
        raise PipelineError("invalid_state", "Hindsight lessons are switched off (RFP_LESSONS), so there is nothing to compare.", 409)
    with ctx.db.session() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise PipelineError("not_found", f"No project {project_id}.", 404)
        if project.state not in DRAFTABLE_STATES:
            raise PipelineError("invalid_state", f"A '{project.state}' project has no requirements to compare yet.", 409)
        count = len(project.requirements)
        if count > MAX_QUESTIONS:
            raise PipelineError("too_large", f"{count} questions × {len(arms)} arms is too many model calls; the limit is {MAX_QUESTIONS} questions.", 422)
        busy = session.scalar(select(Job.id).where(Job.kind == KIND, Job.target_id == project_id, Job.status.in_(("queued", "running"))))
        if busy:
            raise PipelineError("invalid_state", "A comparison is already running for this project.", 409)
    # The conditions are recorded so a later run can tell whether this run's drafts are comparable.
    payload = {"arms": arms, "model": ctx.settings.model, "temperature": TEMPERATURE,
               "relevance": ctx.settings.retrieval_relevance, "min_share": ctx.settings.retrieval_relevance_min_share}
    return ctx.jobs.submit(KIND, project_id, payload)


def _carry_over_source(session, job: Job) -> Job | None:  # noqa: ANN001
    """The run to finish instead of starting over: only the comparison run just before this one,
    only if it was stopped before finishing, and only if it ran under the same arms, model and
    temperature. A finished run is never reused: running again is how you see what changed after
    memory grew, so those questions have to be drafted fresh."""
    previous = session.scalars(
        select(Job).where(Job.kind == KIND, Job.target_id == job.target_id, Job.id < job.id).order_by(Job.id.desc())
    ).first()
    if previous is None or previous.status not in ("cancelled", "interrupted", "failed"):
        return None
    same = ("arms", "model", "temperature", "relevance", "min_share")
    if any(key not in previous.payload or previous.payload[key] != job.payload.get(key) for key in same):
        return None
    return previous


async def run_comparison(ctx: V1Context, job_id: int) -> None:
    from .projects import _live_facts

    settings = ctx.settings
    with ctx.db.session() as session:
        job = session.get(Job, job_id)
        arms = list(job.payload["arms"])
        project = session.get(Project, job.target_id)
        requirements = list(project.requirements)
        done = set(session.execute(select(ComparisonDraft.requirement_id, ComparisonDraft.arm).where(ComparisonDraft.job_id == job_id)).all())
        source = _carry_over_source(session, job)
        carried = 0
        if source is not None:
            wanted = {(r.id, arm) for r in requirements for arm in arms}
            for row in session.scalars(select(ComparisonDraft).where(ComparisonDraft.job_id == source.id)):
                key = (row.requirement_id, row.arm)
                if key not in wanted or key in done or row.status == "failed":
                    continue
                session.add(ComparisonDraft(
                    job_id=job_id, project_id=project.id, requirement_id=row.requirement_id, arm=row.arm,
                    status=row.status, answer=row.answer, sources=row.sources, unsupported_claims=row.unsupported_claims,
                    flags=row.flags, retrieved=row.retrieved, warning=row.warning, model=row.model, error=row.error,
                ))
                done.add(key)
                carried += 1
            if carried:
                job.payload = {**job.payload, "carried_over_from": source.id, "carried_over": carried}
        job.total = len(requirements) * len(arms)
        job.done = len(done)
        session.commit()
    todo = [(r, arm) for r in requirements for arm in arms if (r.id, arm) not in done]  # resume: skip what's done
    company, facts = _live_facts(ctx, project.id)
    semaphore = asyncio.Semaphore(settings.draft_concurrency)
    breaker = StopOnBlocking()

    async def one(requirement: RequirementRow, arm: str) -> None:
        async with semaphore:
            if breaker.tripped:
                breaker.skipped += 1
                return
            breaker.record(await _draft_arm(ctx, job_id, project, requirement, arm, company, facts))
        with ctx.db.session() as session:
            session.execute(update(Job).where(Job.id == job_id).values(done=Job.done + 1))
            session.commit()

    await asyncio.gather(*(one(r, arm) for r, arm in todo))
    total = len(requirements) * len(arms)
    stopped = breaker.summary(len(todo) - breaker.skipped, total, settings.provider, settings.model)
    if stopped:
        # An incomplete comparison can't be read. Failing the run keeps what was drafted: running it
        # again under the same model and temperature carries those drafts over (_carry_over_source).
        raise JobFailed(f"{stopped} Provider said: {breaker.message[:500]}")
    with ctx.db.session() as session:
        failed = session.scalar(select(func.count()).select_from(ComparisonDraft).where(ComparisonDraft.job_id == job_id, ComparisonDraft.status == "failed")) or 0
        if failed:
            session.get(Job, job_id).warning = f"{failed} of {total} drafts failed (model error); the comparison skips them."
            session.commit()
    if failed == total:
        raise JobFailed("Every draft failed, so there is nothing to compare. Check the model and try again.")


async def _draft_arm(ctx: V1Context, job_id: int, project: Project, requirement: RequirementRow, arm: str,
                     company: str, facts: list) -> str | None:
    """Draft one question under one memory state and store it. Returns the model error, if any."""
    settings = ctx.settings
    found = await retrieve(
        requirement.question, memory=ctx.memory, db=ctx.db, k=settings.retrieval_top_k, mode=arm,
        client=project.client, industry=project.industry,
        freshness_half_life_days=settings.retrieval_freshness_half_life_days, lessons=ctx.lessons, relevance=settings.retrieval_relevance,
        min_share=settings.retrieval_relevance_min_share,
    )
    spec = Requirement(id=requirement.code, section=requirement.section, question=requirement.question,
                       mandatory=requirement.mandatory, word_limit=requirement.word_limit, reference=requirement.reference)
    valid_ids = {f.id for f in facts} | {p.id for p in found.past_answers}
    llm = ctx.llm
    try:
        result = await llm.draft_answer(company, facts, spec, found.past_answers, None, temperature=TEMPERATURE)
        evaluated = evaluate_draft(spec, result.output, valid_ids, result.model)
        flags = list(evaluated.flags)
        if settings.evidence_check:
            source_text = {f.id: f.statement for f in facts} | {p.id: p.answer for p in found.past_answers}
            _claims, statuses = check_claims([c.model_dump() for c in evaluated.claims], source_text)
            if "unsupported" in statuses:
                flags.append("evidence_unsupported")
        row = dict(status=evaluated.status, answer=evaluated.answer, sources=evaluated.sources,
                   unsupported_claims=evaluated.unsupported_claims, flags=list(dict.fromkeys(flags)), model=result.model, error=None)
    except LLMError as exc:
        row = dict(status="failed", answer="", sources=[], unsupported_claims=[], flags=[], model=llm.model, error=exc.message)
    with ctx.db.session() as session:
        session.add(ComparisonDraft(job_id=job_id, project_id=project.id, requirement_id=requirement.id, arm=arm,
                                    retrieved=found.retrieved, warning=found.warning, **row))
        try:
            session.commit()
        except IntegrityError:  # a resumed job raced itself
            session.rollback()
    return row["error"]


# --- reading a comparison ---------------------------------------------------------------------


def _answer_history(session) -> dict[str, dict[str, int]]:  # noqa: ANN001
    """Per answer code: how many won-proposal and lost-proposal lessons it has (from real outcomes)."""
    history: dict[str, dict[str, int]] = {}
    codes = {a.id: a.code for a in session.scalars(select(Answer))}
    for lesson in session.scalars(select(Lesson).where(Lesson.answer_id.is_not(None))):
        if "kind:proposal_outcome" not in (lesson.tags or []) or lesson.signal not in ("positive", "negative"):
            continue
        entry = history.setdefault(codes.get(lesson.answer_id, ""), {"won": 0, "lost": 0})
        entry["won" if lesson.signal == "positive" else "lost"] += 1
    return history


def latest_comparison(ctx: V1Context, project_id: int) -> dict[str, Any]:
    with ctx.db.session() as session:
        job = session.scalars(select(Job).where(Job.kind == KIND, Job.target_id == project_id).order_by(Job.id.desc())).first()
        if job is None:
            return {"comparison": None}
        rows = session.scalars(select(ComparisonDraft).where(ComparisonDraft.job_id == job.id)).all()
        requirements = {r.id: r for r in session.scalars(select(RequirementRow).where(RequirementRow.project_id == project_id))}
        history = _answer_history(session)
        arms = list(job.payload.get("arms") or DEFAULT_ARMS)
        answers = {a.code: a.question for a in session.scalars(select(Answer))}

        def cited(row: ComparisonDraft) -> list[str]:
            return [s for s in row.sources if s.startswith("ANS-")]

        def verdict(codes: list[str]) -> dict[str, int]:
            return {"won": sum(1 for c in codes if history.get(c, {}).get("won") and not history.get(c, {}).get("lost")),
                    "lost": sum(1 for c in codes if history.get(c, {}).get("lost") and not history.get(c, {}).get("won"))}

        questions = []
        for requirement in sorted(requirements.values(), key=lambda r: r.order):
            per_arm = {}
            for row in rows:
                if row.requirement_id != requirement.id:
                    continue
                top = row.retrieved[0] if row.retrieved else None
                per_arm[row.arm] = {
                    "status": row.status, "answer": row.answer, "cited_answers": cited(row),
                    "cited_history": verdict(cited(row)), "unsupported_claims": row.unsupported_claims, "flags": row.flags,
                    "warning": row.warning, "error": row.error,
                    "error_info": explain_message(row.error, ctx.settings.provider, row.model),
                    "retrieved": [{"id": r["id"], "question": answers.get(r["id"]), "rank": r.get("rank"), "final": r.get("final"),
                                   "lessons": r.get("lessons"), "lesson_evidence": r.get("lesson_evidence") or [],
                                   "reasons": r.get("reasons") or [], "history": history.get(r["id"])} for r in row.retrieved],
                    "top": top["id"] if top else None,
                }
            first, last = per_arm.get(arms[0]), per_arm.get(arms[-1])
            questions.append({
                "requirement_id": requirement.id, "code": requirement.code, "question": requirement.question, "arms": per_arm,
                "changed": bool(first and last and (first["cited_answers"] != last["cited_answers"] or first["top"] != last["top"])),
            })

        summary = {}
        for arm in arms:
            arm_rows = [r for r in rows if r.arm == arm]
            good = [r for r in arm_rows if r.status != "failed"]
            totals = {"won": 0, "lost": 0}
            for row in good:
                for key, value in verdict(cited(row)).items():
                    totals[key] += value
            summary[arm] = {
                "drafted": len(good), "failed": len(arm_rows) - len(good),
                "needs_sme": sum(r.status == "needs_sme" for r in good),
                "with_unsupported_claims": sum(bool(r.unsupported_claims) for r in good),
                "cited_answers_from_won_proposals": totals["won"], "cited_answers_from_lost_proposals": totals["lost"],
                "cited_any_past_answer": sum(bool(cited(r)) for r in good),
            }
        return {"comparison": {
            "job_id": job.id, "status": job.status, "done": job.done, "total": job.total, "warning": job.warning, "error": job.error,
            "error_info": explain_message(job.error, ctx.settings.provider, job.payload.get("model")),
            "arms": arms, "created_at": job.created_at, "model": next((r.model for r in rows if r.model), None),
            "questions_changed": sum(q["changed"] for q in questions), "questions": questions, "summary": summary,
            "temperature": job.payload.get("temperature"),
            "carried_over": job.payload.get("carried_over", 0), "carried_over_from": job.payload.get("carried_over_from"),
            "controls": ["same project and questions", "same model and prompt", "same fact sheet", "no reviewer instructions",
                         "only the retrieval mode differs; lessons never enter the prompt"]
                        + ([f"temperature {job.payload['temperature']}"] if job.payload.get("temperature") is not None else []),
        }}


def register(jobs) -> None:  # noqa: ANN001
    jobs.register(KIND, run_comparison)
