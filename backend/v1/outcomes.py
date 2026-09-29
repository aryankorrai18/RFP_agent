"""V3 slow outcomes, debrief credit, and explicit answer superseding."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import select

from ..core import PipelineError
from .db import (
    Answer,
    AnswerStats,
    Debrief,
    MemoryEvent,
    Project,
    ProjectOutcome,
    parse_answer_code,
    utcnow,
)
from .sync import touch

RESULTS = ("won", "lost", "no_decision", "unknown")
NON_QUALITY_LOSSES = {"price", "pricing", "budget", "incumbent", "relationship"}
QUALITY_LOSSES = {"technical fit", "technical_fit", "response quality", "response_quality"}


@dataclass(frozen=True)
class DebriefItem:
    section: str | None
    score: float | None
    comment: str | None


def _source_ids(project: Project, section: str | None = None) -> set[int]:
    ids: set[int] = set()
    for requirement in project.requirements:
        if section and (requirement.section or "").strip().lower() != section.strip().lower():
            continue
        if not requirement.drafts:
            continue
        for code in requirement.drafts[-1].sources or []:
            answer_id = parse_answer_code(code)
            if answer_id is not None:
                ids.add(answer_id)
    return ids


def _add_credit(session, answer_ids: set[int], field: str, amount: float) -> None:  # noqa: ANN001
    for answer_id in answer_ids:
        if session.get(Answer, answer_id) is None:
            continue
        stats = session.get(AnswerStats, answer_id)
        if stats is None:
            stats = AnswerStats(answer_id=answer_id)
            session.add(stats)
        setattr(stats, field, (getattr(stats, field) or 0.0) + amount)
        stats.updated_at = utcnow()


def outcome_credit(result: str, loss_reason: str | None) -> float:
    if result == "won":
        return 0.1
    reason = (loss_reason or "").lower()
    if result == "lost" and reason in QUALITY_LOSSES:
        return -0.1
    # Price, budget, incumbent and relationship losses must not punish answer quality.
    return 0.0


def record_outcome(
    ctx, project_id: int, *, result: str, loss_reason: str | None = None, decided_at: date | None = None
) -> ProjectOutcome:  # noqa: ANN001
    if result not in RESULTS:
        raise PipelineError("invalid_request", f"result must be one of {', '.join(RESULTS)}.", 422)
    with ctx.db.session() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise PipelineError("not_found", f"No project {project_id}.", 404)
        outcome = session.get(ProjectOutcome, project_id)
        previous_credit = outcome_credit(outcome.result, outcome.loss_reason) if outcome is not None else 0.0
        if outcome is None:
            outcome = ProjectOutcome(project_id=project_id, result=result)
            session.add(outcome)
        outcome.result = result
        outcome.loss_reason = (loss_reason or "").strip() or None
        outcome.decided_at = decided_at

        credit_delta = outcome_credit(result, outcome.loss_reason) - previous_credit
        if credit_delta:
            _add_credit(session, _source_ids(project), "outcome_credit", credit_delta)
        session.add(MemoryEvent(
            kind="project_outcome", project_id=project_id,
            detail=f"Project recorded as {result}" + (f" ({outcome.loss_reason})" if outcome.loss_reason else "") + ".",
        ))
        session.commit()
        return outcome


def add_debrief(ctx, project_id: int, items: list[DebriefItem]) -> list[Debrief]:  # noqa: ANN001
    with ctx.db.session() as session:
        project = session.get(Project, project_id)
        if project is None:
            raise PipelineError("not_found", f"No project {project_id}.", 404)
        created = []
        for item in items:
            if item.score is not None and not 1 <= item.score <= 5:
                raise PipelineError("invalid_request", "Debrief scores must be between 1 and 5.", 422)
            row = Debrief(
                project_id=project_id, section=(item.section or "").strip() or None,
                score=item.score, comment=(item.comment or "").strip() or None,
            )
            session.add(row)
            created.append(row)
            credit = ((item.score - 3) * 0.1) if item.score is not None else 0.0
            _add_credit(session, _source_ids(project, item.section), "debrief_credit", credit)
            session.add(MemoryEvent(
                kind="debrief", project_id=project_id,
                detail=f"Debrief for {item.section or 'the project'}"
                + (f" scored {item.score}/5" if item.score is not None else "") + ".",
            ))
        session.commit()
        return created


def supersede_answer(ctx, code: str, replacement_code: str) -> Answer:  # noqa: ANN001
    old_id, replacement_id = parse_answer_code(code), parse_answer_code(replacement_code)
    if old_id == replacement_id:
        raise PipelineError("invalid_request", "An answer cannot supersede itself.", 422)
    with ctx.db.session() as session:
        old = session.get(Answer, old_id) if old_id else None
        replacement = session.get(Answer, replacement_id) if replacement_id else None
        if old is None or not old.live:
            raise PipelineError("not_found", f"No live library answer {code}.", 404)
        if replacement is None or not replacement.live:
            raise PipelineError("invalid_request", f"Replacement {replacement_code} is not a live answer.", 422)
        old.superseded_by = replacement.id
        old.hindsight_status = "pending_delete"
        touch(old)
        session.add(MemoryEvent(
            kind="superseded", answer_id=old.id,
            detail=f"{old.code} was superseded by {replacement.code}.",
        ))
        session.commit()
    ctx.schedule_sync()
    return old
