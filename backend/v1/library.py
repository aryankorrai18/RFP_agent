"""The answer library: import past proposals, confirm their pairs, search and delete answers.

Nothing enters the library until a person confirms it (design §3.1). SQLite is written first;
the Hindsight copy follows through the outbox (design §12.2).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from ..llm import LLMError
from ..parser import ParseError, parse_document
from ..core import PipelineError
from .db import Answer, AnswerStats, Database, Document, Job, Pair, PastProposal, parse_answer_code
from .jobs import JobFailed, JobRunner
from .outcomes import outcome_credit
from .retrieval import retrieve
from .storage import store_document
from .sync import touch

if TYPE_CHECKING:
    from .context import V1Context

RESULTS = ("won", "lost", "no_decision", "unknown")
DECISIONS = ("kept", "edited", "dropped")
# A file can't be imported again while an earlier import of it is live; a failed or discarded one can be.
LIVE_STATUSES = ("uploaded", "extracting", "extracted", "confirmed")
DISCARDABLE_STATUSES = ("extracted", "failed")


def register(jobs: JobRunner) -> None:
    jobs.register("extract_pairs", run_extract_pairs, on_cancel=_extraction_stopped)


def _extraction_stopped(ctx: V1Context, job_id: int) -> None:
    with ctx.db.session() as session:
        proposal_id = session.get(Job, job_id).target_id
        proposal = session.get(PastProposal, proposal_id)
        if proposal is None or proposal.status not in ("uploaded", "extracting"):
            return
    _fail_proposal(
        ctx, proposal_id,
        "Stopped before the question-and-answer pairs were extracted. Discard this import, or import the file again.",
    )


# --- import ------------------------------------------------------------------------------------


def create_past_proposal(
    ctx: V1Context,
    *,
    filename: str,
    data: bytes,
    client: str | None,
    industry: str | None,
    submitted_on: date | None,
    result: str,
    loss_reason: str | None,
) -> tuple[PastProposal, Job]:
    if result not in RESULTS:
        raise PipelineError("invalid_request", f"result must be one of {', '.join(RESULTS)}", 422)
    digest = hashlib.sha256(data).hexdigest()
    with ctx.db.session() as session:
        existing = session.scalars(
            select(PastProposal).join(Document)
            .where(Document.sha256 == digest, PastProposal.status.in_(LIVE_STATUSES))
        ).first()
        if existing is not None:
            raise PipelineError(
                "already_imported",
                f"This file was already imported as past proposal #{existing.id} ({existing.status}). "
                "Discard that import first if you want to redo it.",
                409,
            )
    document = store_document(ctx, "past_proposal", filename, data)
    with ctx.db.session() as session:
        proposal = PastProposal(
            document_id=document.id,
            client=_clean(client),
            industry=_clean(industry),
            submitted_on=submitted_on,
            result=result,
            loss_reason=_clean(loss_reason),
            status="extracting",
        )
        session.add(proposal)
        session.commit()
    job = ctx.jobs.submit("extract_pairs", proposal.id)
    return proposal, job


def import_prepared_proposal(
    ctx: V1Context,
    *,
    filename: str,
    data: bytes,
    client: str | None,
    industry: str | None,
    submitted_on: date | None,
    result: str,
    loss_reason: str | None,
    pairs: list[dict],
) -> list[Answer]:
    """Import a past proposal whose question-and-answer pairs are already known (the demo seed),
    with no model call: the pairs are stored as extracted and confirmed as they are. Same checks,
    outcome credit and Hindsight sync as an ordinary import."""
    if result not in RESULTS:
        raise PipelineError("invalid_request", f"result must be one of {', '.join(RESULTS)}", 422)
    usable = [p for p in pairs if (p.get("question") or "").strip() and (p.get("answer") or "").strip()]
    if not usable:
        raise PipelineError("invalid_request", f"{filename} has no question-and-answer pairs to import.", 422)
    digest = hashlib.sha256(data).hexdigest()
    with ctx.db.session() as session:
        existing = session.scalars(
            select(PastProposal).join(Document)
            .where(Document.sha256 == digest, PastProposal.status.in_(LIVE_STATUSES))
        ).first()
        if existing is not None:
            raise PipelineError("already_imported", f"{filename} is already in the library (past proposal #{existing.id}).", 409)
    document = store_document(ctx, "past_proposal", filename, data)
    with ctx.db.session() as session:
        proposal = PastProposal(
            document_id=document.id, client=_clean(client), industry=_clean(industry), submitted_on=submitted_on,
            result=result, loss_reason=_clean(loss_reason), status="extracted",
        )
        session.add(proposal)
        session.flush()
        for order, pair in enumerate(usable, start=1):
            session.add(Pair(
                past_proposal_id=proposal.id, order=order, section=_clean(pair.get("section")),
                reference=_clean(pair.get("reference")), question=pair["question"].strip(), answer=pair["answer"].strip(),
            ))
        session.commit()
        pair_ids = [p.id for p in session.get(PastProposal, proposal.id).pairs]
    return confirm_pairs(ctx, proposal.id, [PairDecision(pair_id=i, decision="kept") for i in pair_ids])


async def run_extract_pairs(ctx: V1Context, job_id: int) -> None:
    with ctx.db.session() as session:
        job = session.get(Job, job_id)
        proposal = session.get(PastProposal, job.target_id)
        filename, path = proposal.document.filename, Path(proposal.document.stored_path)

    try:
        document = parse_document(filename, path.read_bytes(), ctx.settings.max_document_chars)
        result = await ctx.llm.extract_pairs(document)
    except (ParseError, LLMError) as exc:
        _fail_proposal(ctx, proposal.id, exc.message)
        raise JobFailed(exc.message) from exc

    pairs = [p for p in result.output.pairs if p.question.strip() and p.answer.strip()]
    if not pairs:
        message = f"No question-and-answer pairs were found in {filename}."
        _fail_proposal(ctx, proposal.id, message)
        raise JobFailed(message)

    with ctx.db.session() as session:
        # Replace, never append: a resumed job must not duplicate pairs.
        session.execute(delete(Pair).where(Pair.past_proposal_id == proposal.id))
        for order, pair in enumerate(pairs, start=1):
            session.add(
                Pair(
                    past_proposal_id=proposal.id,
                    order=order,
                    section=_clean(pair.section),
                    reference=_clean(pair.reference),
                    question=pair.question.strip(),
                    answer=pair.answer.strip(),
                )
            )
        row = session.get(PastProposal, proposal.id)
        row.status, row.error = "extracted", None
        job = session.get(Job, job_id)
        job.total = job.done = len(pairs)
        session.commit()


def _fail_proposal(ctx: V1Context, proposal_id: int, message: str) -> None:
    with ctx.db.session() as session:
        row = session.get(PastProposal, proposal_id)
        row.status, row.error = "failed", message
        session.commit()


def discard_proposal(ctx: V1Context, proposal_id: int) -> PastProposal:
    """Drop an import that was never confirmed (a duplicate, the wrong file, a failed parse)."""
    with ctx.db.session() as session:
        proposal = session.get(PastProposal, proposal_id)
        if proposal is None or proposal.status == "discarded":
            raise PipelineError("not_found", f"No past proposal {proposal_id}.", 404)
        if proposal.status not in DISCARDABLE_STATUSES:
            hint = " Delete its answers from the library instead." if proposal.status == "confirmed" else ""
            raise PipelineError("invalid_state", f"A '{proposal.status}' proposal can't be discarded.{hint}", 409)
        proposal.status = "discarded"
        session.commit()
    return proposal


# --- confirmation --------------------------------------------------------------------------


@dataclass
class PairDecision:
    pair_id: int
    decision: str
    question: str | None = None
    answer: str | None = None


def confirm_pairs(ctx: V1Context, proposal_id: int, decisions: list[PairDecision]) -> list[Answer]:
    with ctx.db.session() as session:
        proposal = session.get(PastProposal, proposal_id)
        if proposal is None:
            raise PipelineError("not_found", f"No past proposal {proposal_id}.", 404)
        if proposal.status != "extracted":
            raise PipelineError("invalid_state", f"This proposal is '{proposal.status}'; only extracted pairs can be confirmed.", 409)
        pairs = {pair.id: pair for pair in proposal.pairs}
        by_pair = {d.pair_id: d for d in decisions}
        missing = sorted(set(pairs) - set(by_pair))
        if missing:
            raise PipelineError("invalid_request", f"Decide every pair before confirming (missing: {missing}).", 422)

        created: list[Answer] = []
        for pair_id, pair in pairs.items():
            decision = by_pair[pair_id]
            if decision.decision not in DECISIONS:
                raise PipelineError("invalid_request", f"Decision must be one of {', '.join(DECISIONS)}.", 422)
            pair.decision = decision.decision
            if decision.decision == "dropped":
                continue
            if decision.decision == "edited":
                pair.question = (decision.question or pair.question).strip()
                pair.answer = (decision.answer or pair.answer).strip()
            if not pair.answer.strip():
                raise PipelineError("invalid_request", f"Pair {pair_id} has an empty answer; drop it instead.", 422)
            answer = Answer(
                question=pair.question,
                answer=pair.answer,
                source="library",
                past_proposal_id=proposal.id,
                client=proposal.client,
                industry=proposal.industry,
                hindsight_status="pending",
            )
            session.add(answer)
            created.append(answer)
        proposal.status = "confirmed"
        session.flush()
        credit = outcome_credit(proposal.result, proposal.loss_reason)
        for answer in created:
            session.add(AnswerStats(answer_id=answer.id, outcome_credit=credit))
        session.commit()
    ctx.schedule_sync()
    return created


def backfill_historical_outcomes(db: Database) -> int:
    """Seed V3 track records for V1 answers imported before the V3 tables existed."""
    with db.session() as session:
        rows = session.execute(
            select(Answer.id, PastProposal.result, PastProposal.loss_reason)
            .join(PastProposal, Answer.past_proposal_id == PastProposal.id)
            .outerjoin(AnswerStats, AnswerStats.answer_id == Answer.id)
            .where(AnswerStats.answer_id.is_(None))
        ).all()
        for answer_id, result, loss_reason in rows:
            session.add(AnswerStats(
                answer_id=answer_id,
                outcome_credit=outcome_credit(result, loss_reason),
            ))
        session.commit()
        return len(rows)


# --- browsing, search, deletion ----------------------------------------------------------------


async def search_answers(ctx: V1Context, query: str | None, limit: int = 50) -> tuple[list[Answer], list[dict], str | None]:
    """With a query: the same plain retrieval drafting uses. Without: the newest live answers."""
    if query and query.strip():
        settings = ctx.settings
        found = await retrieve(
            query.strip(), memory=ctx.memory, db=ctx.db, k=limit, mode=settings.retrieval_mode,
            freshness_half_life_days=settings.retrieval_freshness_half_life_days,
            lessons=ctx.lessons, relevance=settings.retrieval_relevance,
        min_share=settings.retrieval_relevance_min_share,
        )
        codes = [p.id for p in found.past_answers]
        ids = [parse_answer_code(c) for c in codes]
        with ctx.db.session() as session:
            rows = {a.id: a for a in session.scalars(select(Answer).where(Answer.id.in_(ids)))}
        return [rows[i] for i in ids if i in rows], found.retrieved, found.warning
    with ctx.db.session() as session:
        rows = session.scalars(
            select(Answer).where(Answer.status == "approved").order_by(Answer.id.desc()).limit(limit)
        ).all()
    return list(rows), [], None


def delete_answer(ctx: V1Context, code: str) -> Answer:
    answer_id = parse_answer_code(code)
    with ctx.db.session() as session:
        answer = session.get(Answer, answer_id) if answer_id else None
        if answer is None or answer.status != "approved":
            raise PipelineError("not_found", f"No library answer {code}.", 404)
        # SQLite first: the retrieval filter excludes it immediately, whatever Hindsight says.
        answer.status = "deleted"
        answer.hindsight_status = "pending_delete"
        touch(answer)
        session.commit()
    ctx.schedule_sync()
    return answer


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None
