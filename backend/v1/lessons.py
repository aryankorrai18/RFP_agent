"""Hindsight as the learning memory.

The library bank (chunks mode, memory.py) holds approved answer text word for word so drafts can
quote it. This module feeds a second bank, the LESSONS bank, with what happened to those answers:
which proposals they were in and whether those were won or lost, what reviewers did with drafts that
cited them, debrief scores, and superseded answers. Hindsight extracts facts from each lesson and
consolidates them into observations.

The lessons are then used three ways:
1. Ranking (retrieval mode "hindsight"): for each question, recall the lessons about the candidate
   answers; positive and negative lessons move answers up or down.
2. Client brief: Hindsight Reflect answers "what do we know about this client and industry?".
3. Playbook: a Hindsight mental model answering "what wins, what loses, what do reviewers change?".

Evidence boundary: lessons never reach the drafting prompt. They decide which approved answers are
offered and inform the reviewer; drafts still cite only facts and approved answers.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING, Any, Protocol

import aiohttp
from hindsight_client import Hindsight
from hindsight_client_api.exceptions import ApiException
from sqlalchemy import select

from .db import Answer, Lesson, MemoryEvent, PastProposal, Project
from .memory import MemoryUnavailable

if TYPE_CHECKING:
    from .db import Database

QUALITY_LOSSES = {"technical fit", "technical_fit", "response quality", "response_quality"}
PLAYBOOK_ID = "rfp-playbook"
PLAYBOOK_QUERY = (
    "Across all past proposals, reviews and debriefs: which kinds of RFP answers win, which lose and why, "
    "and what do reviewers repeatedly change? Give concrete, actionable guidance with examples."
)
BANK_MISSION = (
    "Remember what happened to this company's RFP answers: which proposals they were used in, whether those "
    "proposals were won or lost and why, what reviewers accepted, edited or rejected, and debrief feedback. "
    "Learn which kinds of answers win for which clients and industries."
)
REFLECT_MISSION = (
    "You advise a proposal writer. Base every statement on remembered outcomes and reviews, name the clients "
    "and answer IDs involved, and never invent product facts."
)
SYNC_BATCH = 20
_CONNECTION_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, OSError)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def client_tag(client: str) -> str:
    return f"client:{slug(client)}"


def industry_tag(industry: str) -> str:
    return f"industry:{slug(industry)}"


def answer_tag(code: str) -> str:
    return f"answer:{code}"


# --- Hindsight client -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LessonHit:
    text: str
    tags: list[str]
    document_id: str | None
    type: str | None
    rank: int


@dataclass(frozen=True)
class Brief:
    text: str
    based_on: list[dict]


class LessonsMemory(Protocol):
    async def retain(self, items: list[dict]) -> None: ...

    async def recall(self, query: str, tags: list[str], limit: int = 20) -> list[LessonHit]: ...

    async def reflect(self, query: str, tags: list[str] | None = None) -> Brief: ...

    async def playbook(self, refresh: bool = False) -> dict | None: ...

    async def close(self) -> None: ...


class HindsightLessons:
    """The lessons bank. Uses Hindsight's LLM extraction (concise mode) and observations, which
    consume Hindsight credits; calls are batched and the brief and playbook are cached."""

    def __init__(self, base_url: str, bank: str, api_key: str | None = None, timeout: float = 60.0):
        self.base_url = base_url
        self.bank = bank
        self._api_key = api_key or None
        self.timeout = timeout
        self._client: Hindsight | None = None
        self._bank_ready = False

    @property
    def client(self) -> Hindsight:
        if self._client is None:
            self._client = Hindsight(base_url=self.base_url, api_key=self._api_key, timeout=self.timeout, max_attempts=2)
        return self._client

    async def _call(self, what: str, coro_factory):  # noqa: ANN001, ANN202
        try:
            return await coro_factory()
        except ApiException as exc:
            raise MemoryUnavailable(f"Hindsight {what} failed: {exc.status} {exc.reason}") from exc
        except _CONNECTION_ERRORS as exc:
            raise MemoryUnavailable(f"Hindsight is not reachable at {self.base_url}") from exc

    async def ensure_bank(self) -> None:
        if self._bank_ready:
            return
        try:
            await self.client.acreate_bank(
                self.bank, name="RFP lessons", mission=BANK_MISSION, retain_mission=BANK_MISSION,
                retain_extraction_mode="concise", enable_observations=True, reflect_mission=REFLECT_MISSION,
            )
        except ApiException as exc:
            if exc.status not in (400, 409):  # already exists
                raise MemoryUnavailable(f"Hindsight refused to create bank {self.bank!r}: {exc.status}") from exc
        except _CONNECTION_ERRORS as exc:
            raise MemoryUnavailable(f"Hindsight is not reachable at {self.base_url}") from exc
        self._bank_ready = True

    async def retain(self, items: list[dict]) -> None:
        await self.ensure_bank()
        await self._call("retain", lambda: self.client.aretain_batch(self.bank, items=items, retain_async=True))

    async def recall(self, query: str, tags: list[str], limit: int = 20) -> list[LessonHit]:
        await self.ensure_bank()
        response = await self._call("recall", lambda: self.client.arecall(
            self.bank, query=query, tags=tags, tags_match="any", budget="mid", max_tokens=3000,
        ))
        hits = []
        for result in (response.results or [])[:limit]:
            hits.append(LessonHit(text=result.text, tags=list(result.tags or []), document_id=result.document_id,
                                  type=result.type, rank=len(hits) + 1))
        return hits

    async def reflect(self, query: str, tags: list[str] | None = None) -> Brief:
        await self.ensure_bank()
        response = await self._call("reflect", lambda: self.client.areflect(
            self.bank, query=query, budget="low", tags=tags or None, tags_match="any", include_facts=True,
        ))
        memories = []
        based_on = getattr(response, "based_on", None)
        for memory in (getattr(based_on, "memories", None) or [])[:12]:
            memories.append({"id": memory.id, "text": memory.text, "type": memory.type})
        return Brief(text=response.text or "", based_on=memories)

    async def playbook(self, refresh: bool = False) -> dict | None:
        await self.ensure_bank()
        try:
            model = await self.client.aget_mental_model(self.bank, PLAYBOOK_ID, detail="content")
        except ApiException as exc:
            if exc.status != 404:
                raise MemoryUnavailable(f"Hindsight mental model failed: {exc.status}") from exc
            if not refresh:
                return None  # reading is free; creating and refreshing use Hindsight credits
            await self._call("mental model create", lambda: self.client.acreate_mental_model(
                self.bank, name="RFP playbook", source_query=PLAYBOOK_QUERY, id=PLAYBOOK_ID, max_tokens=900,
            ))
            refresh = True
            model = None
        except _CONNECTION_ERRORS as exc:
            raise MemoryUnavailable(f"Hindsight is not reachable at {self.base_url}") from exc
        if refresh:
            await self._call("mental model refresh", lambda: self.client.arefresh_mental_model(self.bank, PLAYBOOK_ID))
            model = await self._call("mental model read", lambda: self.client.aget_mental_model(
                self.bank, PLAYBOOK_ID, detail="content"))
        if model is None:
            return None
        data = model if isinstance(model, dict) else model.to_dict()
        return {"content": data.get("content"), "last_refreshed_at": str(data.get("last_refreshed_at") or "") or None,
                "is_stale": data.get("is_stale")}

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None


# --- building lessons from what SQLite already records -------------------------------------------


def _signal_for_outcome(result: str, loss_reason: str | None) -> str:
    if result == "won":
        return "positive"
    if result == "lost" and (loss_reason or "").strip().lower() in QUALITY_LOSSES:
        return "negative"
    return "neutral"


def _outcome_phrase(result: str, loss_reason: str | None) -> str:
    # Neutral wording: the same code writes lessons for the demo company and for a real one.
    if result == "won":
        return "which was won"
    if result == "lost":
        return f"which was lost on {loss_reason}" if loss_reason else "which was lost"
    return "whose outcome is not known"


def _quote(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _at(day: date | None, fallback: datetime) -> datetime:
    return datetime.combine(day, time(12, 0), tzinfo=UTC) if day else fallback


def _tags(signal: str, kind: str, code: str | None, client: str | None, industry: str | None, extra=()) -> list[str]:  # noqa: ANN001
    tags = [f"kind:{kind}", f"signal:{signal}"]
    if code:
        tags.append(answer_tag(code))
    if client:
        tags.append(client_tag(client))
    if industry:
        tags.append(industry_tag(industry))
    tags.extend(extra)
    return tags


REVIEW = re.compile(r"was (accepted|edited|rewritten|rejected)(?: \(([^)]*)\))?")
NEGATIVE_REASONS = {"outdated", "incorrect", "wrong_product", "too_vague"}


def collect_lessons(db: Database) -> int:
    """Create lesson rows for anything not yet described. Safe to run repeatedly."""
    created = 0
    with db.session() as session:
        existing = set(session.scalars(select(Lesson.key)))

        def add(key: str, **values: Any) -> None:
            nonlocal created
            if key in existing:
                return
            session.add(Lesson(key=key, **values))
            existing.add(key)
            created += 1

        # 1. Each approved answer imported from a past proposal with a known outcome.
        rows = session.execute(
            select(Answer, PastProposal).join(PastProposal, Answer.past_proposal_id == PastProposal.id)
            .where(PastProposal.result.in_(("won", "lost", "no_decision")))
        ).all()
        for answer, proposal in rows:
            signal = _signal_for_outcome(proposal.result, proposal.loss_reason)
            when = proposal.submitted_on.strftime("%B %Y") if proposal.submitted_on else "an earlier bid"
            text = (f"Answer {answer.code} to the question '{_quote(answer.question, 120)}' was submitted in the "
                    f"{proposal.client or 'unnamed client'} ({proposal.industry or 'unknown industry'}) proposal in "
                    f"{when}, {_outcome_phrase(proposal.result, proposal.loss_reason)}. The answer said: "
                    f"'{_quote(answer.answer)}'")
            add(f"answer:{answer.id}:proposal", answer_id=answer.id, signal=signal, text=text,
                tags=_tags(signal, "proposal_outcome", answer.code, proposal.client, proposal.industry,
                           [f"result:{proposal.result}"]),
                happened_at=_at(proposal.submitted_on, answer.created_at))

        # 2. Events from the learning journal: reviews, outcomes, debriefs, superseding.
        events = session.scalars(select(MemoryEvent).order_by(MemoryEvent.id)).all()
        for event in events:
            project = session.get(Project, event.project_id) if event.project_id else None
            client = project.client if project else None
            industry = project.industry if project else None
            answer = session.get(Answer, event.answer_id) if event.answer_id else None
            label = project.name if project and project.name.lower().endswith(("rfp", "questionnaire")) \
                else f"{project.name} RFP" if project else ""
            where = f"the {label} ({client or 'unnamed client'}, {industry or 'unknown industry'})" if project else "a project"
            if event.kind == "review_signal" and answer is not None:
                match = REVIEW.search(event.detail)
                action = match.group(1) if match else "reviewed"
                reasons = [r.strip() for r in (match.group(2) or "").split(",") if r.strip()] if match else []
                signal = ("negative" if action in ("rejected", "rewritten") or set(reasons) & NEGATIVE_REASONS
                          else "positive" if action == "accepted" else "neutral")
                text = (f"While answering {where}, a reviewer {action} a draft that cited answer {answer.code} "
                        f"('{_quote(answer.question, 120)}')" + (f", reason: {', '.join(reasons)}" if reasons else "")
                        + ".")
                add(f"event:{event.id}", answer_id=answer.id, project_id=event.project_id, signal=signal, text=text,
                    tags=_tags(signal, "review", answer.code, client, industry, [f"action:{action}"]),
                    happened_at=event.created_at)
            elif event.kind == "review_signal" and project is not None:
                # A review of a draft that cited no past answer: a lesson about the question and client.
                match = REVIEW.search(event.detail)
                action = match.group(1) if match else "reviewed"
                reasons = [r.strip() for r in (match.group(2) or "").split(",") if r.strip()] if match else []
                signal = ("negative" if action in ("rejected", "rewritten") or set(reasons) & NEGATIVE_REASONS
                          else "positive" if action == "accepted" else "neutral")
                add(f"event:{event.id}", project_id=event.project_id, signal=signal,
                    text=f"While answering {where}, a reviewer found: {event.detail}",
                    tags=_tags(signal, "review", None, client, industry, [f"action:{action}"]),
                    happened_at=event.created_at)
            elif event.kind == "superseded" and answer is not None:
                text = f"Answer {answer.code} ('{_quote(answer.question, 120)}') was marked outdated. {event.detail}"
                add(f"event:{event.id}", answer_id=answer.id, signal="negative", text=text,
                    tags=_tags("negative", "superseded", answer.code, answer.client, answer.industry),
                    happened_at=event.created_at)
            elif event.kind in ("project_outcome", "debrief") and project is not None:
                from .outcomes import _source_ids  # the answers the project's final drafts cited

                if event.kind == "project_outcome":
                    match = re.search(r"recorded as (\w+)(?: \(([^)]*)\))?", event.detail)
                    result, reason = (match.group(1), match.group(2)) if match else ("unknown", None)
                    signal = _signal_for_outcome(result, reason)
                    section, verdict = None, _outcome_phrase(result, reason)
                else:
                    match = re.search(r"Debrief for (.*?)(?: scored (\d)/5)?\.$", event.detail)
                    section = match.group(1) if match and match.group(1) != "the project" else None
                    score = int(match.group(2)) if match and match.group(2) else None
                    signal = "positive" if score and score >= 4 else "negative" if score and score <= 2 else "neutral"
                    verdict = f"whose debrief scored {score}/5" if score else "which received debrief comments"
                for answer_id in sorted(_source_ids(project, section)):
                    cited = session.get(Answer, answer_id)
                    if cited is None:
                        continue
                    text = (f"Answer {cited.code} ('{_quote(cited.question, 120)}') was used in {where}"
                            + (f", section {section}" if section else "") + f", {verdict}.")
                    add(f"event:{event.id}:answer:{answer_id}", answer_id=answer_id, project_id=project.id,
                        signal=signal, text=text,
                        tags=_tags(signal, event.kind, cited.code, client, industry), happened_at=event.created_at)
            elif event.kind == "client_preference" and client:
                text = f"Reviewers working on RFPs for {client} ({industry or 'unknown industry'}): {event.detail}"
                add(f"event:{event.id}", project_id=event.project_id, signal="neutral", text=text,
                    tags=_tags("neutral", "client_preference", None, client, industry), happened_at=event.created_at)
        session.commit()
    return created


@dataclass
class LessonSyncReport:
    retained: int = 0
    failed: int = 0
    pending: int = 0
    collected: int = 0
    errors: list[str] = field(default_factory=list)


async def sync_lessons(db: Database, lessons: LessonsMemory, lock: asyncio.Lock) -> LessonSyncReport:
    report = LessonSyncReport()
    async with lock:
        report.collected = collect_lessons(db)
        with db.session() as session:
            pending = session.scalars(
                select(Lesson).where(Lesson.hindsight_status == "pending").order_by(Lesson.id)
            ).all()
        for start in range(0, len(pending), SYNC_BATCH):
            batch = pending[start:start + SYNC_BATCH]
            items = [{"content": lesson.text, "timestamp": lesson.happened_at, "document_id": f"lesson-{lesson.key}",
                      "tags": lesson.tags, "context": "Proposal team record of what happened to an RFP answer",
                      "metadata": {"lesson_key": lesson.key, "signal": lesson.signal}} for lesson in batch]
            try:
                await lessons.retain(items)
            except MemoryUnavailable as exc:
                report.failed += len(batch)
                report.errors.append(str(exc))
                with db.session() as session:
                    for lesson in batch:
                        row = session.get(Lesson, lesson.id)
                        row.hindsight_attempts += 1
                        row.hindsight_error = str(exc)
                    session.commit()
                break  # try again at the next sync
            with db.session() as session:
                for lesson in batch:
                    row = session.get(Lesson, lesson.id)
                    row.hindsight_status, row.hindsight_error = "retained", None
                session.commit()
            report.retained += len(batch)
        with db.session() as session:
            report.pending = len(session.scalars(select(Lesson.id).where(Lesson.hindsight_status == "pending")).all())
    return report


def pending_lessons(db: Database) -> int:
    with db.session() as session:
        return len(session.scalars(select(Lesson.id).where(Lesson.hindsight_status == "pending")).all())


# --- ranking signals ---------------------------------------------------------------------------------


@dataclass
class AnswerLessons:
    net: float = 0.0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    evidence: list[str] = field(default_factory=list)

    @property
    def factor(self) -> float:
        """Multiplier on the answer's score: strong enough that two positive lessons let the second
        search result overtake a first result with one negative lesson."""
        return round(max(0.25, min(2.5, 1.0 + 0.5 * self.net)), 4)


def answer_signals(hits: list[LessonHit], codes: set[str]) -> dict[str, AnswerLessons]:
    """Net lesson signal per candidate answer. Several facts extracted from one lesson count once,
    and higher-ranked lessons count slightly more."""
    out: dict[str, AnswerLessons] = {}
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        tags = set(hit.tags)
        signal = "positive" if "signal:positive" in tags else "negative" if "signal:negative" in tags else "neutral"
        weight = 1.0 / (1 + 0.1 * (hit.rank - 1))
        for tag in tags:
            if not tag.startswith("answer:"):
                continue
            code = tag.removeprefix("answer:")
            if code not in codes:
                continue
            key = (code, hit.document_id or hit.text)
            if key in seen:
                continue
            seen.add(key)
            entry = out.setdefault(code, AnswerLessons())
            if signal == "positive":
                entry.positive += 1
                entry.net += weight
            elif signal == "negative":
                entry.negative += 1
                entry.net -= weight
            else:
                entry.neutral += 1
            if len(entry.evidence) < 3:
                entry.evidence.append(hit.text)
    return out


def brief_query(client: str | None, industry: str | None) -> str:
    who = f"{client} ({industry})" if client and industry else client or industry or "this client"
    return (f"We are writing an RFP response for {who}. What do we remember about this client and similar "
            f"{industry or ''} clients: which past answers won or lost, why, and what did reviewers change? "
            "Give specific guidance and name the answer IDs. Be brief: at most 10 bullet points. "
            "A loss reason belongs to the whole proposal, so say which answers were in a lost or won "
            "proposal and mark any explanation of why a single answer mattered as your inference, not a recorded fact. "
            "If nothing is recorded for this exact client, say so in one line and use similar clients.")


def brief_tags(client: str | None, industry: str | None) -> list[str]:
    return [t for t in ([client_tag(client)] if client else []) + ([industry_tag(industry)] if industry else [])]

