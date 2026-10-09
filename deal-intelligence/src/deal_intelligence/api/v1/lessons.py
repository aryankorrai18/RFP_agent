"""Hindsight as the learning memory.

The interactions bank (memory.py, chunks mode) holds closed-deal summaries and interactions word for
word. This module feeds a second bank, the LESSONS bank, with what happened to closed deals: one
outcome lesson per deal and one lesson per play used on it. Hindsight extracts facts from each lesson
(concise mode) and consolidates them into observations.

The lessons are used three ways:
1. Ranking (retrieval mode "hindsight"): recalled lessons about a candidate play break ties between
   plays that already passed the structural gate. They never lift a play or deal past it.
2. Deal-type brief: Hindsight Reflect can answer "what do we know about deals like this one?" (`reflect_deal_type`, shown on the Memory evidence tab).
3. Playbook: a Hindsight mental model answering "what wins, what loses, and why?".

Evidence boundary: lessons never become citations in a brief; the brief cites interactions only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING, Protocol

import aiohttp
from hindsight_client import Hindsight
from hindsight_client_api.exceptions import ApiException
from sqlalchemy import select

from .db import QUALITY_LOSSES, Deal, Lesson, Play
from .memory import MemoryUnavailable
from .ranking import (
    LOSS_LABELS, STATUS_PHRASES, DealFacts, LessonEvidence, deal_facts, label, segment_label,
)

if TYPE_CHECKING:
    from .db import Database

PLAYBOOK_ID = "deal-playbook"
PLAYBOOK_QUERY = (
    "Across all closed deals: which plays, taken at which point, tend to win which kinds of deals, which "
    "objections and stakeholder situations tend to lose them, and what should an account executive do "
    "differently? Give concrete, actionable guidance and name the deal and play IDs."
)
BANK_MISSION = (
    "Remember what happened to this company's sales deals: the situation of each deal (segment, industry, "
    "objections and whether they were resolved, competitors, champion and economic buyer), which plays the "
    "team used, whether the deal was won or lost and why. Learn which plays win for which kinds of deals."
)
REFLECT_MISSION = (
    "You advise an account executive before a call. Base every statement on remembered deal outcomes, name "
    "the deal and play IDs involved, say when a loss reason (such as price) is not something a play could "
    "have changed, and never invent customer facts."
)
SYNC_BATCH = 20
_CONNECTION_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, OSError)


def play_tag(code: str) -> str:
    return f"play:{code}"


def deal_tag(code: str) -> str:
    return f"deal:{code}"


# --- Hindsight client -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LessonHit:
    text: str
    tags: list[str]
    document_id: str | None
    type: str | None
    rank: int


@dataclass(frozen=True)
class LessonBrief:
    text: str
    based_on: list[dict]


class LessonsMemory(Protocol):
    async def retain(self, items: list[dict]) -> None: ...

    async def recall(self, query: str, tags: list[str], limit: int = 20) -> list[LessonHit]: ...

    async def reflect(self, query: str, tags: list[str] | None = None) -> LessonBrief: ...

    async def playbook(self, refresh: bool = False) -> dict | None: ...

    async def close(self) -> None: ...


class HindsightLessons:
    """The lessons bank. Uses Hindsight's LLM extraction (concise mode) and observations, which
    consume Hindsight credits; calls are batched and the brief and playbook are only refreshed on request."""

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
                self.bank, name="Deal lessons", mission=BANK_MISSION, retain_mission=BANK_MISSION,
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

    async def reflect(self, query: str, tags: list[str] | None = None) -> LessonBrief:
        await self.ensure_bank()
        response = await self._call("reflect", lambda: self.client.areflect(
            self.bank, query=query, budget="low", tags=tags or None, tags_match="any", include_facts=True,
        ))
        memories = []
        based_on = getattr(response, "based_on", None)
        for memory in (getattr(based_on, "memories", None) or [])[:12]:
            memories.append({"id": memory.id, "text": memory.text, "type": memory.type})
        return LessonBrief(text=response.text or "", based_on=memories)

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
                self.bank, name="Deal playbook", source_query=PLAYBOOK_QUERY, id=PLAYBOOK_ID, max_tokens=900,
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


def signal_for_outcome(result: str, loss_reason: str | None) -> str:
    if result == "won":
        return "positive"
    if result == "lost" and loss_reason in QUALITY_LOSSES:
        return "negative"
    return "neutral"  # price, no decision, timing, champion left, competitor: no verdict on the plays


def _kind(facts: DealFacts) -> str:
    return " ".join(p for p in (segment_label(facts.segment), facts.industry) if p)


def _article(word: str) -> str:
    return "an" if word.lower().startswith(("a", "e", "i", "o", "u", "sso")) else "a"


def _outcome_phrase(result: str, loss_reason: str | None) -> str:
    if result == "won":
        return "was won"
    reason = LOSS_LABELS.get(loss_reason or "", loss_reason)
    return f"was lost because of {reason}" if reason else "was lost"


def _situation(facts: DealFacts) -> str:
    parts = []
    if facts.objections:
        text = "; ".join(f"{_article(label(t))} {label(t)} objection {STATUS_PHRASES.get(s, s)}" for t, s in facts.objections)
        parts.append(text[0].upper() + text[1:] + ".")
    if facts.competitors:
        parts.append(f"The competitor was {', '.join(facts.competitors)}.")
    parts.append({"champion_engaged": "A champion was engaged.", "champion_silent": "The champion went silent.",
                  "none": "There was no champion."}[facts.sponsor_state])
    parts.append({"engaged": "The economic buyer was engaged.", "silent": "The economic buyer was not engaged.",
                  "none": "No economic buyer was identified."}[facts.economic_buyer])
    return " ".join(parts)


def _plays(facts: DealFacts, names: dict[str, str]) -> str:
    if not facts.plays_used:
        return "No plays were recorded."
    return "Plays used: " + ", ".join(f"{p} ({names[p]})" if p in names else p for p in facts.plays_used) + "."


def _base_tags(kind: str, signal: str, facts: DealFacts) -> list[str]:
    tags = [f"kind:{kind}", f"signal:{signal}", deal_tag(facts.code), f"result:{facts.result}"]
    if facts.industry_key:
        tags.append(f"industry:{facts.industry_key}")
    if facts.segment:
        tags.append(f"segment:{facts.segment}")
    tags += [f"objection:{t}" for t in sorted(facts.objection_types)]
    tags += [f"competitor:{c}" for c in sorted(facts.competitor_slugs)]
    if facts.loss_reason:
        tags.append(f"loss:{facts.loss_reason}")
    return tags


def outcome_lesson_text(facts: DealFacts, names: dict[str, str]) -> str:
    kind = _kind(facts)
    return (f"Deal {facts.code}, {_article(kind or 'deal')} {kind + ' ' if kind else ''}deal, "
            f"{_outcome_phrase(facts.result, facts.loss_reason)}. {_situation(facts)} {_plays(facts, names)}")


def play_lesson_text(facts: DealFacts, play: str, names: dict[str, str], signal: str) -> str:
    kind = _kind(facts)
    reason = LOSS_LABELS.get(facts.loss_reason or "", facts.loss_reason) or "an unrecorded reason"
    name = f" ({names[play]})" if play in names else ""
    verdict = {
        "positive": "The deal was won, which counts in favour of the play.",
        "negative": f"The deal was lost because of {reason}, "
                    "a problem a play could plausibly have changed, which counts against the play.",
        "neutral": f"The deal was lost because of {reason}, "
                   "which no play could have changed, so this says little about the play.",
    }[signal]
    return (f"Play {play}{name} was used in deal {facts.code}, {_article(kind or 'deal')} {kind + ' ' if kind else ''}deal. "
            f"{_situation(facts)} {verdict}")


def _at(day: date | None, fallback: datetime) -> datetime:
    return datetime.combine(day, time(12, 0), tzinfo=UTC) if day else fallback


def collect_lessons(db: Database) -> int:
    """Make the lesson rows match the closed deals: create what is missing, rewrite what changed (a
    re-recorded outcome) and drop what no longer applies. Deterministic and idempotent: a second run
    returns 0. Returns the number of rows created, changed or removed."""
    changed = 0
    with db.session() as session:
        names = {p.code: p.name for p in session.scalars(select(Play))}
        wanted: dict[str, dict] = {}
        deals = session.scalars(
            select(Deal).where(Deal.status == "active", Deal.result.in_(("won", "lost"))).order_by(Deal.id)
        ).all()
        for deal in deals:
            facts = deal_facts(deal, deal.signals, list(deal.stakeholders))
            signal = signal_for_outcome(deal.result, deal.loss_reason)
            when = _at(deal.closed_on, deal.updated_at)
            wanted[f"outcome:{deal.code}"] = dict(
                deal_id=deal.id, play_code=None, signal=signal, text=outcome_lesson_text(facts, names),
                tags=_base_tags("deal_outcome", signal, facts), happened_at=when)
            for play in dict.fromkeys(facts.plays_used):
                wanted[f"play:{deal.code}:{play}"] = dict(
                    deal_id=deal.id, play_code=play, signal=signal, text=play_lesson_text(facts, play, names, signal),
                    tags=_base_tags("play_result", signal, facts) + [play_tag(play)], happened_at=when)

        existing = {row.key: row for row in session.scalars(select(Lesson))}
        for key, values in wanted.items():
            row = existing.get(key)
            if row is None:
                session.add(Lesson(key=key, **values))
                changed += 1
            elif (row.text, row.signal, list(row.tags)) != (values["text"], values["signal"], values["tags"]):
                for field_name, value in values.items():
                    setattr(row, field_name, value)
                row.hindsight_status, row.hindsight_attempts, row.hindsight_error = "pending", 0, None
                changed += 1  # Hindsight replaces the lesson: the document_id is the same
        for key, row in existing.items():
            if key not in wanted:  # the deal was deleted or reopened, or the play was unticked
                session.delete(row)
                changed += 1
        session.commit()
    return changed


@dataclass
class LessonSyncReport:
    retained: int = 0
    failed: int = 0
    pending: int = 0
    collected: int = 0
    errors: list[str] = field(default_factory=list)


def lesson_document_id(key: str) -> str:
    return f"lesson-{key}"


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
            items = [{"content": lesson.text, "timestamp": lesson.happened_at,
                      "document_id": lesson_document_id(lesson.key), "tags": lesson.tags,
                      "context": "Sales team record of what happened to a closed deal and the plays used",
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
        report.pending = pending_lessons(db)
    return report


def pending_lessons(db: Database) -> int:
    with db.session() as session:
        return len(session.scalars(select(Lesson.id).where(Lesson.hindsight_status == "pending")).all())


# --- ranking signals ---------------------------------------------------------------------------------


def play_signals(
    hits: list[LessonHit],
    codes: set[str],
    *,
    valid_deals: set[str] | None = None,
    valid_documents: set[str] | None = None,
) -> dict[str, LessonEvidence]:
    """Net lesson evidence per candidate play. Several facts extracted from one lesson count once and
    higher-ranked lessons count slightly more. When `valid_*` are given, a hit counts only if SQLite
    still has the lesson (its document id) and every deal it names (its deal: tags)."""
    out: dict[str, LessonEvidence] = {}
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        tags = set(hit.tags)
        deals = {t.removeprefix("deal:") for t in tags if t.startswith("deal:")}
        if valid_deals is not None and not deals <= valid_deals:
            continue
        if valid_documents is not None and hit.document_id and hit.document_id not in valid_documents:
            continue
        signal = "positive" if "signal:positive" in tags else "negative" if "signal:negative" in tags else "neutral"
        weight = 1.0 / (1 + 0.1 * (hit.rank - 1))
        for tag in tags:
            if not tag.startswith("play:"):
                continue
            code = tag.removeprefix("play:")
            if code not in codes:
                continue
            key = (code, hit.document_id or hit.text)
            if key in seen:
                continue
            seen.add(key)
            entry = out.setdefault(code, LessonEvidence())
            if signal == "positive":
                entry.positive += 1
                entry.net += weight
            elif signal == "negative":
                entry.negative += 1
                entry.net -= weight
            else:
                entry.neutral += 1
            if hit.document_id and len(entry.evidence) < 5:
                entry.evidence.append(hit.document_id)
    return out


# --- reflect and playbook helpers -------------------------------------------------------------------


def deal_type_query(facts: DealFacts) -> str:
    kind = _kind(facts) or "this kind of"
    return (f"We are preparing for a {kind} deal. What do we remember about similar deals: which won or lost, "
            "why, and which plays made a difference? Name the deal and play IDs. Be brief: at most 10 bullet "
            "points. A loss reason such as price belongs to the whole deal, so mark any explanation of why a "
            "single play mattered as your inference, not a recorded fact. If nothing is recorded for this "
            "kind of deal, say so in one line.")


def deal_type_tags(facts: DealFacts) -> list[str]:
    tags = [f"industry:{facts.industry_key}"] if facts.industry_key else []
    if facts.segment:
        tags.append(f"segment:{facts.segment}")
    tags += [f"objection:{t}" for t in sorted(facts.objection_types)]
    return tags


async def reflect_deal_type(lessons: LessonsMemory, facts: DealFacts) -> LessonBrief:
    return await lessons.reflect(deal_type_query(facts), deal_type_tags(facts))


async def read_playbook(lessons: LessonsMemory, refresh: bool = False) -> dict | None:
    return await lessons.playbook(refresh=refresh)
