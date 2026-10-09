"""Outbox: keeps Hindsight's copy of deals in step with SQLite.

SQLite is always written first (a deal is real the moment it is closed) and the row is marked
`pending` or `pending_delete`. This step pushes those changes to Hindsight and retries until it
succeeds. A crash or an unreachable Hindsight can delay when a deal becomes findable; it can never
let a deleted deal reach a brief, because retrieval re-checks SQLite.

What is retained: a summary of each CLOSED, active deal that has signals, and each pending
interaction. Interactions marked `skipped` (unrelated closed history) are never sent.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import select

from .db import Database, Deal, DealSignals, Interaction, Play, utcnow
from .memory import Memory, MemoryItem, MemoryUnavailable, deal_summary_item, interaction_item

log = logging.getLogger(__name__)


@dataclass
class SyncReport:
    retained: int = 0
    deleted: int = 0
    failed: int = 0
    pending: int = 0


@dataclass
class _Work:
    model: type  # Deal or Interaction
    row_id: int
    code: str
    item: MemoryItem | None  # None means delete
    version: object  # updated_at as read before the sync


def _pending_work(db: Database) -> list[_Work]:
    """Every outbox row that can be sent now. Open deals' summaries are not due yet, and a closed deal
    waits for its signals, so neither counts as pending."""
    work: list[_Work] = []
    with db.session() as session:
        names = {p.code: p.name for p in session.scalars(select(Play))}
        deals = session.scalars(
            select(Deal).where(Deal.hindsight_status.in_(("pending", "pending_delete"))).order_by(Deal.id)
        ).all()
        for deal in deals:
            if deal.hindsight_status == "pending_delete" or not deal.live:
                work.append(_Work(Deal, deal.id, deal.code, None, deal.updated_at))
            elif deal.result != "open" and session.get(DealSignals, deal.id) is not None:
                item = deal_summary_item(deal, deal.signals, list(deal.stakeholders), names)
                work.append(_Work(Deal, deal.id, deal.code, item, deal.updated_at))
        interactions = session.scalars(
            select(Interaction).where(Interaction.hindsight_status.in_(("pending", "pending_delete"))).order_by(Interaction.id)
        ).all()
        for row in interactions:
            if row.hindsight_status == "pending_delete" or not row.deal.live:
                work.append(_Work(Interaction, row.id, row.code, None, row.updated_at))
            else:
                work.append(_Work(Interaction, row.id, row.code, interaction_item(row.deal, row), row.updated_at))
    return work


def pending_count(db: Database) -> int:
    return len(_pending_work(db))


async def sync_outbox(db: Database, memory: Memory, lock: asyncio.Lock | None = None) -> SyncReport:
    lock = lock or asyncio.Lock()
    async with lock:  # one sync at a time
        return await _sync(db, memory)


async def _sync(db: Database, memory: Memory) -> SyncReport:
    report = SyncReport()
    for work in _pending_work(db):
        try:
            if work.item is not None:
                await memory.retain(work.item)
                outcome, report.retained = "retained", report.retained + 1
            else:
                await memory.delete(work.code)
                outcome, report.deleted = "deleted", report.deleted + 1
            _mark(db, work.model, work.row_id, status=outcome, error=None, synced_version=work.version)
        except MemoryUnavailable as exc:
            report.failed += 1
            _mark(db, work.model, work.row_id, status=None, error=str(exc), attempt=True)
            break  # Hindsight is down: stop now, retry the rest on the next sync
    report.pending = pending_count(db)
    return report


def _mark(
    db: Database,
    model: type,
    row_id: int,
    *,
    status: str | None,
    error: str | None,
    attempt: bool = False,
    synced_version=None,  # noqa: ANN001  (datetime as read before the sync)
) -> None:
    with db.session() as session:
        row = session.get(model, row_id)
        if row is None:
            return
        # If the row changed while we were syncing it, leave it pending so the newer content is
        # pushed next time instead of being marked done with the old one.
        if status is not None and (synced_version is None or row.updated_at == synced_version):
            row.hindsight_status = status
        if attempt:
            row.hindsight_attempts += 1
        row.hindsight_error = error
        session.commit()


def touch(row: Deal | Interaction | DealSignals) -> None:
    """Call whenever a row's synced content changes, so the outbox re-syncs it."""
    row.updated_at = utcnow()
