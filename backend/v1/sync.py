"""Outbox: keeps Hindsight's copy in step with SQLite (design §12.2).

SQLite is always written first (the answer is real the moment it's approved) and the row is
marked `pending` or `pending_delete`. This step pushes those changes to Hindsight and retries
until it succeeds. A crash or an unreachable Hindsight can delay when an answer becomes
findable; it can never let a deleted answer reach a draft, because retrieval re-checks SQLite.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from sqlalchemy import select

from .db import Answer, Database, utcnow
from .memory import Memory, MemoryItem, MemoryUnavailable


@dataclass
class SyncReport:
    retained: int = 0
    deleted: int = 0
    failed: int = 0
    pending: int = 0


def pending_count(db: Database) -> int:
    with db.session() as session:
        return len(
            session.scalars(
                select(Answer.id).where(Answer.hindsight_status.in_(("pending", "pending_delete")))
            ).all()
        )


async def sync_outbox(db: Database, memory: Memory, lock: asyncio.Lock | None = None) -> SyncReport:
    lock = lock or asyncio.Lock()
    async with lock:  # one sync at a time
        return await _sync(db, memory)


async def _sync(db: Database, memory: Memory) -> SyncReport:
    report = SyncReport()
    with db.session() as session:
        rows = session.scalars(
            select(Answer).where(Answer.hindsight_status.in_(("pending", "pending_delete"))).order_by(Answer.id)
        ).all()

    for row in rows:
        try:
            if row.hindsight_status == "pending" and row.live:
                await memory.retain(
                    MemoryItem(
                        code=row.code,
                        question=row.question,
                        answer=row.answer,
                        client=row.client,
                        industry=row.industry,
                        timestamp=row.updated_at,
                    )
                )
                outcome, report.retained = "retained", report.retained + 1
            else:  # pending_delete, or a pending row that stopped being live
                await memory.delete(row.code)
                outcome, report.deleted = "deleted", report.deleted + 1
            _mark(db, row.id, status=outcome, error=None, synced_version=row.updated_at)
        except MemoryUnavailable as exc:
            report.failed += 1
            _mark(db, row.id, status=None, error=str(exc), attempt=True)
            break  # Hindsight is down: stop now, retry the rest on the next sync

    report.pending = pending_count(db)
    return report


def _mark(
    db: Database,
    answer_id: int,
    *,
    status: str | None,
    error: str | None,
    attempt: bool = False,
    synced_version=None,  # noqa: ANN001  (datetime as read before the sync)
) -> None:
    with db.session() as session:
        row = session.get(Answer, answer_id)
        if row is None:
            return
        # If the answer changed while we were syncing it, leave it pending so the newer
        # text is pushed next time instead of being marked done with the old one.
        if status is not None and (synced_version is None or row.updated_at == synced_version):
            row.hindsight_status = status
        if attempt:
            row.hindsight_attempts += 1
        row.hindsight_error = error
        session.commit()


def touch(answer: Answer) -> None:
    """Call whenever an answer's text or status changes, so the outbox re-syncs it."""
    answer.updated_at = utcnow()
