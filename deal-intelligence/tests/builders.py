"""Test builders shared by every test module: a real V1Context on a temporary SQLite database with
fake memory and a fake model injected, and helpers that insert deals the way the seed does."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from deal_intelligence.api.v1.context import V1Context
from deal_intelligence.api.v1.db import (
    Database, Deal, DealSignals, Interaction, Play, PlayStats, Stakeholder, utcnow,
)
from deal_intelligence.config import Settings


class NullMemory:
    """Stands in for Hindsight when a test does not care about memory."""

    async def ensure_bank(self) -> None:
        return None

    async def retain(self, item) -> None:  # noqa: ANN001
        return None

    async def delete(self, code: str) -> None:
        return None

    async def recall(self, query: str, tags: list[str], limit: int) -> list:
        return []

    async def healthy(self) -> bool:
        return True

    async def extraction_mode(self) -> str | None:
        return "chunks"

    async def close(self) -> None:
        return None


def make_context(tmp_path, *, memory=None, lessons=None, llm=None, **settings_overrides) -> V1Context:
    settings = replace(Settings(), db_path=tmp_path / "deals.db", uploads_dir=tmp_path / "uploads", **settings_overrides)
    return V1Context(
        db=Database(settings.db_path),
        memory=memory if memory is not None else NullMemory(),
        lessons=lessons,
        settings_provider=lambda: settings,
        llm_provider=lambda _settings: llm,
    )


def add_play(db: Database, code: str, name: str, *, category: str = "process", addresses: list[str] | None = None) -> None:
    with db.session() as session:
        if session.get(Play, code) is None:
            session.add(Play(code=code, name=name, description=name, category=category, addresses=addresses or []))
            session.add(PlayStats(play_code=code))
            session.commit()


def add_deal(
    db: Database,
    *,
    name: str,
    account: str,
    industry: str = "fintech",
    segment: str = "mid_market",
    result: str = "open",
    loss_reason: str | None = None,
    stage: str | None = None,
    amount: int = 100_000,
    owner: str = "Priya Nair",
    opened_on: date = date(2026, 6, 1),
    closed_on: date | None = None,
    stakeholders: list[tuple] | None = None,  # (name, title, stance, engaged, economic_buyer)
    interactions: list[tuple] | None = None,  # (kind, occurred_on, text)
    objections: list[tuple] | None = None,  # (type, status)
    competitors: list[str] | None = None,
    plays: list[str] | None = None,
    promises: list[dict] | None = None,
    discount_requested: bool = False,
    signals_status: str = "ready",
) -> int:
    """Insert a deal with stakeholders, interactions and signals. Returns the deal id."""
    with db.session() as session:
        deal = Deal(
            name=name, account=account, industry=industry, segment=segment, amount=amount, owner=owner,
            stage=stage or ("closed" if result != "open" else "evaluation"), opened_on=opened_on,
            closed_on=closed_on if result != "open" else None, result=result, loss_reason=loss_reason,
            signals_status=signals_status,
        )
        session.add(deal)
        session.flush()
        for who in stakeholders or []:
            session.add(Stakeholder(
                deal_id=deal.id, name=who[0], title=who[1], stance=who[2], engaged=who[3],
                economic_buyer=who[4] if len(who) > 4 else False,
            ))
        for index, (kind, occurred_on, text) in enumerate(interactions or []):
            session.add(Interaction(
                deal_id=deal.id, kind=kind, occurred_on=occurred_on, text=text,
                sha256=f"{deal.id:04d}{index:04d}".ljust(64, "0"),
                hindsight_status="pending" if result == "open" else "skipped",
            ))
        session.add(DealSignals(
            deal_id=deal.id,
            objections=[{"type": t, "text": f"{t} concern", "status": s, "first_seen_on": opened_on.isoformat(), "evidence": []}
                        for t, s in (objections or [])],
            competitors=competitors or [],
            pricing={"discount_requested": discount_requested, "notes": ""},
            promises=promises or [],
            plays_used=plays or [],
            source="seed",
            updated_at=utcnow(),
        ))
        session.commit()
        return deal.id
