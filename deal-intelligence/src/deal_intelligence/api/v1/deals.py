"""Deals: create one from uploaded emails and notes, add to it, view it, delete it.

SQLite is written first. The interactions of an open deal go to the Hindsight interactions bank
through the outbox (sync.py); signals are read from them by one model call (signals.py).
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import PurePath
from typing import TYPE_CHECKING

from sqlalchemy import select

from ...errors import PipelineError
from ...parsing.parser import parse_document
from .db import (
    INTERACTION_KINDS, SEGMENTS, STAGES, Deal, DealSignals, Interaction, MemoryEvent, Play, utcnow,
)
from .storage import store_document

if TYPE_CHECKING:
    from .context import V1Context

DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")
KIND_HINTS = (("call", "call_note"), ("meeting", "meeting"), ("crm", "crm_note"), ("note", "call_note"))


def _guess_kind(filename: str) -> str:
    name = PurePath(filename).stem.lower()
    if PurePath(filename).suffix.lower() == ".eml":
        return "email"
    for hint, kind in KIND_HINTS:
        if hint in name:
            return kind
    return "email"


def _guess_date(filename: str, today: date) -> date:
    match = DATE_IN_NAME.search(filename)
    if match:
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            pass
    return today


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _add_interaction(session, deal: Deal, *, kind: str, occurred_on: date, text: str, author: str | None,
                     subject: str | None, document_id: int | None) -> Interaction | None:
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    if session.scalar(select(Interaction.id).where(Interaction.deal_id == deal.id, Interaction.sha256 == digest)):
        return None
    interaction = Interaction(
        deal_id=deal.id, kind=kind, occurred_on=occurred_on, text=text.strip(), author=author, subject=subject,
        sha256=digest, document_id=document_id, hindsight_status="pending" if deal.is_open else "skipped",
    )
    session.add(interaction)
    session.flush()
    return interaction


def create_deal(
    ctx: V1Context, *, name: str, account: str, industry: str | None = None, segment: str | None = None,
    amount: int | None = None, owner: str | None = None, stage: str = "discovery",
    files: list[tuple[str, bytes]] | None = None, today: date | None = None,
) -> Deal:
    name, account = (name or "").strip(), (account or "").strip()
    if not name or not account:
        raise PipelineError("invalid_request", "A deal needs a name and an account.", 422)
    if segment and segment not in SEGMENTS:
        raise PipelineError("invalid_request", f"segment must be one of {', '.join(SEGMENTS)}.", 422)
    if stage not in STAGES or stage == "closed":
        raise PipelineError("invalid_request", f"stage must be one of {', '.join(s for s in STAGES if s != 'closed')}.", 422)
    today = today or date.today()
    # Read every file before anything is stored, so an unreadable or unsupported one leaves no empty deal behind.
    parsed_files = [(filename, data, parse_document(filename, data, ctx.settings.max_document_chars)) for filename, data in files or []]
    with ctx.db.session() as session:
        deal = Deal(
            name=name, account=account, industry=_clean(industry), segment=segment, amount=amount, owner=_clean(owner),
            stage=stage, opened_on=today, result="open", signals_status="none", hindsight_status="skipped",
        )
        session.add(deal)
        session.flush()
        session.add(MemoryEvent(kind="deal_added", deal_id=deal.id, detail=f"Added {deal.code}: {name} ({account})."))
        session.commit()
    for filename, data, parsed in parsed_files:
        _attach_file(ctx, deal.id, filename, data, parsed, today)
    ctx.schedule_sync()
    return deal


def add_files(ctx: V1Context, deal_id: int, files: list[tuple[str, bytes]], *, today: date | None = None) -> list[Interaction]:
    """Each file becomes one interaction (an email, call note or meeting note, guessed from its name)."""
    today = today or date.today()
    deal = _open_deal(ctx, deal_id)
    created: list[Interaction] = []
    for filename, data in files:
        parsed = parse_document(filename, data, ctx.settings.max_document_chars)
        interaction = _attach_file(ctx, deal.id, filename, data, parsed, today)
        if interaction is not None:
            created.append(interaction)
    ctx.schedule_sync()
    return created


def _attach_file(ctx: V1Context, deal_id: int, filename: str, data: bytes, parsed, today: date) -> Interaction | None:  # noqa: ANN001
    """Store the uploaded file and add its text as an interaction (None when the same note is already on the deal)."""
    document = store_document(ctx, "interaction_file", filename, data)
    with ctx.db.session() as session:
        row = session.get(Deal, deal_id)
        interaction = _add_interaction(
            session, row, kind=_guess_kind(filename), occurred_on=_guess_date(filename, today), text=parsed.text,
            author=None, subject=PurePath(filename).stem.replace("_", " ").replace("-", " ").strip() or None,
            document_id=document.id,
        )
        if interaction is not None:
            row.updated_at = utcnow()
            session.add(MemoryEvent(kind="interaction_added", deal_id=row.id, detail=f"Added {interaction.code} from {filename}."))
        session.commit()
        return interaction


def add_note(
    ctx: V1Context, deal_id: int, *, kind: str, text: str, occurred_on: date | None = None, author: str | None = None,
    subject: str | None = None,
) -> Interaction:
    """A pasted email or note."""
    if kind not in INTERACTION_KINDS:
        raise PipelineError("invalid_request", f"kind must be one of {', '.join(INTERACTION_KINDS)}.", 422)
    if not (text or "").strip():
        raise PipelineError("invalid_request", "The note is empty.", 422)
    deal = _open_deal(ctx, deal_id)
    with ctx.db.session() as session:
        row = session.get(Deal, deal.id)
        interaction = _add_interaction(
            session, row, kind=kind, occurred_on=occurred_on or date.today(), text=text, author=_clean(author),
            subject=_clean(subject), document_id=None,
        )
        if interaction is None:
            raise PipelineError("already_added", "This exact note is already on the deal.", 409)
        row.updated_at = utcnow()
        session.add(MemoryEvent(kind="interaction_added", deal_id=row.id, detail=f"Added {interaction.code} (pasted {kind})."))
        session.commit()
    ctx.schedule_sync()
    return interaction


def update_deal(ctx: V1Context, deal_id: int, fields: dict) -> Deal:
    """Change a deal's own details (name, account, industry, segment, amount, owner). Only the fields given change.
    Industry and segment feed the similar-deal matching, so a deal created without them can be completed later."""
    allowed = {"name", "account", "industry", "segment", "amount", "owner"}
    changes = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if "segment" in changes and changes["segment"] not in SEGMENTS:
        raise PipelineError("invalid_request", f"segment must be one of {', '.join(SEGMENTS)}.", 422)
    for key in ("name", "account"):
        if key in changes and not str(changes[key]).strip():
            raise PipelineError("invalid_request", f"{key} can't be empty.", 422)
    with ctx.db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        for key, value in changes.items():
            setattr(deal, key, value if key in ("segment", "amount") else _clean(str(value)))
        if changes:
            from .sync import touch

            touch(deal)
            if deal.hindsight_status == "retained":
                deal.hindsight_status = "pending"  # its summary in memory carries these details: re-sync it
            session.add(MemoryEvent(kind="deal_updated", deal_id=deal.id,
                                    detail=f"Updated {deal.code}: " + ", ".join(f"{k} = {v}" for k, v in changes.items()) + "."))
        session.commit()
        session.refresh(deal)
    if changes:
        ctx.schedule_sync()
    return deal


def delete_deal(ctx: V1Context, deal_id: int) -> None:
    """Soft delete: the row stays for audit, Hindsight's copies are removed through the outbox."""
    with ctx.db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        deal.status = "deleted"
        deal.updated_at = utcnow()
        if deal.hindsight_status in ("retained", "pending"):
            deal.hindsight_status = "pending_delete"
        for interaction in deal.interactions:
            if interaction.hindsight_status in ("retained", "pending"):
                interaction.hindsight_status = "pending_delete"
        session.add(MemoryEvent(kind="deal_deleted", deal_id=deal.id, detail=f"Deleted {deal.code}: {deal.name}."))
        session.commit()
    ctx.schedule_sync()


def _open_deal(ctx: V1Context, deal_id: int) -> Deal:
    with ctx.db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        if not deal.is_open:
            raise PipelineError("deal_closed", f"{deal.code} is already closed.", 409)
        return deal


# ---- Views ------------------------------------------------------------------------------------------

def deal_row(deal: Deal) -> dict:
    return {
        "id": deal.id, "code": deal.code, "name": deal.name, "account": deal.account, "industry": deal.industry,
        "segment": deal.segment, "amount": deal.amount, "stage": deal.stage, "owner": deal.owner,
        "opened_on": deal.opened_on.isoformat() if deal.opened_on else None,
        "closed_on": deal.closed_on.isoformat() if deal.closed_on else None,
        "result": deal.result, "loss_reason": deal.loss_reason, "signals_status": deal.signals_status,
        "signals_error": deal.signals_error, "interactions": len(deal.interactions),
        "last_activity": max((i.occurred_on for i in deal.interactions), default=None).isoformat()
        if deal.interactions else None,
    }


def deal_row_by_id(ctx: V1Context, deal_id: int) -> dict:
    with ctx.db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        return deal_row(deal)


def list_deals(ctx: V1Context) -> list[dict]:
    with ctx.db.session() as session:
        deals = session.scalars(select(Deal).where(Deal.status == "active").order_by(Deal.id)).all()
        return [deal_row(d) for d in deals]


def deal_detail(ctx: V1Context, deal_id: int) -> dict:
    with ctx.db.session() as session:
        deal = session.get(Deal, deal_id)
        if deal is None or not deal.live:
            raise PipelineError("not_found", f"No deal {deal_id}.", 404)
        signals: DealSignals | None = deal.signals
        plays = {p.code: p for p in session.scalars(select(Play)).all()}
        return {
            **deal_row(deal),
            "stakeholders": [
                {"name": s.name, "title": s.title, "stance": s.stance, "engaged": s.engaged,
                 "economic_buyer": s.economic_buyer, "note": s.note}
                for s in deal.stakeholders
            ],
            "interactions": [
                {"id": i.id, "code": i.code, "kind": i.kind, "occurred_on": i.occurred_on.isoformat(), "author": i.author,
                 "subject": i.subject, "text": i.text, "hindsight_status": i.hindsight_status}
                for i in deal.interactions
            ],
            "signals": None if signals is None else {
                "objections": signals.objections, "competitors": signals.competitors, "pricing": signals.pricing,
                "promises": signals.promises, "source": signals.source,
                "plays_used": [{"code": c, "name": plays[c].name if c in plays else c} for c in signals.plays_used],
            },
        }


def save_plays(ctx: V1Context, items: list[dict]) -> dict:
    """Add or update plays in this workspace's catalogue: a company's own sales moves. A play is matched to an existing one by
    its code, else by its name (case-insensitive); a new play gets the next free PLAY-NN code. Objection types the product does
    not know are left out and reported, never guessed; an unknown category becomes "process". Nothing is ever deleted here."""
    from .db import OBJECTION_TYPES, PLAY_CATEGORIES, PlayStats

    added: list[str] = []
    updated: list[str] = []
    ignored: set[str] = set()
    with ctx.db.session() as session:
        existing = {p.code: p for p in session.scalars(select(Play)).all()}
        by_name = {p.name.strip().lower(): p for p in existing.values()}
        numbers = [int(c.split("-")[1]) for c in existing if c.startswith("PLAY-") and c.split("-")[1].isdigit()]
        next_number = max(numbers, default=0) + 1
        for item in items:
            name = " ".join(str(item.get("name") or "").split())[:120]
            if not name:
                continue
            description = " ".join(str(item.get("description") or "").split()) or name
            category = str(item.get("category") or "process").strip().lower()
            category = category if category in PLAY_CATEGORIES else "process"
            wanted = [str(a).strip().lower().replace(" ", "_").replace("-", "_") for a in item.get("addresses") or []]
            addresses = [a for a in dict.fromkeys(wanted) if a in OBJECTION_TYPES]
            ignored.update(a for a in wanted if a and a not in OBJECTION_TYPES)
            code = str(item.get("code") or "").strip().upper()
            play = existing.get(code) or by_name.get(name.lower())
            if play is None:
                code = f"PLAY-{next_number:02d}"
                next_number += 1
                play = Play(code=code, name=name, description=description, category=category, addresses=addresses)
                session.add(play)
                session.add(PlayStats(play_code=code))
                existing[code], by_name[name.lower()] = play, play
                added.append(code)
            else:
                play.name, play.description, play.category, play.addresses = name, description, category, addresses
                updated.append(play.code)
        session.commit()
    return {"added": added, "updated": updated, "ignored_objections": sorted(ignored), "plays": list_plays(ctx)}


def list_plays(ctx: V1Context) -> list[dict]:
    with ctx.db.session() as session:
        return [
            {"code": p.code, "name": p.name, "description": p.description, "category": p.category, "addresses": p.addresses}
            for p in session.scalars(select(Play).order_by(Play.code)).all()
        ]

