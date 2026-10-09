"""Seed a demo workspace with a fictional company's deal history, with no model calls.

Halcyon Software (fictional) sells a revenue-analytics platform. The seed holds 14 closed deals (7 won,
7 lost) and 4 open ones, each with stakeholders, interactions and the structured signals a model would
otherwise extract (seed/halcyon.json). Nothing is extracted, and Hindsight is not called here: the rows are
inserted as outbox entries ("pending") and the normal sync retains them afterwards.

Open deals describe time relatively (`days_ago`, `due_in_days`) so the demo never looks stale.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from sqlalchemy import select

from .db import (
    INTERACTION_KINDS, LOSS_REASONS, OBJECTION_STATUSES, OBJECTION_TYPES, PLAY_CATEGORIES, SEGMENTS, STAGES, STANCES,
    Deal, DealSignals, Interaction, MemoryEvent, Play, PlayStats, Stakeholder, deal_code, interaction_code, utcnow,
)

if TYPE_CHECKING:
    from .context import V1Context

SEED_PATH = Path(__file__).resolve().parents[2] / "seed" / "halcyon.json"

PROMISE_STATUSES = ("open", "kept", "missed")
DEAL_ROLES = ("demo", "closable", "similar_open", "sparse")


def _fail(where: str, message: str) -> NoReturn:
    raise ValueError(f"Demo seed: {where}: {message}")


def _require(where: str, item: Any, field: str, kind: type | tuple[type, ...]) -> Any:
    if not isinstance(item, dict) or field not in item:
        _fail(where, f"missing '{field}'")
    value = item[field]
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        _fail(where, f"'{field}' has the wrong type")
    return value


def _one_of(where: str, field: str, value: Any, allowed: tuple[str, ...] | frozenset[str]) -> None:
    if value not in allowed:
        _fail(where, f"'{field}' is {value!r}; expected one of {sorted(allowed)}")


def _iso(where: str, field: str, value: Any) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        _fail(where, f"'{field}' is not an ISO date: {value!r}")


def load_seed(path: Path | None = None) -> dict:
    """Parse and validate the seed file. Raises ValueError with a clear message when anything is off."""
    path = path or SEED_PATH
    try:
        seed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Demo seed: cannot read {path.name}: {exc}") from exc
    for field in ("vendor", "disclosure"):
        _require("top level", seed, field, str)
    plays = _require("top level", seed, "plays", list)
    deals = _require("top level", seed, "deals", list)

    play_codes: set[str] = set()
    for play in plays:
        code = _require("play", play, "code", str)
        where = f"play {code}"
        if code in play_codes:
            _fail(where, "duplicate play code")
        play_codes.add(code)
        for field in ("name", "description"):
            _require(where, play, field, str)
        _one_of(where, "category", _require(where, play, "category", str), PLAY_CATEGORIES)
        for objection_type in _require(where, play, "addresses", list):
            _one_of(where, "addresses", objection_type, OBJECTION_TYPES)

    interaction_keys: set[str] = set()
    seen_deals: set[tuple[str, str]] = set()
    roles: dict[str, int] = {}
    for deal in deals:
        account = _require("deal", deal, "account", str)
        name = _require("deal", deal, "name", str)
        where = f"deal '{name}'"
        if (account, name) in seen_deals:
            _fail(where, "duplicate (account, name)")
        seen_deals.add((account, name))
        _require(where, deal, "industry", str)
        _one_of(where, "segment", _require(where, deal, "segment", str), SEGMENTS)
        _require(where, deal, "amount", int)
        _require(where, deal, "owner", str)
        result = _require(where, deal, "result", str)
        _one_of(where, "result", result, ("won", "lost", "open"))
        if "role" in deal:
            _one_of(where, "role", deal["role"], DEAL_ROLES)
            roles[deal["role"]] = roles.get(deal["role"], 0) + 1
        relative = result == "open"
        if relative:
            _require(where, deal, "opened_days_ago", int)
            _one_of(where, "stage", _require(where, deal, "stage", str), tuple(s for s in STAGES if s != "closed"))
            if deal.get("loss_reason") is not None:
                _fail(where, "an open deal has no loss_reason")
        else:
            opened = _iso(where, "opened_on", _require(where, deal, "opened_on", str))
            closed = _iso(where, "closed_on", _require(where, deal, "closed_on", str))
            if closed < opened:
                _fail(where, "closed_on is before opened_on")
            if result == "lost":
                _one_of(where, "loss_reason", deal.get("loss_reason"), LOSS_REASONS)
            elif deal.get("loss_reason") is not None:
                _fail(where, "a won deal has no loss_reason")

        for who in _require(where, deal, "stakeholders", list):
            _require(where, who, "name", str)
            _one_of(where, "stance", _require(where, who, "stance", str), STANCES)
            _require(where, who, "engaged", bool)
            _require(where, who, "economic_buyer", bool)

        deal_keys: set[str] = set()
        interactions = _require(where, deal, "interactions", list)
        if not interactions:
            _fail(where, "needs at least one interaction")
        for item in interactions:
            key = _require(where, item, "key", str)
            if key in interaction_keys:
                _fail(where, f"interaction key {key!r} is not unique in the file")
            interaction_keys.add(key)
            deal_keys.add(key)
            _one_of(f"{where} / {key}", "kind", _require(where, item, "kind", str), INTERACTION_KINDS)
            _require(f"{where} / {key}", item, "text", str)
            if relative:
                _require(f"{where} / {key}", item, "days_ago", int)
            else:
                _iso(f"{where} / {key}", "date", _require(f"{where} / {key}", item, "date", str))

        signals = deal.get("signals")
        if signals is None:
            if not relative:
                _fail(where, "only an open deal may have signals: null")
            continue
        if not isinstance(signals, dict):
            _fail(where, "signals must be an object or null")

        def check_evidence(label: str, evidence: Any) -> None:
            if not isinstance(evidence, list):
                _fail(where, f"{label}: evidence must be a list")
            for key in evidence:
                if key not in interaction_keys or key not in deal_keys:
                    _fail(where, f"{label}: evidence key {key!r} is not an interaction of this deal")

        for objection in _require(where, signals, "objections", list):
            _one_of(where, "objection type", _require(where, objection, "type", str), OBJECTION_TYPES)
            _require(where, objection, "text", str)
            _one_of(where, "objection status", _require(where, objection, "status", str), OBJECTION_STATUSES)
            if relative:
                _require(where, objection, "first_seen_days_ago", int)
            else:
                _iso(where, "first_seen_on", objection.get("first_seen_on"))
            check_evidence(f"objection {objection['type']}", _require(where, objection, "evidence", list))
        _require(where, signals, "competitors", list)
        _require(where, _require(where, signals, "pricing", dict), "discount_requested", bool)
        for promise in _require(where, signals, "promises", list):
            _require(where, promise, "text", str)
            _require(where, promise, "owner", str)
            _one_of(where, "promise status", _require(where, promise, "status", str), PROMISE_STATUSES)
            if relative:
                _require(where, promise, "due_in_days", int)
            else:
                _iso(where, "due_on", promise.get("due_on"))
            check_evidence("promise", _require(where, promise, "evidence", list))
        for code in _require(where, signals, "plays_used", list):
            if code not in play_codes:
                _fail(where, f"plays_used references unknown play {code!r}")

    for role in ("demo", "closable"):
        if roles.get(role) != 1:
            raise ValueError(f"Demo seed: exactly one deal must have role {role!r}, found {roles.get(role, 0)}")
    return seed


def seed_demo(ctx: V1Context, today: date | None = None) -> dict[str, Any]:
    """Insert the plays and deals. Makes no model call and does not touch Hindsight; the outbox sync does."""
    seed = load_seed()
    today = today or date.today()
    plays_added = deals_added = closed_added = open_added = interactions_added = already_there = 0
    codes: dict[str, str | None] = {"demo": None, "closable": None}
    open_accounts = {d["account"] for d in seed["deals"] if d["result"] == "open"}

    with ctx.db.session() as session:
        for play in seed["plays"]:
            if session.get(Play, play["code"]) is None:
                session.add(Play(code=play["code"], name=play["name"], description=play["description"],
                                 category=play["category"], addresses=list(play["addresses"])))
                session.add(PlayStats(play_code=play["code"]))
                plays_added += 1
        session.flush()

        for item in seed["deals"]:
            existing = session.scalars(select(Deal).where(Deal.account == item["account"], Deal.name == item["name"])).first()
            if existing is not None:
                already_there += 1
                if item.get("role") in codes:
                    codes[item["role"]] = existing.code
                continue

            is_open = item["result"] == "open"
            opened_on = today - timedelta(days=item["opened_days_ago"]) if is_open else date.fromisoformat(item["opened_on"])
            deal = Deal(
                name=item["name"], account=item["account"], industry=item["industry"], segment=item["segment"],
                amount=item["amount"], stage=item["stage"] if is_open else "closed", owner=item["owner"],
                opened_on=opened_on, closed_on=None if is_open else date.fromisoformat(item["closed_on"]),
                result=item["result"], loss_reason=item.get("loss_reason"),
                signals_status="none" if item["signals"] is None else "ready",
                hindsight_status="skipped" if is_open else "pending",
            )
            session.add(deal)
            session.flush()
            for who in item["stakeholders"]:
                session.add(Stakeholder(
                    deal_id=deal.id, name=who["name"], title=who.get("title"), stance=who["stance"],
                    engaged=who["engaged"], economic_buyer=who["economic_buyer"], note=who.get("note"),
                ))

            # Closed-deal interactions are only retained when the account has an open deal (account history).
            interaction_status = "pending" if is_open or item["account"] in open_accounts else "skipped"
            code_by_key: dict[str, str] = {}
            for entry in item["interactions"]:
                row = Interaction(
                    deal_id=deal.id, kind=entry["kind"],
                    occurred_on=today - timedelta(days=entry["days_ago"]) if is_open else date.fromisoformat(entry["date"]),
                    author=entry.get("author"), subject=entry.get("subject"), text=entry["text"],
                    sha256=hashlib.sha256(entry["text"].encode("utf-8")).hexdigest(),
                    hindsight_status=interaction_status,
                )
                session.add(row)
                session.flush()
                code_by_key[entry["key"]] = interaction_code(row.id)
                interactions_added += 1

            if item["signals"] is not None:
                signals = item["signals"]
                session.add(DealSignals(
                    deal_id=deal.id,
                    objections=[{
                        "type": o["type"], "text": o["text"], "status": o["status"],
                        "first_seen_on": (today - timedelta(days=o["first_seen_days_ago"]) if is_open
                                          else date.fromisoformat(o["first_seen_on"])).isoformat(),
                        "evidence": [code_by_key[key] for key in o["evidence"]],
                    } for o in signals["objections"]],
                    competitors=list(signals["competitors"]),
                    pricing=dict(signals["pricing"]),
                    promises=[{
                        "text": p["text"], "owner": p["owner"],
                        "due_on": (today + timedelta(days=p["due_in_days"]) if is_open else date.fromisoformat(p["due_on"])).isoformat(),
                        "status": p["status"], "evidence": [code_by_key[key] for key in p["evidence"]],
                    } for p in signals["promises"]],
                    plays_used=list(signals["plays_used"]),
                    source="seed", updated_at=utcnow(),
                ))

            outcome = "open" if is_open else item["result"]
            session.add(MemoryEvent(kind="deal_added", deal_id=deal.id, detail=(
                f"Added {deal_code(deal.id)} {item['account']} ({outcome}, {item['segment']}, {item['industry']}) "
                f"with {len(item['interactions'])} interactions."
            )))
            deals_added += 1
            if is_open:
                open_added += 1
            else:
                closed_added += 1
            if item.get("role") in codes:
                codes[item["role"]] = deal.code
        session.commit()

    ctx.schedule_sync()
    return {
        "plays": plays_added, "deals": deals_added, "closed": closed_added, "open": open_added,
        "interactions": interactions_added, "already_there": already_there,
        "demo_deal": codes["demo"], "closable_deal": codes["closable"],
    }
