"""What each company has used, for the administrator: people, chats, model calls and tokens, and storage.

Everything here is read from records, never produced by a model. The hub's own model calls (reading a message) come from its
own ledger; each agent reports what a workspace has used through GET /v1/usage, and a company's numbers are the sum over the
workspaces it was given. Workspaces no company has been given (the agents' own `main`, for instance) are shown separately, so
the totals still add up. Tokens count from when each ledger was added; earlier calls were not recorded."""

from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .auth import OPERATOR_ID, OPERATOR_NAME
from .clients import TENANT, WORKSPACES, AgentDown, UpstreamError

AGENTS = (("deals", "Deal Intelligence"), ("rfp", "RFP Memory Assistant"))
ZERO = {"calls": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "storage_bytes": 0}


@contextmanager
def unrestricted():  # noqa: ANN201
    """The administrator looks across every company, so the signed-in person's own company limits are lifted for this look."""
    tenant, spaces = TENANT.set(None), WORKSPACES.set({})
    try:
        yield
    finally:
        TENANT.reset(tenant)
        WORKSPACES.reset(spaces)


async def _fetch(client: Any, agent_id: str, workspace: str) -> dict[str, Any] | None:
    WORKSPACES.set({agent_id: workspace})  # inside its own task, so it applies to this call only
    try:
        return await client._json("GET", "/v1/usage")
    except (AgentDown, UpstreamError):
        return None


def _folder_bytes(path: Path) -> int:
    try:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0
    except OSError:
        return 0


def _sum(parts: list[dict[str, Any] | None]) -> dict[str, Any]:
    total, missing = dict(ZERO), 0
    for part in parts:
        if part is None:
            missing += 1
            continue
        model, disk = part["model"], part["storage"]
        total["calls"] += model["calls"]
        total["failed"] += model["failed"]
        total["input_tokens"] += model["input_tokens"]
        total["output_tokens"] += model["output_tokens"]
        total["storage_bytes"] += disk["total_bytes"]
    return {**total, "workspaces": len(parts), "unavailable": missing}


async def build_report(engine: Any) -> dict[str, Any]:
    auth, store = engine.auth, engine.store
    companies = auth.list_companies() if auth is not None else []
    everyone = auth.list_users() if auth is not None else []
    people = Counter(u["company_id"] for u in everyone if u["company_id"] != OPERATOR_ID)  # customers only
    admins = sum(1 for u in everyone if u["company_id"] == OPERATOR_ID)

    def by_company(sql: str) -> dict[str | None, Any]:
        return {r["company_id"]: r["n"] for r in store.sql_all(sql)} if store.exists() else {}

    chats = by_company("SELECT u.company_id AS company_id, COUNT(*) AS n FROM conversations c JOIN users u ON u.id = c.user_id GROUP BY u.company_id")
    messages = by_company("SELECT u.company_id AS company_id, COUNT(*) AS n FROM events e JOIN conversations c ON c.id = e.conversation_id "
                          "JOIN users u ON u.id = c.user_id WHERE e.role = 'user' GROUP BY u.company_id")
    uploads = by_company("SELECT u.company_id AS company_id, COALESCE(SUM(up.size), 0) AS n FROM uploads up JOIN conversations c ON c.id = up.conversation_id "
                         "JOIN users u ON u.id = c.user_id GROUP BY u.company_id")
    hub_calls = {r["company_id"]: r for r in store.sql_all(
        "SELECT company_id, COUNT(*) AS calls, COALESCE(SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END), 0) AS failed, "
        "COALESCE(SUM(input_tokens), 0) AS i, COALESCE(SUM(output_tokens), 0) AS o, MIN(at) AS first FROM model_usage GROUP BY company_id")} if store.exists() else {}

    listing: dict[str, list[dict] | None] = {}
    with unrestricted():
        for agent_id, _name in AGENTS:
            client = engine.deal if agent_id == "deals" else engine.rfp
            try:
                listing[agent_id] = (await client._json("GET", "/v1/workspaces")).get("workspaces", [])
            except (AgentDown, UpstreamError):
                listing[agent_id] = None
        jobs = {(a, w["id"]): _fetch(engine.deal if a == "deals" else engine.rfp, a, w["id"])
                for a, spaces in listing.items() for w in (spaces or [])}
        fetched = dict(zip(jobs, await asyncio.gather(*jobs.values())))

    def agent_totals(agent_id: str, ids: list[str]) -> dict[str, Any]:
        if listing[agent_id] is None:
            return {**ZERO, "workspaces": len(ids), "unavailable": len(ids), "up": False}
        return {**_sum([fetched.get((agent_id, i)) for i in ids]), "up": True}

    since = [r["first"] for r in hub_calls.values() if r["first"]]
    for part in fetched.values():
        if part and part["model"].get("since"):
            since.append(part["model"]["since"])

    rows, given = [], {a: set() for a, _ in AGENTS}
    for c in companies:
        agents = {a: agent_totals(a, list(c["workspaces"].get(a, []))) for a, _ in AGENTS}
        for a, _ in AGENTS:
            given[a].update(c["workspaces"].get(a, []))
        h = hub_calls.get(c["id"])
        hub = {"calls": h["calls"] if h else 0, "failed": h["failed"] if h else 0, "input_tokens": h["i"] if h else 0, "output_tokens": h["o"] if h else 0}
        rows.append({"id": c["id"], "name": c["name"], "people": people.get(c["id"], 0), "chats": chats.get(c["id"], 0),
                     "messages": messages.get(c["id"], 0), "hub": hub, "agents": agents,
                     "calls": hub["calls"] + sum(v["calls"] for v in agents.values()),
                     "failed": hub["failed"] + sum(v["failed"] for v in agents.values()),
                     "input_tokens": hub["input_tokens"] + sum(v["input_tokens"] for v in agents.values()),
                     "output_tokens": hub["output_tokens"] + sum(v["output_tokens"] for v in agents.values()),
                     "storage_bytes": uploads.get(c["id"], 0) + sum(v["storage_bytes"] for v in agents.values())})

    spare = {a: [w for w in (listing[a] or []) if w["id"] not in given[a]] for a, _ in AGENTS}
    unassigned = {"agents": {a: agent_totals(a, [w["id"] for w in spare[a]]) for a, _ in AGENTS},
                  "workspaces": [{"agent": n, "id": w["id"], "name": w.get("name") or w["id"]} for (a, n) in AGENTS for w in spare[a]]}
    for key in ("calls", "failed", "input_tokens", "output_tokens", "storage_bytes"):
        unassigned[key] = sum(v[key] for v in unassigned["agents"].values())
    op = hub_calls.get(OPERATOR_ID)  # the administrators' own use of the hub: the control plane, not a customer
    operator = {"name": OPERATOR_NAME, "people": admins, "calls": op["calls"] if op else 0, "failed": op["failed"] if op else 0,
                "input_tokens": op["i"] if op else 0, "output_tokens": op["o"] if op else 0}
    nobody = hub_calls.get(None)  # chats made with sign-in off, or by a company that no longer exists
    unattributed = {"calls": nobody["calls"] if nobody else 0, "failed": nobody["failed"] if nobody else 0,
                    "input_tokens": nobody["i"] if nobody else 0, "output_tokens": nobody["o"] if nobody else 0}
    hub_disk = (store.db_path.stat().st_size if store.exists() and store.db_path.exists() else 0) + _folder_bytes(store.data_dir / "uploads")

    totals = {"companies": len(rows), "people": sum(people.values()), "chats": sum(r["chats"] for r in rows), "messages": sum(r["messages"] for r in rows),
              "calls": sum(r["calls"] for r in rows) + unassigned["calls"] + unattributed["calls"] + operator["calls"],
              "failed": sum(r["failed"] for r in rows) + unassigned["failed"] + unattributed["failed"] + operator["failed"],
              "input_tokens": sum(r["input_tokens"] for r in rows) + unassigned["input_tokens"] + unattributed["input_tokens"] + operator["input_tokens"],
              "output_tokens": sum(r["output_tokens"] for r in rows) + unassigned["output_tokens"] + unattributed["output_tokens"] + operator["output_tokens"],
              "storage_bytes": sum(r["storage_bytes"] for r in rows) + unassigned["storage_bytes"] + hub_disk - sum(uploads.values())}
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tracking_since": min(since) if since else None,
            "totals": totals, "companies": rows, "unassigned": unassigned, "unattributed_hub_calls": unattributed, "operator": operator,
            "hub_storage_bytes": hub_disk, "agents_down": [name for a, name in AGENTS if listing[a] is None]}


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"
