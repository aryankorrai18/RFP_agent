"""Creating and removing a company's workspaces in the agents, for the admin page only.

The chat clients (clients.py) deliberately have no way to create, switch or delete a workspace, and nothing in the chat path
imports this module: it is reached only from an administrator's click on /admin, and every use is written to the activity log.

Two facts about the agents shape this code. Creating a workspace through an agent's API also *switches that agent to it*, and
the agent's own screens show whatever is active, so the previously active workspace is restored straight afterwards. And an
agent refuses to delete "main" or its active workspace, so a workspace in either state is kept and reported, never forced."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .clients import AgentDown, UpstreamError, _AgentClient

AGENT_NAMES = {"deals": "Deal Intelligence", "rfp": "RFP Memory Assistant"}
WORKSPACE_LABEL = {"deals": "My deals", "rfp": "RFPs"}


class ProvisionError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def workspace_name(company: str, agent_id: str) -> str:
    return f"{company.strip()} - {WORKSPACE_LABEL[agent_id]}"


async def create_workspace(client: _AgentClient, agent_id: str, company: str) -> str:
    """One new, empty, company-type workspace in the agent. Returns its id and leaves the agent's active workspace as it was."""
    try:
        before = (await client._json("GET", "/v1/workspaces")).get("active")
        created = await client._json("POST", "/v1/workspaces", json={"name": workspace_name(company, agent_id), "kind": "company"})
    except (AgentDown, UpstreamError) as exc:
        raise ProvisionError(f"{AGENT_NAMES[agent_id]} could not create the workspace: {_why(exc)}") from exc
    new_id = created["workspace"]["id"]
    if before and before != new_id:  # creating switched the agent to the new workspace; put its own screens back
        try:
            await client._json("POST", f"/v1/workspaces/{before}/activate")
        except (AgentDown, UpstreamError):
            pass  # not fatal: the workspace exists; the person can switch back in the agent's own screen
    return new_id


async def delete_workspace(client: _AgentClient, workspace_id: str) -> None:
    await client._json("DELETE", f"/v1/workspaces/{workspace_id}")


def _why(exc: Exception) -> str:
    if isinstance(exc, AgentDown):
        return "it is not running"
    return getattr(exc, "message", None) or str(exc)


@dataclass
class PlanItem:
    agent: str
    id: str
    name: str
    kind: str
    deletable: bool
    reason: str | None = None  # why it will be kept, when it is not deletable

    @property
    def key(self) -> str:
        return f"{self.agent}:{self.id}"

    def public(self) -> dict[str, Any]:
        return {"agent": self.agent, "agent_name": AGENT_NAMES[self.agent], "id": self.id, "name": self.name, "kind": self.kind,
                "deletable": self.deletable, "reason": self.reason, "key": self.key}


async def deletion_plan(clients: dict[str, _AgentClient], company: dict[str, Any], others: list[dict[str, Any]]) -> list[PlanItem]:
    """What would happen to each workspace of a company if its data were deleted too: which can go, and why any cannot."""
    items: list[PlanItem] = []
    for agent_id, client in clients.items():
        ids = list(company["workspaces"].get(agent_id, []))
        if not ids:
            continue
        try:
            data = await client._json("GET", "/v1/workspaces")
            known = {w["id"]: w for w in data.get("workspaces", [])}
            active = data.get("active")
            up = True
        except (AgentDown, UpstreamError):
            known, active, up = {}, None, False
        for ws_id in ids:
            shared = [o["name"] for o in others if ws_id in o["workspaces"].get(agent_id, [])]
            space = known.get(ws_id)
            name, kind = (space.get("name") or ws_id, space.get("kind", "")) if space else (ws_id, "")
            reasons: list[str] = []
            if not up:
                reasons.append(f"{AGENT_NAMES[agent_id]} is not running")
            elif space is None:
                reasons.append("it no longer exists there")
            elif kind == "main":
                reasons.append("it is the agent's original workspace, which can never be removed")
            if shared:
                reasons.append("another company also uses it (" + ", ".join(shared) + ")")
            if up and space is not None and ws_id == active:
                reasons.append(f"it is the one {AGENT_NAMES[agent_id]} currently has open; switch to another workspace there first")
            reason = "; ".join(reasons) or None
            items.append(PlanItem(agent_id, ws_id, name, kind, reason is None, reason))
    return items
