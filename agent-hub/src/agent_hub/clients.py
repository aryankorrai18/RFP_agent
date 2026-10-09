"""Typed HTTP clients for the two live agents (Deal Intelligence, RFP Memory Assistant).

Only the routes the chat needs are implemented. There is deliberately no way to create, activate or
delete a workspace, or to choose a model: those change state that belongs to the agents' own screens,
so the hub only reads them."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import httpx

# The workspace each agent should use for the work in progress ({"deals": id, "rfp": id}). The engine sets it
# around a conversation's work; the clients send it as X-Workspace, so the hub never switches the workspace
# the agents' own screens show. Unset means "the agent's active workspace".
WORKSPACES: ContextVar[dict[str, str]] = ContextVar("agent_hub_workspaces", default={})
# The workspaces each agent may be used with for the work in progress ({"deals": [ids], "rfp": [ids]}), or None when
# sign-in is off. Set from the signed-in person's company, never from anything they type. When set, a request that does
# not name one of these workspaces is refused here, before it reaches an agent, whatever the flow above asked for.
TENANT: ContextVar[dict[str, list[str]] | None] = ContextVar("agent_hub_tenant", default=None)
# Whether the person may be sent to an agent's own screen. Those screens have no sign-in and list every workspace of every
# company, so a client account never gets a link to them; an administrator (the person running the hub) does.
OPEN_LINKS: ContextVar[bool] = ContextVar("agent_hub_open_links", default=True)

@contextmanager
def workspace_override(agent_id: str, workspace_id: str | None) -> Iterator[None]:
    """Look at another workspace of one agent for a moment (read only), then go back."""
    chosen = {k: v for k, v in WORKSPACES.get().items() if k != agent_id}
    if workspace_id:
        chosen[agent_id] = workspace_id
    token = WORKSPACES.set(chosen)
    try:
        yield
    finally:
        WORKSPACES.reset(token)


DEFAULT_TIMEOUT = 30.0
SLOW_TIMEOUT = 120.0

# Plain-language rewrites of upstream error codes, with what to do about them.
FRIENDLY: dict[str, tuple[str, str | None]] = {
    "no_company_facts": (
        "This workspace has no company facts yet, so there is nothing to ground the answers in.",
        "Add your company facts in the RFP assistant first.",
    ),
    "no_interactions": (
        "This deal has no emails or notes yet, so there is nothing to read.",
        "Attach the emails or notes to the deal first.",
    ),
    "signals_missing": (
        "The deal has to be read before a brief can be written.",
        "Say \"brief me on it\" again and I will read it first.",
    ),
    "not_final": (
        "Some answers are not final yet, so the response can't be exported.",
        "Accept, edit or rewrite every answer first (SME answers need a person).",
    ),
    "file_too_large": ("That file is too large for the app.", "Try a smaller file."),
    "no_library": (
        "This workspace has no company facts and no approved answers to answer from yet.",
        "Add the company facts or import past proposals in the RFP assistant first.",
    ),
    "no_deals": ("This workspace has no deals yet, so there is nothing to look across.", "Create or import deals first."),
}


class UpstreamError(Exception):
    """An agent answered with an error. `message` is safe to show in the chat as is."""

    def __init__(self, code: str, message: str, action: str | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.action, self.status = code, message, action, status

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    @classmethod
    def from_response(cls, response: httpx.Response) -> UpstreamError:
        code, message, action = "http_error", f"The app answered with an error (HTTP {response.status_code}).", None
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            envelope = body.get("error")
            if isinstance(envelope, dict):
                code = str(envelope.get("code") or code)
                message = str(envelope.get("message") or message)
                info = envelope.get("error_info")
                if isinstance(info, dict):
                    action = info.get("action")
            elif isinstance(body.get("detail"), str):
                message = body["detail"]
        if code in FRIENDLY:
            message, action = FRIENDLY[code][0], action or FRIENDLY[code][1]
        return cls(code, message, action, response.status_code)


class AgentDown(Exception):
    """The agent could not be reached."""

    def __init__(self, agent_name: str, start_hint: str | None = None) -> None:
        super().__init__(f"{agent_name} is not reachable")
        self.agent_name, self.start_hint = agent_name, start_hint

    @property
    def message(self) -> str:
        hint = f" Start it with {self.start_hint}, then ask me again." if self.start_hint else ""
        return f"{self.agent_name} isn't running right now.{hint}"


def describe_error_info(info: dict | None, fallback: str | None = None, agent_name: str | None = None) -> str:
    """Turn a job's `error_info` ({title, detail, action, switch_model, ...}) into chat text."""
    if not isinstance(info, dict):
        return fallback or "Something went wrong."
    parts = [str(info.get("title") or "").strip().rstrip(".")]
    if info.get("detail"):
        parts.append(str(info["detail"]).strip())
    text = ". ".join(p for p in parts if p)
    if info.get("action"):
        text += f" What to do: {str(info['action']).strip()}"
    if info.get("switch_model") and agent_name:
        text += f" (Models are changed in {agent_name} itself; the hub never changes them.)"
    return text or (fallback or "Something went wrong.")


@dataclass(frozen=True)
class WorkspaceInfo:
    id: str | None
    name: str | None
    kind: str | None = None
    company_set_up: bool | None = None  # RFP only


class _AgentClient:
    def __init__(
        self, agent: dict, transport: httpx.AsyncBaseTransport | None = None, timeout: float = DEFAULT_TIMEOUT,
        slow_timeout: float = SLOW_TIMEOUT,
    ) -> None:
        self.agent = agent
        self.name: str = agent["name"]
        self.start_hint: str | None = agent.get("start_hint")
        self.open_url: str = agent["url"] + agent.get("open_path", "/")
        self.base_url: str = agent["url"]
        self.slow_timeout = slow_timeout
        self._http = httpx.AsyncClient(base_url=agent["url"], transport=transport, timeout=timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    @property
    def workspace_id(self) -> str | None:
        """The workspace this conversation chose for this agent, if any."""
        return WORKSPACES.get().get(self.agent["id"])

    async def workspaces(self) -> tuple[str | None, list[dict[str, Any]]]:
        """(the agent's active workspace id, every workspace it has). Read only."""
        data = await self._json("GET", "/v1/workspaces")
        active, spaces = data.get("active"), data.get("workspaces", [])
        allowed = TENANT.get()
        if allowed is not None:  # a company only ever sees (and so can only ever choose) its own workspaces
            ok = set(allowed.get(self.agent["id"], []))
            spaces = [s for s in spaces if s.get("id") in ok]
            active = active if active in ok else None
        return active, spaces

    async def _request(self, method: str, path: str, *, slow: bool = False, **kwargs: Any) -> httpx.Response:
        if slow:
            kwargs["timeout"] = self.slow_timeout
        workspace = self.workspace_id
        allowed = TENANT.get()
        if allowed is not None and not (method == "GET" and path in ("/v1/workspaces", self.agent.get("health_path", "/health"))):
            if not workspace:
                raise UpstreamError("no_workspace", f"No {self.name} workspace is set up for your company.",
                                    "Ask your administrator to give your company one (administrators: open /admin, "
                                    "Companies, Edit, and tick one).")
            if workspace not in allowed.get(self.agent["id"], []):
                raise UpstreamError("forbidden_workspace", "That workspace isn't available to your company.")
        headers = dict(kwargs.get("headers") or {})
        token = os.environ.get(self.agent.get("token_env", ""), "") if self.agent.get("token_env") else ""
        if token:  # the agent only accepts callers that hold its service token (set when the agents are not on a private machine)
            headers["X-Service-Token"] = token
        if workspace:
            headers["X-Workspace"] = workspace
        if headers:
            kwargs["headers"] = headers
        try:
            response = await self._http.request(method, path, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.NetworkError) as exc:
            raise AgentDown(self.name, self.start_hint) from exc
        except httpx.TimeoutException as exc:
            raise UpstreamError("timeout", f"{self.name} took too long to answer.") from exc
        except httpx.HTTPError as exc:
            raise AgentDown(self.name, self.start_hint) from exc
        if response.status_code >= 400:
            raise UpstreamError.from_response(response)
        return response

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        return (await self._request(method, path, **kwargs)).json()

    async def health(self) -> bool:
        await self._request("GET", self.agent.get("health_path", "/health"))
        return True

    async def get_job(self, job_id: int | str) -> dict[str, Any]:
        return await self._json("GET", f"/v1/jobs/{job_id}")


class DealClient(_AgentClient):
    async def status(self) -> WorkspaceInfo:
        """The workspace in use (read only): the one this conversation chose, else the agent's active one."""
        active, spaces = await self.workspaces()
        wanted = self.workspace_id or active
        for space in spaces:
            if space.get("id") == wanted:
                return WorkspaceInfo(space.get("id"), space.get("name"), space.get("kind"))
        if self.workspace_id:
            raise UpstreamError("not_found", f"The workspace {wanted!r} no longer exists in {self.name}.",
                                "Say \"use the default workspace\" or pick another one.", 404)
        return WorkspaceInfo(active, None)

    async def list_deals(self) -> list[dict[str, Any]]:
        return (await self._json("GET", "/v1/deals")).get("deals", [])

    async def get_deal(self, deal_id: int) -> dict[str, Any]:
        return await self._json("GET", f"/v1/deals/{deal_id}")

    async def create_deal(
        self, name: str, account: str, files: list[tuple[str, bytes]] | None = None, **fields: Any
    ) -> dict[str, Any]:
        data = {"name": name, "account": account, **{k: str(v) for k, v in fields.items() if v not in (None, "")}}
        multipart = [("files", (fname, content)) for fname, content in files or []]
        return await self._json("POST", "/v1/deals", data=data, files=multipart or None, slow=True)

    async def add_files(self, deal_id: int, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        multipart = [("files", (fname, content)) for fname, content in files]
        return await self._json("POST", f"/v1/deals/{deal_id}/files", files=multipart, slow=True)

    async def add_note(self, deal_id: int, text: str, kind: str = "call_note", **fields: Any) -> dict[str, Any]:
        return await self._json("POST", f"/v1/deals/{deal_id}/notes", json={"kind": kind, "text": text, **fields})

    async def start_signals(self, deal_id: int) -> int:
        """One model call. The caller must have the person's go-ahead."""
        return (await self._json("POST", f"/v1/deals/{deal_id}/signals"))["job_id"]

    async def start_brief(self, deal_id: int, mode: str | None = None) -> int:
        """One model call. Needs signals ready."""
        return (await self._json("POST", f"/v1/deals/{deal_id}/brief", json={"mode": mode} if mode else {}))["job_id"]

    async def get_brief(self, deal_id: int, mode: str | None = None) -> dict[str, Any] | None:
        try:
            return await self._json("GET", f"/v1/deals/{deal_id}/brief", params={"mode": mode} if mode else None)
        except UpstreamError as exc:
            if exc.code == "no_brief":
                return None
            raise

    async def ask(self, deal_id: int, question: str) -> dict[str, Any]:
        """One model call. The caller must have the person's go-ahead."""
        return await self._json("POST", f"/v1/deals/{deal_id}/ask", json={"question": question}, slow=True)

    async def ask_portfolio(self, question: str) -> dict[str, Any]:
        """One model call over the whole pipeline. The caller must have the person's go-ahead."""
        return await self._json("POST", "/v1/portfolio/ask", json={"question": question}, slow=True)

    async def followup(self, deal_id: int, kind: str = "email", play_code: str | None = None) -> dict[str, Any]:
        """One model call. Drafts from the latest brief's next step; the caller must have the person's go-ahead."""
        body: dict[str, Any] = {"kind": kind}
        if play_code:
            body["play_code"] = play_code
        return await self._json("POST", f"/v1/deals/{deal_id}/followup", json=body, slow=True)

    async def update_deal(self, deal_id: int, **fields: Any) -> dict[str, Any]:
        """Free: change a deal's own details (industry, segment, name, account). Fields left out stay as they are."""
        return await self._json("PATCH", f"/v1/deals/{deal_id}", json={k: v for k, v in fields.items() if v not in (None, "")})

    async def record_outcome(
        self, deal_id: int, result: str, loss_reason: str | None, plays_used: list[str], closed_on: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"result": result, "loss_reason": loss_reason, "plays_used": plays_used}
        if closed_on:
            body["closed_on"] = closed_on
        return await self._json("PUT", f"/v1/deals/{deal_id}/outcome", json=body)


    async def list_plays(self) -> list[dict[str, Any]]:
        return (await self._json("GET", "/v1/plays")).get("plays", [])

    async def save_plays(self, plays: list[dict[str, Any]]) -> dict[str, Any]:
        """Adds or updates the company's own plays (no model call). Only after the person confirmed it in the chat."""
        return await self._json("POST", "/v1/plays", json={"plays": plays})


class RfpClient(_AgentClient):
    async def workspace_info(self) -> WorkspaceInfo:
        data = await self._json("GET", "/v1/workspace")
        return WorkspaceInfo(data.get("id"), data.get("name"), data.get("kind"), bool(data.get("company_set_up")))

    async def ask_library(self, question: str) -> dict[str, Any]:
        """One model call over the company facts and approved answers. The caller must have the person's go-ahead."""
        return await self._json("POST", "/v1/library/ask", json={"question": question}, slow=True)

    async def create_project(
        self, filename: str, data: bytes, name: str | None = None, client: str | None = None,
        industry: str | None = None,
    ) -> dict[str, Any]:
        """One model call (extraction). Not idempotent: a second call makes a second project."""
        form = {k: v for k, v in {"name": name, "client": client, "industry": industry}.items() if v}
        return await self._json("POST", "/v1/projects", data=form, files={"file": (filename, data)}, slow=True)

    async def list_projects(self) -> list[dict[str, Any]]:
        found = await self._json("GET", "/v1/projects")
        return found if isinstance(found, list) else found.get("projects", [])

    async def get_project(self, project_id: int) -> dict[str, Any]:
        return await self._json("GET", f"/v1/projects/{project_id}")

    async def start_draft(self, project_id: int, requirement_ids: list[int] | None = None) -> dict[str, Any]:
        """One model call per requirement. The caller must have the person's go-ahead."""
        body = {"requirement_ids": requirement_ids} if requirement_ids is not None else {}
        return await self._json("POST", f"/v1/projects/{project_id}/draft", json=body)

    async def review(self, requirement_id: int, action: str, final_text: str | None = None,
                     reason_tags: list[str] | None = None) -> dict[str, Any]:
        """One review decision through the RFP assistant's own review route; it scores and learns from it."""
        body: dict[str, Any] = {"action": action}
        if final_text is not None:
            body["final_text"] = final_text
        if reason_tags:
            body["reason_tags"] = reason_tags
        return await self._json("POST", f"/v1/requirements/{requirement_id}/review", json=body)

    async def export(self, project_id: int, fmt: str) -> tuple[bytes, str, str]:
        """(content, media type, Content-Disposition) of the finished response."""
        response = await self._request("GET", f"/v1/projects/{project_id}/export", params={"format": fmt}, slow=True)
        return (
            response.content,
            response.headers.get("content-type", "application/octet-stream"),
            response.headers.get("content-disposition", f'attachment; filename="response.{fmt}"'),
        )

    async def record_outcome(self, project_id: int, result: str, loss_reason: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"result": result}
        if loss_reason:
            body["loss_reason"] = loss_reason
        return await self._json("PUT", f"/v1/projects/{project_id}/outcome", json=body)

    # -- feeding the memory (each only after the person confirmed it in the chat) --------------------------

    async def get_company(self) -> dict[str, Any]:
        """The workspace's company name and official facts, and whether it is the read-only sample."""
        return await self._json("GET", "/v1/company")

    async def save_company(self, company: str, facts: list[dict[str, Any]]) -> dict[str, Any]:
        """Replaces the whole fact sheet, so callers send the existing facts plus the new ones."""
        return await self._json("PUT", "/v1/company", json={"company": company, "facts": facts})

    async def import_proposal(self, filename: str, data: bytes, *, client: str | None = None, industry: str | None = None,
                              submitted_on: str | None = None, result: str | None = None, loss_reason: str | None = None) -> dict[str, Any]:
        """One model call (reading the pairs out). Not idempotent: the same file twice is refused by the agent."""
        form = {k: v for k, v in {"client": client, "industry": industry, "submitted_on": submitted_on,
                                  "result": result or "unknown", "loss_reason": loss_reason}.items() if v}
        return await self._json("POST", "/v1/library", data=form, files={"file": (filename, data)}, slow=True)

    async def get_proposal(self, proposal_id: int) -> dict[str, Any]:
        return await self._json("GET", f"/v1/library/proposals/{proposal_id}")

    async def keep_pairs(self, proposal_id: int, pair_ids: list[int]) -> dict[str, Any]:
        return await self._json("POST", f"/v1/library/proposals/{proposal_id}/confirm",
                                json={"pairs": [{"id": i, "decision": "kept"} for i in pair_ids]})

    async def discard_proposal(self, proposal_id: int) -> dict[str, Any]:
        return await self._json("POST", f"/v1/library/proposals/{proposal_id}/discard")
