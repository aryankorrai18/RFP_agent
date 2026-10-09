"""The admin page's API: companies and accounts. Only an administrator may call it (the one exception is first-run setup).

Nothing here ever returns a password or a session token, and every change is written to the activity log with who made it
(never the password). The workspaces a company may use are checked against what the agents actually have, so a typo does
not quietly give a company nothing."""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import provisioning
from .auth import OPERATOR_ID, OPERATOR_NAME, AuthError, User
from .clients import AgentDown, UpstreamError
from .deps import get_auth, require_admin, set_session_cookie, setup_available

router = APIRouter(prefix="/api/admin")

AGENT_NAMES = {"deals": "Deal Intelligence", "rfp": "RFP Memory Assistant"}


class SetupIn(BaseModel):
    email: str = Field(max_length=200)
    name: str | None = Field(default=None, max_length=120)
    password: str = Field(max_length=500)


class CompanyIn(BaseModel):
    name: str = Field(max_length=120)
    deals: list[str] = Field(default_factory=list, max_length=50)
    rfp: list[str] = Field(default_factory=list, max_length=50)
    provision: bool = False  # also create one new, empty workspace in each agent and give the company those


class CompanyDelete(BaseModel):
    delete_data: bool = False  # also permanently delete the company's workspaces (default: keep the data)
    confirm: list[str] = Field(default_factory=list, max_length=100)  # "agent:workspace-id" of every workspace to be deleted


class CompanyPatch(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    deals: list[str] | None = Field(default=None, max_length=50)
    rfp: list[str] | None = Field(default=None, max_length=50)


class UserIn(BaseModel):
    email: str = Field(max_length=200)
    name: str | None = Field(default=None, max_length=120)
    company: str = Field(max_length=120)
    role: str = "member"
    password: str = Field(max_length=500)


class UserPatch(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    company: str | None = Field(default=None, max_length=120)
    role: str | None = None
    disabled: bool | None = None
    password: str | None = Field(default=None, max_length=500)


def _engine(request: Request):  # noqa: ANN202
    return request.app.state.engine


def _error(exc: AuthError) -> JSONResponse:
    return JSONResponse({"detail": exc.message, "code": exc.code}, status_code=exc.status)


async def agent_workspaces(request: Request) -> dict[str, dict[str, Any]]:
    """What each agent has, for the pickers: {"deals": {"up": bool, "workspaces": [{id, name, kind}]}, ...}."""
    engine, found = _engine(request), {}
    for agent_id, client in (("deals", engine.deal), ("rfp", engine.rfp)):
        try:
            _active, spaces = await client.workspaces()
            found[agent_id] = {"up": True, "name": AGENT_NAMES[agent_id], "workspaces": [
                {"id": s["id"], "name": s.get("name") or s["id"], "kind": s.get("kind", "")} for s in spaces]}
        except (AgentDown, UpstreamError):
            found[agent_id] = {"up": False, "name": AGENT_NAMES[agent_id], "workspaces": []}
    return found


async def agent_security(request: Request) -> dict[str, dict[str, Any]]:
    """Can anyone who can reach each agent read it without the hub? Asked the way an outsider would ask: no token, no
    workspace. "open" = yes, "protected" = refused, "down" = not running. `hub_token` says whether the hub holds the
    token it would need (an agent that is protected while the hub has no token cannot be used at all)."""
    engine, found = _engine(request), {}
    for agent_id, client in (("deals", engine.deal), ("rfp", engine.rfp)):
        token_env = client.agent.get("token_env", "")
        state = "down"
        try:
            reply = await client._http.get("/v1/workspaces", timeout=3.0)  # the client's own connection, but with none of its headers
            state = "open" if reply.status_code < 400 else "protected" if reply.status_code == 401 else "down"
        except httpx.HTTPError:
            pass
        found[agent_id] = {"name": AGENT_NAMES[agent_id], "state": state, "url": client.base_url,
                           "hub_token": bool(token_env and os.environ.get(token_env, "").strip()), "token_env": token_env}
    return found


async def _checked(request: Request, deals: list[str] | None, rfp: list[str] | None) -> dict[str, list[str]]:
    """The workspace ids to store, refusing one an agent that is running does not have."""
    known = await agent_workspaces(request)
    chosen: dict[str, list[str]] = {}
    for agent_id, ids in (("deals", deals), ("rfp", rfp)):
        if ids is None:
            continue
        have = {w["id"] for w in known[agent_id]["workspaces"]}
        if known[agent_id]["up"]:
            missing = [i for i in ids if i not in have]
            if missing:
                raise AuthError("unknown_workspace", f"{AGENT_NAMES[agent_id]} has no workspace {missing[0]!r}.", 422)
        chosen[agent_id] = ids
    return chosen


@router.post("/setup")
async def setup(body: SetupIn, request: Request) -> JSONResponse:
    """The first administrator, from this machine, while there are no accounts yet. Signs them in."""
    auth = get_auth(request)
    if auth is None or not setup_available(request):
        return JSONResponse({"detail": "Setup is only available on this machine, before the first account exists."}, status_code=403)
    try:
        auth.create_first_admin(body.email, body.password, body.name)
        token, user = auth.login(body.email, body.password, request.client.host if request.client else "")
    except AuthError as exc:
        return _error(exc)
    response = JSONResponse({"auth": "on", "user": user.public()})
    set_session_cookie(response, request, auth, token)
    return response


@router.get("/setup-info")
async def setup_info(request: Request) -> JSONResponse:
    """What the first-run form's workspace pickers need (the agents' workspaces), while setup is still open."""
    if not setup_available(request):
        return JSONResponse({"detail": "Setup is only available on this machine, before the first account exists."}, status_code=403)
    return JSONResponse({"agents": await agent_workspaces(request)})


@router.get("/overview")
async def overview(request: Request, admin: User = Depends(require_admin)) -> dict:
    auth = get_auth(request)
    companies, users = auth.list_companies(), auth.list_users()
    for company in companies:
        company["users"] = sum(1 for u in users if u["company_id"] == company["id"])
    for user in users:
        user["operator"] = user["company_id"] == OPERATOR_ID
    return {"me": admin.public(), "companies": companies, "users": users, "agents": await agent_workspaces(request),
            "security": await agent_security(request), "operator": {"id": OPERATOR_ID, "name": OPERATOR_NAME}}


@router.get("/usage")
async def usage_report(request: Request, admin: User = Depends(require_admin)) -> dict:
    """Per company: people, chats, model calls and tokens, and storage; read from records, no model is used."""
    from . import usage

    return await usage.build_report(_engine(request))


@router.get("/audit")
async def audit(request: Request, limit: int = Query(50, ge=1, le=500), admin: User = Depends(require_admin)) -> dict:
    return {"entries": get_auth(request).list_audit(limit)}


def _agent_clients(request: Request) -> dict[str, Any]:
    engine = _engine(request)
    return {"deals": engine.deal, "rfp": engine.rfp}


@router.post("/companies")
async def add_company(body: CompanyIn, request: Request, admin: User = Depends(require_admin)) -> JSONResponse:
    auth = get_auth(request)
    try:
        ticked = await _checked(request, body.deals, body.rfp)
        company_id = auth.create_company(body.name, ticked)  # first, so a duplicate name stops before anything is created
    except AuthError as exc:
        return _error(exc)
    created: dict[str, str] = {}
    if body.provision:
        try:
            created = await _provision(request, body.name, list(provisioning.AGENT_NAMES))
        except provisioning.ProvisionError as exc:
            with suppress(AuthError):
                auth.delete_company(company_id)  # nothing half-made is left behind
            return JSONResponse({"detail": f"{exc.message} Nothing was created.", "code": "provision_failed"}, status_code=502)
        auth.update_company(company_id, workspaces={a: [*ticked.get(a, []), ws] for a, ws in created.items()})
        for agent_id, workspace_id in created.items():
            auth.audit(admin.email, "workspace.create", f"{provisioning.AGENT_NAMES[agent_id]}: {workspace_id} for {body.name.strip()!r}")
    auth.audit(admin.email, "company.add", f"{body.name.strip()!r}")
    return JSONResponse({"ok": True, "created": created}, status_code=201)


async def _provision(request: Request, company_name: str, agent_ids: list[str]) -> dict[str, str]:
    """One new, empty workspace per listed agent, or none at all: if one fails, those already made are taken back out."""
    clients, created = _agent_clients(request), {}
    try:
        for agent_id in agent_ids:
            created[agent_id] = await provisioning.create_workspace(clients[agent_id], agent_id, company_name)
    except provisioning.ProvisionError:
        for agent_id, workspace_id in created.items():
            with suppress(AgentDown, UpstreamError):
                await provisioning.delete_workspace(clients[agent_id], workspace_id)
        raise
    return created


@router.post("/companies/{company_id}/provision")
async def provision_company(company_id: str, request: Request, admin: User = Depends(require_admin)) -> JSONResponse:
    """Give an existing company its own new, empty workspace in each agent where it has none yet."""
    auth = get_auth(request)
    try:
        company = _company(auth, company_id)
    except AuthError as exc:
        return _error(exc)
    missing = [a for a in provisioning.AGENT_NAMES if not company["workspaces"].get(a)]
    if not missing:
        return JSONResponse({"detail": "This company already has a workspace in both agents.", "code": "already_provisioned"}, status_code=409)
    try:
        created = await _provision(request, company["name"], missing)
    except provisioning.ProvisionError as exc:
        return JSONResponse({"detail": f"{exc.message} Nothing was created.", "code": "provision_failed"}, status_code=502)
    auth.update_company(company_id, workspaces={a: [*company["workspaces"].get(a, []), ws] for a, ws in created.items()})
    for agent_id, workspace_id in created.items():
        auth.audit(admin.email, "workspace.create", f"{provisioning.AGENT_NAMES[agent_id]}: {workspace_id} for {company['name']!r}")
    return JSONResponse({"ok": True, "created": created})


@router.patch("/companies/{company_id}")
async def edit_company(company_id: str, body: CompanyPatch, request: Request, admin: User = Depends(require_admin)) -> JSONResponse:
    auth = get_auth(request)
    try:
        workspaces = await _checked(request, body.deals, body.rfp)
        auth.update_company(company_id, name=body.name, workspaces=workspaces or None)
        changes = [k for k, v in (("name", body.name), ("deals", body.deals), ("rfp", body.rfp)) if v is not None]
        auth.audit(admin.email, "company.update", f"{company_id}: {', '.join(changes) or 'nothing'}")
    except AuthError as exc:
        return _error(exc)
    return JSONResponse({"ok": True})


@router.delete("/companies/{company_id}")
async def remove_company(company_id: str, request: Request, admin: User = Depends(require_admin)) -> JSONResponse:
    auth = get_auth(request)
    try:
        auth.delete_company(company_id)
        auth.audit(admin.email, "company.delete", company_id)
    except AuthError as exc:
        return _error(exc)
    return JSONResponse({"ok": True})


def _company(auth, company_id: str) -> dict[str, Any]:  # noqa: ANN001
    row = next((c for c in auth.list_companies() if c["id"] == company_id), None)
    if row is None:
        raise AuthError("no_company", f"No company {company_id!r}.", 404)
    return row


async def _plan(request: Request, company_id: str) -> tuple[dict[str, Any], int, list[provisioning.PlanItem]]:
    auth = get_auth(request)
    company = _company(auth, company_id)
    people = sum(1 for u in auth.list_users() if u["company_id"] == company_id)
    others = [c for c in auth.list_companies() if c["id"] != company_id]
    return company, people, await provisioning.deletion_plan(_agent_clients(request), company, others)


@router.get("/companies/{company_id}/deletion-plan")
async def deletion_plan(company_id: str, request: Request, admin: User = Depends(require_admin)) -> JSONResponse:
    """What deleting this company would do to its data, so the page can ask before anything is deleted."""
    try:
        company, people, plan = await _plan(request, company_id)
    except AuthError as exc:
        return _error(exc)
    return JSONResponse({"company": company["name"], "people": people, "workspaces": [i.public() for i in plan]})


@router.post("/companies/{company_id}/delete")
async def delete_company(company_id: str, body: CompanyDelete, request: Request, admin: User = Depends(require_admin)) -> JSONResponse:
    """Delete a company. Its workspaces and their data are kept unless `delete_data` is set AND `confirm` lists exactly the
    workspaces that would be deleted, so a page showing an older list cannot delete something it did not show."""
    auth = get_auth(request)
    try:
        company, people, plan = await _plan(request, company_id)
        if people:
            raise AuthError("company_in_use", "Move or disable its people first: this company still has accounts.", 409)
        deletable = sorted(i.key for i in plan if i.deletable)
        if body.delete_data and sorted(set(body.confirm)) != deletable:
            raise AuthError("confirm_mismatch", "The list of workspaces to delete has changed. Review it and confirm again.", 409)
        auth.delete_company(company_id)
    except AuthError as exc:
        return _error(exc)
    clients, deleted, kept = _agent_clients(request), [], []
    for item in plan:
        if not body.delete_data:
            kept.append({"key": item.key, "name": item.name, "reason": "you chose to keep the data"})
        elif not item.deletable:
            kept.append({"key": item.key, "name": item.name, "reason": item.reason})
        else:
            try:
                await provisioning.delete_workspace(clients[item.agent], item.id)
                deleted.append(item.key)
                auth.audit(admin.email, "workspace.delete", f"{provisioning.AGENT_NAMES[item.agent]}: {item.id} (company {company['name']!r} deleted)")
            except (AgentDown, UpstreamError) as exc:
                kept.append({"key": item.key, "name": item.name, "reason": provisioning._why(exc)})
    auth.audit(admin.email, "company.delete", f"{company_id}: " + (f"deleted {len(deleted)} workspace(s), kept {len(kept)}" if body.delete_data else "data kept"))
    return JSONResponse({"ok": True, "deleted": deleted, "kept": kept})


@router.post("/users")
async def add_user(body: UserIn, request: Request, admin: User = Depends(require_admin)) -> JSONResponse:
    auth = get_auth(request)
    try:
        auth.create_user(body.email, body.password, body.company, body.name, body.role)
        auth.audit(admin.email, "user.add", f"{body.email.strip().lower()} ({body.role}) in {body.company}")
    except AuthError as exc:
        return _error(exc)
    return JSONResponse({"ok": True}, status_code=201)


@router.patch("/users/{email}")
async def edit_user(email: str, body: UserPatch, request: Request, admin: User = Depends(require_admin)) -> JSONResponse:
    auth = get_auth(request)
    try:
        if body.password is not None:
            auth._check_password(body.password)  # before anything is changed, so a weak password leaves the account as it was
        auth.update_user(email, name=body.name, company=body.company, role=body.role, disabled=body.disabled)
        changes = [k for k, v in (("name", body.name), ("company", body.company), ("role", body.role),
                                  ("disabled", body.disabled)) if v is not None]
        if body.password is not None:
            auth.set_password(email, body.password)
            changes.append("password")
        auth.audit(admin.email, "user.update", f"{email.strip().lower()}: {', '.join(changes) or 'nothing'}")
    except AuthError as exc:
        return _error(exc)
    return JSONResponse({"ok": True})
