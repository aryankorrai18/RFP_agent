"""FastAPI entry point for Deal Intelligence."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from dotenv import dotenv_values
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from hindsight_client import Hindsight
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import workspaces
from .api.v1.context import build_context
from .api.v1.router import router as v1_router
from .config import PROVIDER_KEY_VARS, ROOT, Settings, has_key
from .errors import PipelineError
from .parsing.parser import ParseError
from .providers.base import LLM, MonitoredLLM
from .providers.claude import ClaudeLLM
from .providers.gemini import GeminiLLM
from .providers.groq import GroqLLM
from .providers.model_choice import (
    MODEL_NAME, apply_choice, default_model, list_models, model_source, refresh, write_choice,
)

ENV_FILE = ROOT / ".env"


def apply_env_file() -> None:
    """Re-read .env so a key saved there takes effect on the next request, without a restart.
    Only non-empty values are applied, so a blank line never erases a key set in the shell."""
    for key, value in dotenv_values(ENV_FILE).items():
        if value:
            os.environ[key] = value


apply_env_file()

PARSE_ERROR_STATUS = {
    "unsupported_file_type": 415,
    "document_too_long": 413,
    "unreadable_file": 422,
    "empty_document": 422,
}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Start the workspace services, resume jobs, and start Hindsight outbox sync.
    Tests replace the context by setting app.state.v1_factory before starting the app."""
    if getattr(app.state, "v1_factory", None) is None:
        workspaces.ensure_registry(base_settings())  # first run: the environment's own data becomes workspace "main"
    ctx = _context_factory(app)()
    app.state.v1 = ctx
    await ctx.startup()
    try:
        yield
    finally:
        await _close_pooled()
        await ctx.shutdown()
        app.state.v1 = None


app = FastAPI(title="Deal Intelligence", version="0.1.0", lifespan=lifespan)
app.include_router(v1_router)


# --- one context per workspace a caller names (the X-Workspace header) ------------------------------

_pool_lock = asyncio.Lock()
app.state.pool = {}  # workspace id -> context, for workspaces other than the active one


async def _context_for(workspace_id: str):  # noqa: ANN202
    """The services for a request that names its workspace. The active workspace uses the main context; any
    other one is opened once and kept, so two callers can work in different workspaces at the same time and
    nobody has to switch the workspace the app's own screens show."""
    active = workspaces.active()
    current = getattr(app.state, "v1", None)
    if active is not None and active.id == workspace_id and current is not None:
        return current
    if workspaces.get(workspace_id) is None:
        raise PipelineError("not_found", f"No workspace {workspace_id!r}.", 404)
    async with _pool_lock:
        pool = app.state.pool
        if workspace_id not in pool:
            factory = getattr(app.state, "workspace_factory", None)  # tests supply their own
            ctx = factory(workspace_id) if factory else build_context(
                lambda: workspaces.apply_workspace(base_settings(), workspace_id), _llm_for)
            ctx.workspace_id = workspace_id
            await ctx.startup()
            pool[workspace_id] = ctx
        return pool[workspace_id]


async def _close_pooled(workspace_id: str | None = None, *, check_busy: bool = False) -> None:
    """Close the context opened for one workspace (or all). Refuses, when asked, while one of its jobs is running."""
    async with _pool_lock:
        pool = app.state.pool
        for key in [workspace_id] if workspace_id else list(pool):
            ctx = pool.get(key)
            if ctx is None:
                continue
            if check_busy and ctx.jobs.busy():
                raise PipelineError(
                    "workspace_busy", "A background job is still running in this workspace. Let it finish or stop it, then switch.", 409
                )
            del pool[key]
            await ctx.shutdown()


app.state.context_for = _context_for


@app.middleware("http")
async def service_token(request: Request, call_next):  # noqa: ANN001, ANN201
    """When DEAL_SERVICE_TOKEN is set, only callers that hold it (the Agent Hub) may use the API. The agents' own screens are then
    for a person on this machine with the token unset; once the agents sit behind the hub they are not reachable by
    clients at all. Unset (the default) changes nothing."""
    expected = os.environ.get("DEAL_SERVICE_TOKEN", "").strip()
    if expected and request.url.path.startswith("/v1/"):
        given = request.headers.get("x-service-token", "")
        if not hmac.compare_digest(given.encode(), expected.encode()):
            return JSONResponse(status_code=401, content={"error": {"code": "service_token", "message": "This API only accepts calls from the Agent Hub."}})
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):  # noqa: ANN001, ANN201
    """Apply safe browser defaults and prevent sensitive API responses being cached."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if request.url.path.startswith("/v1/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


def base_settings() -> Settings:
    """Settings from .env and the UI model choice, before the active workspace is applied."""
    apply_env_file()
    return apply_choice(Settings.from_env())


def get_settings() -> Settings:
    """Settings for the active workspace: its database, uploads and Hindsight banks."""
    return workspaces.apply_workspace(base_settings())


def _context_factory(app: FastAPI):  # noqa: ANN202
    return getattr(app.state, "v1_factory", None) or (lambda: build_context(get_settings, _llm_for))


@lru_cache
def _llm_for(settings: Settings) -> LLM:
    if settings.provider == "gemini":
        inner: LLM = GeminiLLM(settings)
    else:
        inner = GroqLLM(settings) if settings.provider == "groq" else ClaudeLLM(settings)
    return MonitoredLLM(inner, settings.provider)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(PipelineError)
async def _pipeline_error(_: Request, exc: PipelineError) -> JSONResponse:
    return _error(exc.http_status, exc.code, exc.message)


@app.exception_handler(ParseError)
async def _parse_error(_: Request, exc: ParseError) -> JSONResponse:
    return _error(PARSE_ERROR_STATUS.get(exc.code, 422), exc.code, exc.message)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = "not_found" if exc.status_code == 404 else "http_error"
    return _error(exc.status_code, code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def _request_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    return _error(422, "invalid_request", problems)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "app.html")


@app.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "provider": settings.provider,
        "model": settings.model,
        "llm_credentials": "env" if has_key(settings.provider) else "not_found_in_env",
        "key_variable": PROVIDER_KEY_VARS[settings.provider][0],
    }


class ModelChoiceIn(BaseModel):
    model: str | None = None  # None resets to DEAL_MODEL from .env, or the provider default


async def _model_view(settings: Settings) -> dict:
    options, list_error = await list_models(settings.provider)
    return {
        "provider": settings.provider,
        "current": settings.model,
        "source": model_source(settings),
        "env_model": os.environ.get("DEAL_MODEL", "").strip() or None,
        "default": default_model(settings.provider),
        "options": [{"id": o.id, "label": o.label, "input_token_limit": o.input_token_limit} for o in options],
        "list_error": list_error,
    }


@app.get("/v1/models")
async def models(settings: Settings = Depends(get_settings)) -> dict:
    """The active model, where it comes from, and the models this provider offers."""
    return await _model_view(settings)


@app.put("/v1/models")
async def choose_model(body: ModelChoiceIn, settings: Settings = Depends(get_settings)) -> dict:
    """Choose the model for all later calls. A running job finishes with its model."""
    chosen = (body.model or "").strip() or None
    if chosen is not None:
        if not MODEL_NAME.match(chosen):
            raise PipelineError("invalid_model", f"{chosen!r} isn't a valid model name.", 422)
        options, list_error = await list_models(settings.provider)
        if options and chosen not in {o.id for o in options}:
            raise PipelineError("unknown_model", f"{chosen} isn't available to this {settings.provider} key.", 422)
        if not options and list_error and settings.provider != "anthropic":
            raise PipelineError("model_list_unavailable", list_error, 503)
    write_choice(settings.provider, chosen)
    return await _model_view(refresh(settings))


# --- workspaces ------------------------------------------------------------------------------------

_switch_lock = asyncio.Lock()


class WorkspaceIn(BaseModel):
    name: str | None = None
    kind: str = "company"  # company (start from scratch) | demo (seeded fictional history)


def _workspace_view(space: workspaces.Workspace, active_id: str | None) -> dict:
    return {"id": space.id, "name": space.name, "kind": space.kind, "created_at": space.created_at,
            "active": space.id == active_id}


async def _switch(workspace_id: str) -> None:
    """Close the current workspace's memory and open another's. Refused while a job is running."""
    async with _switch_lock:
        target = workspaces.get(workspace_id)
        if target is None:
            raise PipelineError("not_found", f"No workspace {workspace_id!r}.", 404)
        current = getattr(app.state, "v1", None)
        if current is not None and current.jobs.busy():
            raise PipelineError(
                "workspace_busy", "A background job is still running in this workspace. Let it finish or stop it, then switch.", 409
            )
        await _close_pooled(workspace_id, check_busy=True)  # one context per workspace: the main one takes over
        previous = workspaces.active()
        if current is not None:
            await current.shutdown()
        workspaces.activate(workspace_id)
        try:
            ctx = _context_factory(app)()
            await ctx.startup()
        except Exception:
            if previous is not None:  # put things back as they were
                workspaces.activate(previous.id)
                ctx = _context_factory(app)()
                await ctx.startup()
                app.state.v1 = ctx
            raise
        app.state.v1 = ctx


@app.get("/v1/workspaces")
async def list_workspaces() -> dict:
    active_id, spaces = workspaces.load()
    return {"active": active_id, "workspaces": [_workspace_view(w, active_id) for w in spaces]}


@app.post("/v1/workspaces", status_code=201)
async def create_workspace(body: WorkspaceIn) -> dict:
    """Start a new workspace and switch to it. kind=demo seeds a fictional seller's deal history (no
    model calls; Hindsight stores the seeded summaries and interactions, and lessons are built from
    the closed deals); kind=company starts empty. Every demo gets fresh Hindsight banks."""
    from .api.v1 import demo, lessons, outcomes

    name = (body.name or "").strip() or ("Halcyon Software (demo)" if body.kind == "demo" else "")
    try:
        space = workspaces.create(name, body.kind, base_settings())
    except ValueError as exc:
        raise PipelineError("invalid_request", str(exc), 422) from exc
    await _switch(space.id)
    seeded = None
    if body.kind == "demo":
        seeded = demo.seed_demo(app.state.v1)
        outcomes.rebuild_play_stats(app.state.v1.db)
        lessons.collect_lessons(app.state.v1.db)
        app.state.v1.schedule_sync()
    return {"workspace": _workspace_view(space, space.id), "seeded": seeded}


@app.post("/v1/workspaces/{workspace_id}/activate")
async def activate_workspace(workspace_id: str) -> dict:
    await _switch(workspace_id)
    active_id, spaces = workspaces.load()
    return {"active": active_id, "workspaces": [_workspace_view(w, active_id) for w in spaces]}


async def _delete_hindsight_banks(settings: Settings, space: workspaces.Workspace) -> None:
    """Best-effort: also remove the workspace's two Hindsight banks, so no cloud data is left
    behind. Never raises: a bank that was never created, or an unreachable Hindsight, isn't fatal.
    A separate function so tests can replace it instead of making a real network call."""
    cloud = "vectorize.io" in settings.hindsight_url
    if cloud and not settings.hindsight_api_key:  # a cloud bank needs a key to reach; local doesn't
        return
    for bank in (space.bank, space.lessons_bank):
        client = Hindsight(base_url=settings.hindsight_url, api_key=settings.hindsight_api_key, max_attempts=1)
        try:
            await client.adelete_bank(bank)
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                await client.aclose()


@app.delete("/v1/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str) -> dict:
    """Permanently delete a workspace: its database, uploads and its two Hindsight banks.
    Refused for "main" and for the currently active workspace (switch away first)."""
    await _close_pooled(workspace_id)  # a context opened for it by name must not outlive it
    try:
        space = workspaces.remove(workspace_id)
    except KeyError as exc:
        raise PipelineError("not_found", f"No workspace {workspace_id!r}.", 404) from exc
    except ValueError as exc:
        raise PipelineError("invalid_request", str(exc), 422) from exc
    await _delete_hindsight_banks(base_settings(), space)
    active_id, spaces = workspaces.load()
    return {"active": active_id, "workspaces": [_workspace_view(w, active_id) for w in spaces]}


@app.get("/v1/samples/{name}", include_in_schema=False)
async def sample(name: str) -> FileResponse:
    """Bundled sample files the page can load with one click. The name must match a file that is
    directly inside samples/; it is never used as a path."""
    folder = ROOT / "samples"
    known = {p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md", ".eml", ".csv"}} \
        if folder.is_dir() else set()
    if name not in known:
        raise PipelineError("sample_not_found", f"No sample named {name!r}.", 404)
    return FileResponse(folder / name, filename=name)
