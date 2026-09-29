"""FastAPI entry point for the RFP Memory Assistant."""

from __future__ import annotations

import asyncio
import contextlib
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
from .config import PROVIDER_KEY_VARS, ROOT, Settings, has_key
from .llm import LLM, ClaudeLLM, MonitoredLLM
from .llm_gemini import GeminiLLM
from .llm_groq import GroqLLM
from .model_choice import (
    MODEL_NAME, apply_choice, default_model, list_models, model_source, refresh, write_choice,
)
from .parser import ParseError
from .core import PipelineError
from .v1.api import router as v1_router
from .v1.context import build_context

ENV_FILE = ROOT / ".env"


def apply_env_file() -> None:
    """Re-read .env so a key saved there takes effect on the next request, without a restart.
    Only non-empty values are applied, so a blank line never erases a key set in the shell."""
    for key, value in dotenv_values(ENV_FILE).items():
        if value:
            os.environ[key] = value


apply_env_file()

# Bundled sample documents the page can load with one click: name -> file under samples/. A fixed
# list, never a path from the request.
SAMPLE_FILES = {
    "sample_rfp.docx": "sample_rfp.docx",
    "past_northwind_bank_2025.docx": "past_northwind_bank_2025.docx",
    "past_meridian_health_2026.docx": "past_meridian_health_2026.docx",
    "past_cobalt_insurance_2025.docx": "past_cobalt_insurance_2025.docx",
    # The demo workspace's ready-to-run RFPs.
    "2026-12_ashford_community_bank_rfp.docx": "corpus_v2/2026-12_ashford_community_bank_rfp.docx",
    "2026-12_meadowgate_health_questionnaire.xlsx": "corpus_v2/2026-12_meadowgate_health_questionnaire.xlsx",
}

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
        workspaces.ensure_registry(base_settings())  # first run: today's data becomes workspace "main"
    ctx = _context_factory(app)()
    app.state.v1 = ctx
    await ctx.startup()
    try:
        yield
    finally:
        await ctx.shutdown()
        app.state.v1 = None


app = FastAPI(title="RFP Memory Assistant", version="4.0.0", lifespan=lifespan)
app.include_router(v1_router)


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
    """Settings for the active workspace: its database, uploads, fact sheet and Hindsight banks."""
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
    model: str | None = None  # None resets to RFP_MODEL from .env, or the provider default


async def _model_view(settings: Settings) -> dict:
    options, list_error = await list_models(settings.provider)
    return {
        "provider": settings.provider,
        "current": settings.model,
        "source": model_source(settings),
        "env_model": os.environ.get("RFP_MODEL", "").strip() or None,
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
    """Choose the model for all later calls. A running drafting job finishes with its model."""
    chosen = (body.model or "").strip() or None
    if chosen is not None:
        if not MODEL_NAME.match(chosen):
            raise PipelineError("invalid_model", f"{chosen!r} isn't a valid model name.", 422)
        options, list_error = await list_models(settings.provider)
        if options and chosen not in {o.id for o in options}:
            raise PipelineError(
                "unknown_model", f"{chosen} isn't available to this {settings.provider} key.", 422
            )
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
            "fact_sheet_locked": space.fact_sheet_locked, "active": space.id == active_id}


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
    """Start a new workspace and switch to it. kind=demo seeds a fictional company's history (no
    model calls; Hindsight stores the seeded answers and lessons); kind=company starts empty."""
    from .v1 import demo

    name = (body.name or "").strip() or ("Larkspur Data (demo)" if body.kind == "demo" else "")
    try:
        space = workspaces.create(name, body.kind, base_settings())
    except ValueError as exc:
        raise PipelineError("invalid_request", str(exc), 422) from exc
    await _switch(space.id)
    seeded = demo.seed_demo(app.state.v1) if body.kind == "demo" else None
    return {"workspace": _workspace_view(space, space.id), "seeded": seeded}


@app.post("/v1/workspaces/{workspace_id}/activate")
async def activate_workspace(workspace_id: str) -> dict:
    await _switch(workspace_id)
    active_id, spaces = workspaces.load()
    return {"active": active_id, "workspaces": [_workspace_view(w, active_id) for w in spaces]}


async def _delete_hindsight_banks(settings: Settings, space: workspaces.Workspace) -> None:
    """Best-effort: also remove the workspace's two Hindsight banks, so no cloud data is left
    behind. Never raises — a bank that was never created, or an unreachable Hindsight, isn't fatal.
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
    """Permanently delete a workspace: its database, uploads, fact sheet, and its two Hindsight
    banks. Refused for "main" and for the currently active workspace (switch away first)."""
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
    if name not in SAMPLE_FILES:
        raise PipelineError("sample_not_found", f"No sample named {name!r}.", 404)
    return FileResponse(ROOT / "samples" / SAMPLE_FILES[name], filename=name)


