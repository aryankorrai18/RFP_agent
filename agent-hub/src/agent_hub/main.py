"""The hub: a launcher for the agents, a chat that does the work in them, and the old keyword demo chat."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from urllib.parse import urlsplit
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field

from .clients import AgentDown, DealClient, RfpClient, UpstreamError
from .admin_api import router as admin_router
from .auth import Auth, AuthError, User
from .deps import SESSION_COOKIE, get_auth, guard, set_session_cookie, setup_available
from .engine import Engine
from .llm import apply_env_file, get_llm
from .planner import Planner
from .router import Routing, load_registry, route
from . import pagepolicy, ratelimit
from .store import MAX_UPLOAD_BYTES, Store

ROOT = Path(__file__).resolve().parent.parent.parent
apply_env_file()  # HUB_MODEL, HUB_ROUTER_CAP and the key come from agent-hub/.env
AGENTS = load_registry()
MAX_TEXT_CHARS = 20_000


def build_engine() -> Engine:
    by_id = {a["id"]: a for a in AGENTS}
    store = Store()
    return Engine(store, DealClient(by_id["deals"]), RfpClient(by_id["rfp"]), AGENTS, ask_workspace=True,
                  planner=Planner(get_llm, cap=int(os.environ.get("HUB_ROUTER_CAP", "300"))), auth=Auth(store))


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    # Tests put their own engine (with fake agents) on app.state before startup; the hub then leaves it alone.
    owned = getattr(application.state, "engine", None) is None
    if owned:
        application.state.engine = build_engine()
    engine: Engine = application.state.engine
    await engine.resume()
    try:
        yield
    finally:
        await engine.shutdown()
        if owned:
            await engine.deal.aclose()
            await engine.rfp.aclose()
            engine.store.close()
            application.state.engine = None


app = FastAPI(title="Agent Hub", version="0.2.0", lifespan=lifespan)
STATUS_TTL_SECONDS = 5.0
_status_cache: tuple[float, dict[str, str]] = (0.0, {})


UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
PAGE_HEADERS = {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
                "Permissions-Policy": pagepolicy.PERMISSIONS_POLICY}


def _wants_page(request: Request) -> bool:
    """A browser asking for a page (not the API): it gets the designed error page instead of JSON."""
    return (not request.url.path.startswith("/api/") and request.method in ("GET", "HEAD")
            and "text/html" in request.headers.get("accept", ""))


@app.exception_handler(StarletteHTTPException)
async def page_or_json_errors(request: Request, exc: StarletteHTTPException):  # noqa: ANN201
    if exc.status_code == 404 and _wants_page(request):
        return pagepolicy.page(ROOT / "frontend" / "404.html", status_code=404, headers=PAGE_HEADERS)
    return await http_exception_handler(request, exc)  # the API keeps its JSON {"detail": ...}


@app.exception_handler(Exception)
async def crash(request: Request, exc: Exception):  # noqa: ANN201
    # Runs outside the middleware, so the security headers are set here by hand.
    if _wants_page(request):
        return pagepolicy.page(ROOT / "frontend" / "500.html", status_code=500, headers=PAGE_HEADERS)
    return JSONResponse({"detail": "Something went wrong on the hub."}, status_code=500, headers=PAGE_HEADERS)


@app.middleware("http")
async def security_headers(request: Request, call_next):  # noqa: ANN001, ANN201
    # A page on another site can make a browser send our cookie. Browsers name the page's origin on such requests, so a
    # state-changing request whose origin is not this host is refused (SameSite=Lax on the cookie is the second line).
    origin = request.headers.get("origin")
    if request.method in UNSAFE_METHODS and origin and request.url.path.startswith("/api/"):
        if urlsplit(origin).netloc != request.headers.get("host", ""):
            return JSONResponse({"detail": "That request came from another site."}, status_code=403)
    if ratelimit.enabled():
        rule = ratelimit.rule_for(request.method, request.url.path)
        if rule is not None:
            limiter = getattr(request.app.state, "rate_limiter", None)
            if limiter is None:
                limiter = request.app.state.rate_limiter = ratelimit.RateLimiter()
            wait = limiter.hit(rule, ratelimit.who(request.cookies.get(SESSION_COOKIE),
                                                     request.client.host if request.client else "", rule))
            if wait is not None:
                return JSONResponse(
                    {"detail": f"You're going a bit fast ({rule.name}). Try again in {wait} seconds."}, status_code=429,
                    headers={"Retry-After": str(wait), **PAGE_HEADERS, "Cache-Control": "no-store"})
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", pagepolicy.PERMISSIONS_POLICY)
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


async def _ping(client: httpx.AsyncClient, agent: dict) -> str:
    if agent["status"] != "live":
        return "planned"
    try:
        response = await client.get(agent["url"] + agent.get("health_path", "/health"))
        return "up" if response.status_code < 500 else "down"
    except httpx.HTTPError:
        return "down"


async def statuses() -> dict[str, str]:
    global _status_cache
    now = time.monotonic()
    if now - _status_cache[0] < STATUS_TTL_SECONDS and _status_cache[1]:
        return _status_cache[1]
    async with httpx.AsyncClient(timeout=1.0) as client:
        states = await asyncio.gather(*(_ping(client, a) for a in AGENTS))
    _status_cache = (now, {a["id"]: s for a, s in zip(AGENTS, states, strict=True)})
    return _status_cache[1]


def agent_view(agent: dict, state: str) -> dict:
    live = agent["status"] == "live"
    return {
        "id": agent["id"], "number": agent["number"], "name": agent["name"], "status": state,
        "description": agent["description"], "examples": agent["examples"],
        "open_url": agent["url"] + agent.get("open_path", "/") if live else None,
        "start_hint": agent.get("start_hint"),
    }


def reply_for(routing: Routing, states: dict[str, str]) -> tuple[str, dict | None, list[dict]]:
    live = [a for a in AGENTS if a["status"] == "live"]
    if routing.kind in ("greeting", "none"):
        intro = (
            "I can point you to the right agent. Right now I know about "
            + " and ".join(f"{a['name']} ({a['description'].split(',')[0].rstrip('.')})" for a in live)
            + f". Ask me something like \"{live[0]['examples'][0]}\"."
        )
        if routing.kind == "none":
            intro = "I'm not sure which agent fits that. " + intro
        return intro, None, [agent_view(a, states.get(a["id"], "down")) for a in live]
    best = routing.best
    agent = best.agent
    state = states.get(agent["id"], "down")
    why = ", ".join(f'"{w}"' for w in best.matched[:4])
    alternatives = [agent_view(m.agent, states.get(m.agent["id"], "planned")) for m in routing.alternatives]
    if agent["status"] != "live":
        closest = next((m for m in routing.alternatives if m.agent["status"] == "live"), None)
        text = f"That sounds like {agent['name']} (agent {agent['number']}, matched {why}), which isn't built yet."
        if closest:
            text += f" The closest agent available now is {closest.agent['name']}."
        return text, agent_view(agent, "planned"), alternatives
    if state == "up":
        text = f"That sounds like {agent['name']} (matched {why}). Opening it takes you straight to its workspace."
        return text, agent_view(agent, state), alternatives
    text = (
        f"That sounds like {agent['name']} (matched {why}), but it isn't running right now. "
        f"Start it with {agent.get('start_hint', 'its run.ps1')}, then open it here."
    )
    return text, agent_view(agent, state), alternatives


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return pagepolicy.page(ROOT / "frontend" / "app.html")


@app.get("/privacy", include_in_schema=False)
async def privacy_page() -> FileResponse:
    return pagepolicy.page(ROOT / "frontend" / "privacy.html")


@app.get("/favicon.svg", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "favicon.svg", media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico() -> Response:
    return Response(status_code=308, headers={"Location": "/favicon.svg"})

@app.get("/robots.txt", include_in_schema=False)
async def robots() -> Response:
    """A private work tool: ask every search engine not to index any of it."""
    return Response("User-agent: *\nDisallow: /\n", media_type="text/plain")


@app.get("/admin", include_in_schema=False)
async def admin_page() -> FileResponse:
    """The page itself is public (like the chat page); everything it shows comes from /api/admin, which needs an administrator."""
    return pagepolicy.page(ROOT / "frontend" / "admin.html")


app.include_router(admin_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "agents": len(AGENTS), "live": sum(1 for a in AGENTS if a["status"] == "live")}


@app.get("/api/agents")
async def list_agents(request: Request) -> dict:
    guard(request)
    states = await statuses()
    return {"agents": [agent_view(a, states.get(a["id"], "down")) for a in AGENTS]}


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request) -> JSONResponse:
    guard(request)
    routing = route(body.message, AGENTS)
    states = await statuses()
    text, agent, alternatives = reply_for(routing, states)
    return JSONResponse({
        "reply": text, "agent": agent, "alternatives": alternatives, "confidence": routing.confidence,
        "method": "keywords", "matched": routing.best.matched if routing.best else [],
    })


# -- the conversation API (the chat page talks only to these) ---------------------------------------------


class LoginIn(BaseModel):
    email: str = Field(max_length=200)
    password: str = Field(max_length=500)


@app.get("/api/auth/me")
async def auth_me(request: Request) -> dict:
    auth = get_auth(request)
    mode = auth.mode() if auth else "off"
    user = auth.user_for_token(request.cookies.get(SESSION_COOKIE)) if mode == "on" else None
    return {"auth": mode, "user": user.public() if user else None, "setup_available": setup_available(request)}


@app.post("/api/auth/login")
async def auth_login(body: LoginIn, request: Request) -> JSONResponse:
    auth = get_auth(request)
    if auth is None or auth.mode() != "on":
        raise HTTPException(400, "Sign-in is not turned on for this hub.")
    try:
        token, user = auth.login(body.email, body.password, request.client.host if request.client else "",
                                 request.headers.get("user-agent", ""))
    except AuthError as exc:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return JSONResponse({"detail": exc.message}, status_code=exc.status, headers=headers)
    response = JSONResponse({"auth": "on", "user": user.public()})
    set_session_cookie(response, request, auth, token, user)
    return response


def _signed_in(request: Request) -> tuple[Auth, User]:
    auth = get_auth(request)
    if auth is None or auth.mode() != "on":
        raise HTTPException(400, "Sign-in is not turned on for this hub.")
    user = guard(request)
    assert user is not None
    return auth, user


@app.get("/api/auth/sessions")
async def my_sessions(request: Request) -> dict:
    """Where the signed-in person is signed in: browser, address, when they signed in and last used it."""
    auth, user = _signed_in(request)
    return {"sessions": auth.list_sessions(user.id, request.cookies.get(SESSION_COOKIE)),
            "limits": {"idle_minutes": int((auth.admin_idle if user.is_admin else auth.idle).total_seconds() // 60),
                       "lifetime_hours": int(auth.lifetime(user.role).total_seconds() // 3600)}}


@app.delete("/api/auth/sessions/{session_id}")
async def end_my_session(session_id: str, request: Request) -> JSONResponse:
    auth, user = _signed_in(request)
    current = [s for s in auth.list_sessions(user.id, request.cookies.get(SESSION_COOKIE)) if s["current"]]
    if not auth.end_session(user.id, session_id):
        raise HTTPException(404, "That session has already ended.")
    auth.audit(user.email, "auth.session_ended", f"ended session {session_id}")
    response = JSONResponse({"ok": True, "signed_out": bool(current and current[0]["id"] == session_id)})
    if current and current[0]["id"] == session_id:
        response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.post("/api/auth/sessions/sign-out-others")
async def end_my_other_sessions(request: Request) -> dict:
    auth, user = _signed_in(request)
    ended = auth.end_other_sessions(user.id, request.cookies.get(SESSION_COOKIE))
    auth.audit(user.email, "auth.sessions_ended", f"signed out of {ended} other session(s)")
    return {"ended": ended}


@app.post("/api/auth/logout")
async def auth_logout(request: Request) -> JSONResponse:
    auth = get_auth(request)
    if auth:
        auth.logout(request.cookies.get(SESSION_COOKIE))
    response = JSONResponse({"auth": auth.mode() if auth else "off", "user": None})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


def get_engine(request: Request) -> Engine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(503, "The hub is still starting.")
    return engine


def known_conversation(engine: Engine, conv: str, request: Request) -> None:
    """The chat exists and belongs to the signed-in person. Someone else's chat looks exactly like a missing one."""
    user = guard(request)
    if not engine.store.conversation_exists(conv) or (user is not None and engine.store.conversation_owner(conv) != user.id):
        raise HTTPException(404, "That conversation does not exist.")


def public_state(st: dict) -> dict:
    """What the page may know about the conversation (not the action tokens or file paths)."""
    pending = st.get("pending") or {}
    return {
        "flow": st.get("flow"), "deal": st.get("deal_label"), "project": (st.get("project") or {}).get("name"),
        "awaiting": (st.get("awaiting") or {}).get("type"), "pending_cost": pending.get("cost"),
        "workspaces": dict(st.get("workspace_names") or {}), "company": st.get("company_name"),
        "router_calls": int(st.get("router_calls", 0)),
    }


def event_page(engine: Engine, conv: str, after: int) -> dict:
    events = engine.store.events(conv, after)
    return {"events": events, "next": events[-1]["n"] if events else after, "busy": engine.busy(conv),
            "workspaces": dict(engine.load(conv).get("workspace_names") or {}), "company": engine.load(conv).get("company_name"),
            "router_calls": int(engine.load(conv).get("router_calls", 0))}


class ActionIn(BaseModel):
    action_id: str = Field(min_length=1, max_length=200)


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=80)


@app.get("/api/conversations")
async def list_conversations(request: Request, limit: int = Query(30, ge=1, le=100), before: str | None = Query(None, max_length=120),
                             q: str | None = Query(None, max_length=100)) -> dict:
    """The signed-in person's chat history (only theirs), latest first; `next` is the cursor for the page after."""
    user = guard(request)
    items, nxt = get_engine(request).store.list_conversations(user.id if user else None, limit=limit, before=before, query=q)
    return {"conversations": items, "next": nxt}


@app.patch("/api/conversations/{conv}")
async def rename_conversation(conv: str, body: RenameIn, request: Request) -> dict:
    engine = get_engine(request)
    known_conversation(engine, conv, request)
    title = " ".join(body.title.split())
    if not title:
        raise HTTPException(422, "Give the chat a name.")
    engine.store.rename_conversation(conv, title)
    return {"id": conv, "title": title}


@app.delete("/api/conversations/{conv}")
async def delete_conversation(conv: str, request: Request) -> dict:
    engine = get_engine(request)
    known_conversation(engine, conv, request)
    engine.store.delete_conversation(conv)
    return {"ok": True}


@app.post("/api/conversations")
async def new_conversation(request: Request) -> dict:
    return {"id": get_engine(request).new_conversation(guard(request))}


@app.get("/api/conversations/{conv}")
async def get_conversation(conv: str, request: Request) -> dict:
    engine = get_engine(request)
    known_conversation(engine, conv, request)
    return {"id": conv, "title": engine.store.conversation_title(conv), "state": public_state(engine.load(conv)),
            **event_page(engine, conv, 0)}


@app.post("/api/conversations/{conv}/messages")
async def post_message(
    conv: str, request: Request, text: str | None = Form(None), files: list[UploadFile] = File(default_factory=list),
    _signed_in: User | None = Depends(guard),  # first, so a request with no session is refused before its body is read
) -> dict:
    engine = get_engine(request)
    known_conversation(engine, conv, request)
    text = (text or "").strip()
    attached = [f for f in files if f.filename]
    if not text and not attached:
        raise HTTPException(422, "Type a message or attach a file first.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(422, f"That message is too long (over {MAX_TEXT_CHARS} characters). Attach it as a file instead.")
    payload: list[tuple[str, bytes]] = []
    for f in attached:
        data = await f.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{f.filename} is over the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
        payload.append((f.filename, data))
    return {"accepted": True, "n": engine.accept_message(conv, text, payload)}


@app.get("/api/conversations/{conv}/events")
async def get_events(conv: str, request: Request, after: int = Query(0, ge=0)) -> dict:
    engine = get_engine(request)
    known_conversation(engine, conv, request)
    return event_page(engine, conv, after)


@app.post("/api/conversations/{conv}/actions")
async def post_action(conv: str, body: ActionIn, request: Request, _signed_in: User | None = Depends(guard)) -> dict:
    engine = get_engine(request)
    known_conversation(engine, conv, request)
    engine.accept_action(conv, body.action_id)
    return {"accepted": True}


@app.get("/api/conversations/{conv}/downloads/{token}")
async def get_download(conv: str, token: str, request: Request) -> Response:
    engine = get_engine(request)
    known_conversation(engine, conv, request)
    try:
        content, media_type, disposition = await engine.download(conv, token)
    except KeyError:
        raise HTTPException(404, "That download link is not valid.") from None
    except AgentDown as exc:
        raise HTTPException(503, exc.message) from exc
    except UpstreamError as exc:
        raise HTTPException(409 if exc.status == 409 else 502, f"{exc.message} {exc.action or ''}".strip()) from exc
    return Response(content, media_type=media_type, headers={"Content-Disposition": disposition})
