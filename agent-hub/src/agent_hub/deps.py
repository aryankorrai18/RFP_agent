"""What every protected route needs: who is signed in, who is an administrator, and the session cookie."""

from __future__ import annotations

import os

from fastapi import HTTPException, Request
from fastapi.responses import Response

from .auth import Auth, User

SESSION_COOKIE = "hub_session"
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def get_auth(request: Request) -> Auth | None:
    engine = getattr(request.app.state, "engine", None)
    return getattr(engine, "auth", None)


def guard(request: Request) -> User | None:
    """The signed-in person, or None when sign-in is off. Raises 401 when it is on and there is no valid session."""
    auth = get_auth(request)
    if auth is None or auth.mode() != "on":
        return None
    user = auth.user_for_token(request.cookies.get(SESSION_COOKIE))
    if user is None:
        raise HTTPException(401, "Sign in to continue.")
    return user


def require_admin(request: Request) -> User:
    """An administrator with a valid session. Everything on the admin page goes through this."""
    auth = get_auth(request)
    if auth is None or auth.mode() != "on":
        raise HTTPException(401, "Sign in to continue.")
    user = guard(request)
    assert user is not None
    if not user.is_admin:
        raise HTTPException(403, "Only an administrator can do that.")
    return user


def is_loopback(request: Request) -> bool:
    """True when the request comes from this machine itself."""
    return bool(request.client) and request.client.host in LOOPBACK


def setup_available(request: Request) -> bool:
    """First-run setup (the first administrator, made in the browser) is open only while there are no accounts at all, only
    from this machine, and not when sign-in was switched off on purpose."""
    auth = get_auth(request)
    if auth is None or os.environ.get("HUB_AUTH", "auto").strip().lower() == "off":
        return False
    return not auth.has_users() and is_loopback(request)


def set_session_cookie(response: Response, request: Request, auth: Auth, token: str) -> None:
    secure = request.url.scheme == "https" or os.environ.get("HUB_COOKIE_SECURE", "").strip() in ("1", "true", "on")
    response.set_cookie(SESSION_COOKIE, token, max_age=auth.session_days * 86400, httponly=True, samesite="lax",
                        secure=secure, path="/")
