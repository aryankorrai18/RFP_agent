"""Sign-in and the company each person belongs to.

A company is a name plus the workspaces it may use in each agent (its own deals, its own RFP library). A user belongs
to exactly one company; the company, not the person, decides which workspaces the hub will ever send to an agent, so
one company can never see another's data however the person words a request. Accounts are made by an administrator
(`python -m agent_hub.admin`); there is no self sign-up.

Passwords are stored as salted scrypt hashes. A session is a random token in an HttpOnly cookie; only its SHA-256 is
stored, so a copy of the database cannot be replayed as a session. Failed sign-ins are counted per email and address
and locked out for a while, and an unknown email costs the same time as a wrong password.

`HUB_AUTH` = on (sign-in always required), off (open, for one person on their own computer) or auto (the default:
required once the first account exists)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .store import Store

AGENT_IDS = ("deals", "rfp")
ROLES = ("member", "admin")
# Administrators run the hub; they are not a customer. They all belong to this built-in team, which can never be given company
# data, renamed or deleted, and only it can hold administrators. Customer companies hold members and their own data.
OPERATOR_ID, OPERATOR_NAME = "hub-administrators", "Hub administrators"
MIN_PASSWORD_CHARS = 10
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1
MAX_FAILURES, FAILURE_WINDOW = 5, timedelta(minutes=15)
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# What a wrong password costs, spent on an unknown email too, so the two cannot be told apart by timing.
_DUMMY_HASH: str | None = None


class AuthError(Exception):
    """A sign-in or account problem. `message` is safe to show; `status` is the HTTP status to use."""

    def __init__(self, code: str, message: str, status: int = 400, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.status, self.retry_after = code, message, status, retry_after


@dataclass(frozen=True)
class User:
    id: str
    email: str
    name: str | None
    company_id: str
    company_name: str
    workspaces: dict[str, tuple[str, ...]]  # agent id -> the workspace ids this company may use
    role: str = "member"  # "admin" may manage companies and accounts in the admin page

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_operator(self) -> bool:
        """An administrator of the hub itself: manages companies and people, never sees a company's data."""
        return self.company_id == OPERATOR_ID

    def allowed(self, agent_id: str) -> tuple[str, ...]:
        return self.workspaces.get(agent_id, ())

    def public(self) -> dict[str, Any]:
        return {"email": self.email, "name": self.name, "company": self.company_name, "role": self.role, "operator": self.is_operator}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(
        SCRYPT_N, SCRYPT_R, SCRYPT_P, base64.b64encode(salt).decode(), base64.b64encode(digest).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        candidate = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(candidate, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "company"


def describe_browser(user_agent: str | None) -> str:
    """A short, human description of a browser ("Chrome on Windows"), never the raw header."""
    ua = user_agent or ""
    system = next((name for marker, name in (("Windows", "Windows"), ("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android"),
                                              ("CrOS", "ChromeOS"), ("Macintosh", "macOS"), ("Mac OS X", "macOS"), ("Linux", "Linux"))
                   if marker in ua), "")
    browser = next((name for marker, name in (("Edg/", "Edge"), ("OPR/", "Opera"), ("Firefox/", "Firefox"), ("Chrome/", "Chrome"),
                                               ("Safari/", "Safari")) if marker in ua), "")
    if not (browser or system):
        return "Unknown browser" if ua else "Unknown device"
    return f"{browser or 'Browser'} on {system}" if system else browser


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Auth:
    def __init__(self, store: Store, session_days: int | None = None) -> None:
        self.store = store
        self.session_days = session_days or int(os.environ.get("HUB_SESSION_DAYS", "7"))
        # A session ends at its lifetime or after this long without use, whichever comes first. Administrators get
        # shorter ones: their sessions can change every company's setup.
        self.idle = timedelta(hours=float(os.environ.get("HUB_SESSION_IDLE_HOURS", "12")))
        self.admin_lifetime = timedelta(hours=float(os.environ.get("HUB_ADMIN_SESSION_HOURS", "12")))
        self.admin_idle = timedelta(minutes=float(os.environ.get("HUB_ADMIN_IDLE_MINUTES", "60")))
        self._lookups = 0
        if store.exists():
            self.ensure_operator()

    def ensure_operator(self) -> None:
        """The administrators' team exists, and every administrator is in it (older hubs kept admins inside a customer company)."""
        if self.store.sql_one("SELECT 1 FROM companies WHERE id = ?", (OPERATOR_ID,)) is None:
            self.store.sql_exec("INSERT INTO companies VALUES (?, ?, ?, ?)",
                                (OPERATOR_ID, OPERATOR_NAME, json.dumps({a: [] for a in AGENT_IDS}), _iso(_now())))
        moved = self.store.sql_exec("UPDATE users SET company_id = ? WHERE role = 'admin' AND company_id != ?", (OPERATOR_ID, OPERATOR_ID))
        if moved:
            self.audit("hub", "operator.move", f"moved {moved} administrator(s) to {OPERATOR_NAME}; administrators no longer use a company's data")

    @staticmethod
    def _refuse_operator() -> None:
        raise AuthError("operator_company", f"{OPERATOR_NAME} is the hub's own team: it can never be given company data, renamed or deleted.", 409)

    # -- mode -------------------------------------------------------------------------------------

    def mode(self) -> str:
        """"on" or "off", from HUB_AUTH (auto = on once an active account exists)."""
        setting = os.environ.get("HUB_AUTH", "auto").strip().lower()
        if setting in ("on", "off"):
            return setting
        if not self.store.exists():
            return "off"
        row = self.store.sql_one("SELECT 1 FROM users WHERE disabled = 0 LIMIT 1")
        return "on" if row else "off"

    # -- administration ---------------------------------------------------------------------------

    @staticmethod
    def _clean_workspaces(workspaces: dict[str, list[str]] | None) -> dict[str, list[str]]:
        cleaned: dict[str, list[str]] = {}
        for agent_id, ids in (workspaces or {}).items():
            if agent_id not in AGENT_IDS:
                raise AuthError("bad_company", f"Unknown agent {agent_id!r}; use {' or '.join(AGENT_IDS)}.")
            cleaned[agent_id] = list(dict.fromkeys(i.strip() for i in ids if i and i.strip()))
        return cleaned

    def create_company(self, name: str, workspaces: dict[str, list[str]] | None = None) -> str:
        self.ensure_operator()
        name = (name or "").strip()
        if not name:
            raise AuthError("bad_company", "A company needs a name.")
        company_id = _slug(name)
        if company_id == OPERATOR_ID or name.lower() == OPERATOR_NAME.lower():
            raise AuthError("reserved_name", f"\"{OPERATOR_NAME}\" is reserved for the hub's administrators. Choose another name.", 409)
        if self.store.sql_one("SELECT 1 FROM companies WHERE id = ? OR name = ?", (company_id, name)):
            raise AuthError("company_exists", f"There is already a company called {name!r}.", 409)
        self.store.sql_exec("INSERT INTO companies VALUES (?, ?, ?, ?)",
                            (company_id, name, json.dumps(self._clean_workspaces(workspaces)), _iso(_now())))
        return company_id

    def set_company_workspaces(self, company: str, workspaces: dict[str, list[str]]) -> None:
        row = self._company(company)
        if row["id"] == OPERATOR_ID and any(self._clean_workspaces(workspaces).values()):
            self._refuse_operator()
        current = json.loads(row["workspaces"])
        current.update(self._clean_workspaces(workspaces))
        self.store.sql_exec("UPDATE companies SET workspaces = ? WHERE id = ?", (json.dumps(current), row["id"]))

    def _company(self, company: str):  # noqa: ANN202
        row = self.store.sql_one("SELECT * FROM companies WHERE id = ? OR name = ?", (company, company))
        if row is None:
            raise AuthError("no_company", f"No company {company!r}. Create it first.", 404)
        return row

    def list_companies(self, include_operator: bool = False) -> list[dict[str, Any]]:
        """Customer companies (and, when asked, the administrators' team)."""
        if not self.store.exists():
            return []
        return [{"id": r["id"], "name": r["name"], "workspaces": json.loads(r["workspaces"])}
                for r in self.store.sql_all("SELECT * FROM companies ORDER BY name") if include_operator or r["id"] != OPERATOR_ID]

    def create_user(self, email: str, password: str, company: str, name: str | None = None, role: str = "member") -> str:
        if role not in ROLES:
            raise AuthError("bad_role", f"A role is one of: {', '.join(ROLES)}.")
        email = (email or "").strip().lower()
        if not EMAIL.match(email):
            raise AuthError("bad_email", "That does not look like an email address.")
        self._check_password(password)
        self.ensure_operator()
        company_row = self._company(OPERATOR_ID if role == "admin" else company)  # an administrator always joins the hub's own team
        if role != "admin" and company_row["id"] == OPERATOR_ID:
            raise AuthError("operator_company", f"Only administrators belong to {OPERATOR_NAME}. Choose the company this person works for.")
        if self.store.sql_one("SELECT 1 FROM users WHERE email = ?", (email,)):
            raise AuthError("user_exists", f"{email} already has an account.", 409)
        user_id = secrets.token_hex(8)
        self.store.sql_exec("INSERT INTO users (id, email, name, password_hash, company_id, created_at, disabled, role) "
                            "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                            (user_id, email, (name or "").strip() or None, hash_password(password), company_row["id"], _iso(_now()), role))
        return user_id

    # -- the admin page -----------------------------------------------------------------------------

    def has_users(self) -> bool:
        return self.store.exists() and self.store.sql_one("SELECT 1 FROM users LIMIT 1") is not None

    def _active_admins(self) -> int:
        return self.store.sql_one("SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND disabled = 0")["n"]

    def update_user(self, email: str, *, name: str | None = None, company: str | None = None, role: str | None = None,
                    disabled: bool | None = None) -> None:
        """Change an account. The last active administrator cannot be demoted or disabled: that would lock everyone out
        of the admin page."""
        email = (email or "").strip().lower()
        row = self.store.sql_one("SELECT * FROM users WHERE email = ?", (email,))
        if row is None:
            raise AuthError("no_user", f"No account for {email}.", 404)
        if role is not None and role not in ROLES:
            raise AuthError("bad_role", f"A role is one of: {', '.join(ROLES)}.")
        loses_admin = row["role"] == "admin" and not row["disabled"] and (role == "member" or disabled is True)
        if loses_admin and self._active_admins() <= 1:
            raise AuthError("last_admin", "That is the only active administrator. Make another one first.", 409)
        new_role = role or row["role"]
        company_id = self._company(company)["id"] if company is not None else row["company_id"]
        if new_role == "admin":
            company_id = OPERATOR_ID  # an administrator always belongs to the hub's own team
        elif company_id == OPERATOR_ID:
            raise AuthError("operator_company", f"A member must belong to a customer company, not {OPERATOR_NAME}. Choose their company.")
        self.store.sql_exec(
            "UPDATE users SET name = ?, company_id = ?, role = ?, disabled = ? WHERE id = ?",
            ((name.strip() or None) if name is not None else row["name"], company_id, role or row["role"],
             int(disabled) if disabled is not None else row["disabled"], row["id"]))
        if disabled or role == "member" or company_id != row["company_id"]:
            self.store.sql_exec("DELETE FROM sessions WHERE user_id = ?", (row["id"],))  # the change applies at once

    def update_company(self, company: str, *, name: str | None = None, workspaces: dict[str, list[str]] | None = None) -> None:
        row = self._company(company)
        if row["id"] == OPERATOR_ID and ((name is not None and name.strip() != row["name"]) or any(self._clean_workspaces(workspaces).values())):
            self._refuse_operator()
        if name is not None and (_slug(name) == OPERATOR_ID or name.strip().lower() == OPERATOR_NAME.lower()) and row["id"] != OPERATOR_ID:
            raise AuthError("reserved_name", f"\"{OPERATOR_NAME}\" is reserved for the hub's administrators. Choose another name.", 409)
        if name is not None:
            name = name.strip()
            if not name:
                raise AuthError("bad_company", "A company needs a name.")
            if self.store.sql_one("SELECT 1 FROM companies WHERE name = ? AND id != ?", (name, row["id"])):
                raise AuthError("company_exists", f"There is already a company called {name!r}.", 409)
            self.store.sql_exec("UPDATE companies SET name = ? WHERE id = ?", (name, row["id"]))
        if workspaces is not None:
            current = json.loads(row["workspaces"])
            current.update(self._clean_workspaces(workspaces))
            self.store.sql_exec("UPDATE companies SET workspaces = ? WHERE id = ?", (json.dumps(current), row["id"]))

    def delete_company(self, company: str) -> None:
        row = self._company(company)
        if row["id"] == OPERATOR_ID:
            self._refuse_operator()
        if self.store.sql_one("SELECT 1 FROM users WHERE company_id = ?", (row["id"],)):
            raise AuthError("company_in_use", "Move or disable its people first: this company still has accounts.", 409)
        self.store.sql_exec("DELETE FROM companies WHERE id = ?", (row["id"],))

    def create_first_admin(self, email: str, password: str, name: str | None = None) -> str:
        """First-run setup: the first administrator (in the hub's own team), only while there are no accounts at all.
        Customer companies are added afterwards on the admin page."""
        if self.has_users():
            raise AuthError("already_set_up", "Accounts already exist. Ask an administrator.", 409)
        self._check_password(password)
        if not EMAIL.match((email or "").strip().lower()):
            raise AuthError("bad_email", "That does not look like an email address.")
        user_id = self.create_user(email, password, OPERATOR_ID, name, "admin")
        self.audit(email, "setup", "created the first administrator")
        return user_id

    def audit(self, actor: str, action: str, detail: str) -> None:
        self.store.sql_exec("INSERT INTO audit VALUES (?, ?, ?, ?)", (_iso(_now()), actor, action, detail))

    def list_audit(self, limit: int = 50) -> list[dict[str, str]]:
        if not self.store.exists():
            return []
        rows = self.store.sql_all("SELECT * FROM audit ORDER BY at DESC, rowid DESC LIMIT ?", (max(1, min(limit, 500)),))
        return [{"at": r["at"], "actor": r["actor"], "action": r["action"], "detail": r["detail"]} for r in rows]

    @staticmethod
    def _check_password(password: str) -> None:
        if len(password or "") < MIN_PASSWORD_CHARS:
            raise AuthError("weak_password", f"Use a password of at least {MIN_PASSWORD_CHARS} characters.")

    def set_password(self, email: str, password: str) -> None:
        self._check_password(password)
        user = self.store.sql_one("SELECT id FROM users WHERE email = ?", ((email or "").strip().lower(),))
        if user is None:
            raise AuthError("no_user", f"No account for {email}.", 404)
        self.store.sql_exec("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(password), user["id"]))
        self.store.sql_exec("DELETE FROM sessions WHERE user_id = ?", (user["id"],))  # a new password ends every session

    def disable_user(self, email: str, disabled: bool = True) -> None:
        changed = self.store.sql_exec("UPDATE users SET disabled = ? WHERE email = ?", (int(disabled), (email or "").strip().lower()))
        if not changed:
            raise AuthError("no_user", f"No account for {email}.", 404)
        if disabled:
            self.store.sql_exec("DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE email = ?)",
                                ((email or "").strip().lower(),))

    def list_users(self) -> list[dict[str, Any]]:
        if not self.store.exists():
            return []
        rows = self.store.sql_all(
            "SELECT u.email, u.name, u.disabled, u.role, c.name AS company, c.id AS company_id FROM users u "
            "JOIN companies c ON c.id = u.company_id ORDER BY u.email")
        return [{"email": r["email"], "name": r["name"], "company": r["company"], "company_id": r["company_id"],
                 "role": r["role"], "disabled": bool(r["disabled"])} for r in rows]

    # -- signing in -------------------------------------------------------------------------------

    def _failures(self, key: str) -> int:
        cutoff = _iso(_now() - FAILURE_WINDOW)
        self.store.sql_exec("DELETE FROM login_failures WHERE at < ?", (cutoff,))
        return self.store.sql_one("SELECT COUNT(*) AS n FROM login_failures WHERE key = ?", (key,))["n"]

    def lifetime(self, role: str) -> timedelta:
        return self.admin_lifetime if role == "admin" else timedelta(days=self.session_days)

    def session_seconds(self, user: User | None) -> int:
        """How long the browser should keep the cookie (the server enforces the real limits)."""
        return int(self.lifetime(user.role if user else "member").total_seconds())

    def login(self, email: str, password: str, address: str = "", user_agent: str = "") -> tuple[str, User]:
        global _DUMMY_HASH
        email = (email or "").strip().lower()
        key = f"{email}|{address}"
        where = f"from {address or 'an unknown address'}, {describe_browser(user_agent)}"
        if self._failures(key) >= MAX_FAILURES:
            self.audit(email[:200] or "unknown", "auth.locked", f"sign-in refused while locked out, {where}")
            raise AuthError("locked", "Too many failed sign-ins. Wait a few minutes and try again.", 429,
                            retry_after=int(FAILURE_WINDOW.total_seconds()))
        row = self.store.sql_one("SELECT * FROM users WHERE email = ?", (email,)) if self.store.exists() else None
        if _DUMMY_HASH is None:
            _DUMMY_HASH = hash_password("not-a-real-password")
        good = verify_password(password or "", row["password_hash"] if row else _DUMMY_HASH)
        if not (row and good) or row["disabled"]:
            self.store.sql_exec("INSERT INTO login_failures VALUES (?, ?)", (key, _iso(_now())))
            self.audit(email[:200] or "unknown", "auth.signin_failed", where)
            if self._failures(key) >= MAX_FAILURES:
                self.audit(email[:200] or "unknown", "auth.locked", f"{MAX_FAILURES} failed sign-ins: locked for "
                                                                    f"{int(FAILURE_WINDOW.total_seconds() // 60)} minutes, {where}")
            raise AuthError("bad_login", "That email and password do not match an account.", 401)
        self.store.sql_exec("DELETE FROM login_failures WHERE key = ?", (key,))
        self.cleanup_sessions()
        token = secrets.token_urlsafe(32)
        now = _now()
        self.store.sql_exec(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at, last_seen, address, user_agent) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_token_hash(token), row["id"], _iso(now), _iso(now + self.lifetime(row["role"])), _iso(now), address[:64],
             (user_agent or "")[:300]))
        self.audit(email, "auth.signin", where)
        return token, self._user(row["id"])  # type: ignore[return-value]

    def _user(self, user_id: str) -> User | None:
        row = self.store.sql_one(
            "SELECT u.*, c.name AS company_name, c.workspaces AS company_workspaces FROM users u "
            "JOIN companies c ON c.id = u.company_id WHERE u.id = ? AND u.disabled = 0", (user_id,))
        if row is None:
            return None
        workspaces = {a: tuple(ids) for a, ids in json.loads(row["company_workspaces"]).items()}
        return User(row["id"], row["email"], row["name"], row["company_id"], row["company_name"], workspaces, row["role"])

    def user_for_token(self, token: str | None, touch: bool = True) -> User | None:
        """The person a session belongs to, or None when it has ended (lifetime reached, or unused for too long).
        `touch` counts this request as use; background polling passes False, so an open tab alone doesn't keep a
        session alive forever."""
        return self.check_token(token, touch)[0]

    def check_token(self, token: str | None, touch: bool = True) -> tuple[User | None, str | None]:
        """The person, or None and why: "none" (no session sent), "idle" (unused too long), "expired" (past its
        lifetime) or "ended" (signed out, ended from another device or by an administrator, or a password change)."""
        if not token or not self.store.exists():
            return None, "none"
        hashed = _token_hash(token)
        row = self.store.sql_one("SELECT s.user_id, s.expires_at, s.last_seen, u.role FROM sessions s JOIN users u ON u.id = s.user_id "
                                 "WHERE s.token_hash = ?", (hashed,))
        if row is None:
            return None, "ended"
        now = _now()
        idle = self.admin_idle if row["role"] == "admin" else self.idle
        last = row["last_seen"]
        if row["expires_at"] < _iso(now) or (last and last < _iso(now - idle)):
            self.store.sql_exec("DELETE FROM sessions WHERE token_hash = ?", (hashed,))
            return None, ("expired" if row["expires_at"] < _iso(now) else "idle")
        if touch and (not last or last < _iso(now - timedelta(seconds=60))):  # at most one write a minute per session
            self.store.sql_exec("UPDATE sessions SET last_seen = ? WHERE token_hash = ?", (_iso(now), hashed))
        self._lookups += 1
        if self._lookups % 500 == 0:
            self.cleanup_sessions()
        user = self._user(row["user_id"])
        return user, (None if user else "ended")

    def cleanup_sessions(self) -> int:
        """Remove sessions past their lifetime or idle limit (otherwise they'd only go when presented again)."""
        if not self.store.exists():
            return 0
        now = _now()
        removed = self.store.sql_exec("DELETE FROM sessions WHERE expires_at < ?", (_iso(now),))
        removed += self.store.sql_exec(
            "DELETE FROM sessions WHERE last_seen < ? AND user_id IN (SELECT id FROM users WHERE role != 'admin')", (_iso(now - self.idle),))
        removed += self.store.sql_exec(
            "DELETE FROM sessions WHERE last_seen < ? AND user_id IN (SELECT id FROM users WHERE role = 'admin')", (_iso(now - self.admin_idle),))
        return removed

    def list_sessions(self, user_id: str, current_token: str | None = None) -> list[dict[str, Any]]:
        """A person's signed-in sessions, most recently used first. `id` is a short prefix of the token's hash: it
        names a session without revealing anything that could be used to sign in."""
        current = _token_hash(current_token) if current_token else None
        rows = self.store.sql_all("SELECT * FROM sessions WHERE user_id = ? ORDER BY last_seen DESC", (user_id,))
        return [{"id": r["token_hash"][:16], "browser": describe_browser(r["user_agent"]), "address": r["address"] or "",
                 "signed_in": r["created_at"], "last_seen": r["last_seen"] or r["created_at"], "expires": r["expires_at"],
                 "current": r["token_hash"] == current} for r in rows]

    def end_session(self, user_id: str, session_id: str) -> int:
        if not re.fullmatch(r"[0-9a-f]{16}", session_id or ""):
            return 0
        return self.store.sql_exec("DELETE FROM sessions WHERE user_id = ? AND substr(token_hash, 1, 16) = ?", (user_id, session_id))

    def end_other_sessions(self, user_id: str, current_token: str | None) -> int:
        keep = _token_hash(current_token) if current_token else ""
        return self.store.sql_exec("DELETE FROM sessions WHERE user_id = ? AND token_hash != ?", (user_id, keep))

    def end_all_sessions(self, user_id: str) -> int:
        return self.store.sql_exec("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def check_password(self, user_id: str, password: str) -> bool:
        """Re-entering the password before something that can't be undone."""
        row = self.store.sql_one("SELECT password_hash FROM users WHERE id = ? AND disabled = 0", (user_id,))
        return bool(row) and verify_password(password or "", row["password_hash"])

    def user_id_for_email(self, email: str) -> str | None:
        row = self.store.sql_one("SELECT id FROM users WHERE email = ?", ((email or "").strip().lower(),)) if self.store.exists() else None
        return row["id"] if row else None

    def user_by_id(self, user_id: str | None) -> User | None:
        return self._user(user_id) if user_id and self.store.exists() else None

    def logout(self, token: str | None) -> None:
        if token and self.store.exists():
            user = self.user_for_token(token, touch=False)
            self.store.sql_exec("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
            if user is not None:
                self.audit(user.email, "auth.signout", "signed out")
