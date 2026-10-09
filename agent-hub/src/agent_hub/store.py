"""SQLite persistence for the chat: conversations, their event log, uploaded files and background jobs.

Small and synchronous on purpose: one connection, one lock, a handful of rows per conversation. The
database file is only created when something first writes to it, so importing the hub or running
the old stateless chat leaves no file behind."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.environ.get("HUB_DATA_DIR") or Path(__file__).resolve().parent.parent.parent / "data")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".txt", ".md", ".eml", ".csv", ".json")

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    conversation_id TEXT NOT NULL, n INTEGER NOT NULL, role TEXT NOT NULL, kind TEXT NOT NULL,
    payload TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (conversation_id, n));
CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, filename TEXT NOT NULL, size INTEGER NOT NULL,
    sha256 TEXT NOT NULL, path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS companies (
    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE, workspaces TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE COLLATE NOCASE, name TEXT, password_hash TEXT NOT NULL,
    company_id TEXT NOT NULL, created_at TEXT NOT NULL, disabled INTEGER NOT NULL DEFAULT 0,
    role TEXT NOT NULL DEFAULT 'member');
CREATE TABLE IF NOT EXISTS audit (at TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS login_failures (key TEXT NOT NULL, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model_usage (
    at TEXT NOT NULL, company_id TEXT, user_id TEXT, purpose TEXT NOT NULL, model TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0, ok INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS jobs (
    conversation_id TEXT NOT NULL, app TEXT NOT NULL, upstream_job_id TEXT NOT NULL, step TEXT NOT NULL,
    status TEXT NOT NULL, PRIMARY KEY (conversation_id, app, upstream_job_id));
"""


class UploadRejected(Exception):
    """A file the hub will not take. `message` is written for the person in the chat."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


@dataclass(frozen=True)
class Upload:
    id: str
    filename: str
    size: int
    sha256: str
    path: Path

    @property
    def ext(self) -> str:
        return Path(self.filename).suffix.lower()

    def read(self) -> bytes:
        return self.path.read_bytes()


def _first_words(payload: dict[str, Any], limit: int = 60) -> str:
    """A short title or preview from an event: its text, else a card's title, else the first file's name."""
    text = " ".join(str(payload.get("text") or "").split())
    if not text and payload.get("cards"):
        text = " ".join(str(payload["cards"][0].get("title") or "").split())
    if not text and payload.get("files"):
        text = str(payload["files"][0].get("name") or "")
    text = text or "New chat"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def check_upload(filename: str, size: int) -> None:
    """Raise UploadRejected unless the file type and size are acceptable."""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        shown = ext or "that kind of"
        raise UploadRejected(
            "unsupported_type",
            f"I can't read {shown} files. I take {', '.join(ALLOWED_EXTENSIONS)}.",
        )
    if size <= 0:
        raise UploadRejected("empty_file", f"{filename} is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise UploadRejected("too_large", f"{filename} is over the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")


class Store:
    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.db_path = self.data_dir / "hub.db"
        self.uploads_dir = self.data_dir / "uploads"
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # -- connection -------------------------------------------------------------------------------

    def exists(self) -> bool:
        return self._conn is not None or self.db_path.exists()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)
            columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(conversations)")}
            if "user_id" not in columns:  # chats made before sign-in existed belong to nobody
                self._conn.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
            if "role" not in {row["name"] for row in self._conn.execute("PRAGMA table_info(users)")}:
                self._conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'member'")
            if "last_activity" not in columns:  # chat history: titles, ordering and deleting
                for column in ("title", "first_message", "last_activity", "deleted_at"):
                    self._conn.execute(f"ALTER TABLE conversations ADD COLUMN {column} TEXT")
                self._backfill_history(self._conn)
            self._conn.commit()
        return self._conn

    def sql_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db().execute(sql, params).fetchall()

    def sql_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._db().execute(sql, params).fetchone()

    def sql_exec(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            db = self._db()
            cursor = db.execute(sql, params)
            db.commit()
            return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- conversations ----------------------------------------------------------------------------

    @staticmethod
    def _backfill_history(db: sqlite3.Connection) -> None:
        """Chats made before the history list: their last activity and first message, from their events."""
        for row in db.execute("SELECT id, created_at FROM conversations").fetchall():
            last = db.execute("SELECT MAX(created_at) FROM events WHERE conversation_id = ?", (row["id"],)).fetchone()[0]
            first = db.execute("SELECT payload FROM events WHERE conversation_id = ? AND role = 'user' ORDER BY n LIMIT 1",
                               (row["id"],)).fetchone()
            db.execute("UPDATE conversations SET last_activity = ?, first_message = ? WHERE id = ?",
                       (last or row["created_at"], _first_words(json.loads(first["payload"])) if first else None, row["id"]))

    def create_conversation(self, user_id: str | None = None) -> str:
        conv_id = str(uuid.uuid4())
        with self._lock:
            db = self._db()
            now = _now()
            db.execute("INSERT INTO conversations (id, created_at, state, user_id, last_activity) VALUES (?, ?, ?, ?, ?)",
                       (conv_id, now, "{}", user_id, now))
            db.commit()
        return conv_id

    def conversation_owner(self, conv_id: str) -> str | None:
        """The user a chat belongs to (None for a chat made while sign-in was off)."""
        row = self.sql_one("SELECT user_id FROM conversations WHERE id = ?", (conv_id,)) if self.exists() else None
        return row["user_id"] if row else None

    def record_usage(self, company_id: str | None, user_id: str | None, purpose: str, model: str | None,
                     tokens_in: int, tokens_out: int, ok: bool) -> None:
        """One model call the hub itself made (reading a message). Never raises: counting must not break the chat."""
        try:
            self.sql_exec("INSERT INTO model_usage (at, company_id, user_id, purpose, model, input_tokens, output_tokens, ok) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (datetime.now(timezone.utc).isoformat(timespec="seconds"), company_id, user_id, purpose, model,
                           int(tokens_in or 0), int(tokens_out or 0), 1 if ok else 0))
        except Exception:  # noqa: BLE001
            pass

    def get_setting(self, key: str) -> Any | None:
        if not self.exists():
            return None
        with self._lock:
            row = self._db().execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def set_setting(self, key: str, value: Any | None) -> None:
        with self._lock:
            db = self._db()
            if value is None:
                db.execute("DELETE FROM settings WHERE key = ?", (key,))
            else:
                db.execute("INSERT INTO settings VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                           (key, json.dumps(value)))
            db.commit()

    def conversation_exists(self, conv_id: str) -> bool:
        if not self.exists():
            return False
        with self._lock:
            row = self._db().execute("SELECT 1 FROM conversations WHERE id = ? AND deleted_at IS NULL", (conv_id,)).fetchone()
        return row is not None

    # -- history ----------------------------------------------------------------------------------

    def list_conversations(self, user_id: str | None, *, limit: int = 30, before: str | None = None,
                           query: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
        """A person's chats, latest activity first, with a title and a short preview. Chats where they never wrote
        anything (only the welcome) are left out. Chats made with sign-in off (no owner) belong to the open hub."""
        where = ["deleted_at IS NULL", "first_message IS NOT NULL", "user_id IS ?" if user_id is None else "user_id = ?"]
        params: list[Any] = [user_id]
        if before:  # the cursor is "<last_activity>|<id>": exact even when two chats share a timestamp
            when, _, after_id = before.partition("|")
            if after_id:
                where.append("(last_activity < ? OR (last_activity = ? AND id < ?))")
                params += [when, when, after_id]
            else:
                where.append("last_activity < ?")
                params.append(when)
        if query and query.strip():
            where.append("LOWER(COALESCE(title, first_message)) LIKE ?")
            params.append(f"%{query.strip().lower()}%")
        with self._lock:
            db = self._db()
            rows = db.execute(
                f"SELECT id, created_at, title, first_message, last_activity FROM conversations WHERE {' AND '.join(where)} "
                "ORDER BY last_activity DESC, id DESC LIMIT ?", (*params, limit + 1),
            ).fetchall()
            out = []
            for r in rows[:limit]:
                latest = db.execute("SELECT payload FROM events WHERE conversation_id = ? AND kind IN ('text', 'cards') "
                                    "ORDER BY n DESC LIMIT 1", (r["id"],)).fetchone()
                preview = _first_words(json.loads(latest["payload"]), 90) if latest else ""
                out.append({"id": r["id"], "title": r["title"] or r["first_message"], "created_at": r["created_at"],
                            "last_activity": r["last_activity"], "preview": preview or ""})
        return out, (f"{rows[limit - 1]['last_activity']}|{rows[limit - 1]['id']}" if len(rows) > limit else None)

    def conversation_title(self, conv_id: str) -> str | None:
        row = self.sql_one("SELECT title, first_message FROM conversations WHERE id = ?", (conv_id,))
        return (row["title"] or row["first_message"]) if row else None

    def rename_conversation(self, conv_id: str, title: str) -> None:
        with self._lock:
            db = self._db()
            db.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
            db.commit()

    def delete_conversation(self, conv_id: str) -> None:
        """Gone from the person's history and unreachable by its link; the events stay on disk for the audit trail."""
        with self._lock:
            db = self._db()
            db.execute("UPDATE conversations SET deleted_at = ? WHERE id = ?", (_now(), conv_id))
            db.commit()

    def get_state(self, conv_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._db().execute("SELECT state FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        return json.loads(row["state"]) if row else {}

    def set_state(self, conv_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            db = self._db()
            db.execute("UPDATE conversations SET state = ? WHERE id = ?", (json.dumps(state), conv_id))
            db.commit()

    def conversation_ids(self) -> list[str]:
        if not self.exists():
            return []
        with self._lock:
            return [r["id"] for r in self._db().execute("SELECT id FROM conversations ORDER BY created_at")]

    # -- events -----------------------------------------------------------------------------------

    def add_event(self, conv_id: str, role: str, kind: str, **payload: Any) -> dict[str, Any]:
        """Append an event and return it as the API shape. `n` counts up from 1 per conversation."""
        payload = {k: v for k, v in payload.items() if v not in (None, [], {})}
        created = _now()
        with self._lock:
            db = self._db()
            n = (db.execute("SELECT COALESCE(MAX(n), 0) FROM events WHERE conversation_id = ?", (conv_id,)).fetchone()[0]) + 1
            db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)", (conv_id, n, role, kind, json.dumps(payload), created)
            )
            db.execute("UPDATE conversations SET last_activity = ? WHERE id = ?", (created, conv_id))
            if role == "user":
                db.execute("UPDATE conversations SET first_message = ? WHERE id = ? AND first_message IS NULL",
                           (_first_words(payload), conv_id))
            db.commit()
        return {"n": n, "role": role, "kind": kind, **payload, "created_at": created}

    def events(self, conv_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db().execute(
                "SELECT n, role, kind, payload, created_at FROM events WHERE conversation_id = ? AND n > ? ORDER BY n",
                (conv_id, after),
            ).fetchall()
        return [
            {"n": r["n"], "role": r["role"], "kind": r["kind"], **json.loads(r["payload"]), "created_at": r["created_at"]}
            for r in rows
        ]

    def last_n(self, conv_id: str) -> int:
        with self._lock:
            return self._db().execute(
                "SELECT COALESCE(MAX(n), 0) FROM events WHERE conversation_id = ?", (conv_id,)
            ).fetchone()[0]

    # -- uploads ----------------------------------------------------------------------------------

    def save_upload(self, conv_id: str, filename: str, data: bytes) -> Upload:
        """Store the bytes under uploads/<sha256><ext> (identical files share one copy)."""
        check_upload(filename, len(data))
        digest = hashlib.sha256(data).hexdigest()
        ext = Path(filename).suffix.lower()
        path = self.uploads_dir / f"{digest}{ext}"
        upload = Upload(uuid.uuid4().hex[:12], Path(filename).name, len(data), digest, path)
        with self._lock:
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
            db = self._db()
            db.execute(
                "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?)",
                (upload.id, conv_id, upload.filename, upload.size, digest, str(path)),
            )
            db.commit()
        return upload

    def get_upload(self, upload_id: str) -> Upload | None:
        with self._lock:
            r = self._db().execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        return Upload(r["id"], r["filename"], r["size"], r["sha256"], Path(r["path"])) if r else None

    # -- jobs (so a restarted hub can re-attach) --------------------------------------------------

    def add_job(self, conv_id: str, app: str, upstream_job_id: Any, step: str) -> None:
        with self._lock:
            db = self._db()
            db.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, 'running')", (conv_id, app, str(upstream_job_id), step)
            )
            db.commit()

    def finish_job(self, conv_id: str, app: str, upstream_job_id: Any, status: str) -> None:
        with self._lock:
            db = self._db()
            db.execute(
                "UPDATE jobs SET status = ? WHERE conversation_id = ? AND app = ? AND upstream_job_id = ?",
                (status, conv_id, app, str(upstream_job_id)),
            )
            db.commit()

    def running_jobs(self) -> list[dict[str, str]]:
        if not self.exists():
            return []
        with self._lock:
            rows = self._db().execute("SELECT * FROM jobs WHERE status = 'running'").fetchall()
        return [dict(r) for r in rows]
