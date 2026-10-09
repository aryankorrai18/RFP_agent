"""What a workspace has used: every model call (counted, with its tokens) and the space its data takes on disk.

`MeteredLLM` wraps the model for everything that goes through `ctx.llm`; it records the call and returns the result untouched.
Recording can never break a call: if the record cannot be written, the call still succeeds. Counting starts when the
`model_usage` table was added, so calls made before that are not in it."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import case, func, select

from .db import Database, ModelUsage

log = logging.getLogger(__name__)


class MeteredLLM:
    def __init__(self, inner: Any, db: Database) -> None:
        self._inner = inner
        self._db = db

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not inspect.iscoroutinefunction(attr):
            return attr

        async def metered(*args: Any, **kwargs: Any) -> Any:
            try:
                result = await attr(*args, **kwargs)
            except Exception:  # a failed call is counted too (an LLMError or anything unexpected)
                self._record(name, getattr(self._inner, "model", None), 0, 0, ok=False)
                raise
            usage = getattr(result, "usage", None)
            self._record(name, getattr(result, "model", None) or getattr(self._inner, "model", None),
                         int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0), ok=True)
            return result

        return metered

    def _record(self, purpose: str, model: str | None, tokens_in: int, tokens_out: int, *, ok: bool) -> None:
        try:
            with self._db.session() as session:
                session.add(ModelUsage(purpose=purpose[:60], model=(model or "")[:100] or None, input_tokens=tokens_in,
                                       output_tokens=tokens_out, ok=ok))
                session.commit()
        except Exception:  # noqa: BLE001 - metering must never break the call it is measuring
            log.exception("could not record model usage")


def summary(db: Database) -> dict[str, Any]:
    """Calls and tokens, in total and by what was called."""
    failed = func.coalesce(func.sum(case((ModelUsage.ok, 0), else_=1)), 0)
    with db.session() as session:
        rows = session.execute(select(ModelUsage.purpose, func.count(), failed, func.coalesce(func.sum(ModelUsage.input_tokens), 0),
                                      func.coalesce(func.sum(ModelUsage.output_tokens), 0)).group_by(ModelUsage.purpose)).all()
        first, last = session.execute(select(func.min(ModelUsage.at), func.max(ModelUsage.at))).one()
    by_purpose = {p: {"calls": int(n), "failed": int(f), "input_tokens": int(i), "output_tokens": int(o)} for p, n, f, i, o in rows}
    return {"calls": sum(v["calls"] for v in by_purpose.values()), "failed": sum(v["failed"] for v in by_purpose.values()),
            "input_tokens": sum(v["input_tokens"] for v in by_purpose.values()),
            "output_tokens": sum(v["output_tokens"] for v in by_purpose.values()),
            "by_purpose": by_purpose, "since": first.isoformat() if first else None, "last": last.isoformat() if last else None}


def _size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _folder(path: Path) -> tuple[int, int]:
    total = files = 0
    try:
        for item in path.rglob("*") if path.is_dir() else []:
            if item.is_file():
                total += _size(item)
                files += 1
    except OSError:
        pass
    return total, files


def storage(ctx: Any) -> dict[str, int]:
    """Bytes this workspace's data takes: its database (with the write-ahead files), its uploads and any fact sheet."""
    settings = ctx.settings
    db_path = Path(settings.db_path)
    database = sum(_size(p) for p in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")))
    uploads, files = _folder(Path(settings.uploads_dir))
    sheet = getattr(settings, "fact_sheet_path", None)
    other = _size(Path(sheet)) if sheet else 0
    return {"database_bytes": database, "uploads_bytes": uploads, "uploads_files": files, "other_bytes": other,
            "total_bytes": database + uploads + other}
