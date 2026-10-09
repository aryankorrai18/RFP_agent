"""Workspaces: the same engine with different starting memory.

A workspace is everything one sales team's memory lives in: its own SQLite database, uploads folder
and pair of Hindsight banks. Two ways to start one:

- Demo: seeded with a fictional seller's deal history (closed deals with won/lost results, plus open
  deals to brief), so the learning loop is visible in minutes. No model calls; Hindsight stores the
  seeded interactions and summaries. A fresh demo workspace gets fresh banks, so repeated demo runs
  never contaminate each other.
- From scratch: empty. The team uploads deals and records outcomes, and every closed deal adds memory.

The registry (data/workspaces.json) records the workspaces and which one is active. Without a
registry file (tests, scripts that set DEAL_DB_PATH) settings are left exactly as the environment gives.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .config import ROOT, Settings

REGISTRY_ENV = "DEAL_WORKSPACES_FILE"
KINDS = ("demo", "company", "main")
ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


@dataclass(frozen=True)
class Workspace:
    id: str
    name: str
    kind: str  # main (the environment's own paths and banks) | demo | company
    db_path: str
    uploads_dir: str
    bank: str
    lessons_bank: str
    created_at: str


def registry_file() -> Path:
    raw = os.environ.get(REGISTRY_ENV)
    if not raw:
        return ROOT / "data" / "workspaces.json"
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _abs(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load() -> tuple[str | None, list[Workspace]]:
    try:
        data = json.loads(registry_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None, []
    fields = set(Workspace.__dataclass_fields__)
    spaces = [Workspace(**{k: v for k, v in item.items() if k in fields}) for item in data.get("workspaces", [])]
    return data.get("active"), spaces


def _save(active: str, spaces: list[Workspace]) -> None:
    path = registry_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"active": active, "workspaces": [asdict(w) for w in spaces]}, indent=2), encoding="utf-8")
    tmp.replace(path)


def active() -> Workspace | None:
    active_id, spaces = load()
    return next((w for w in spaces if w.id == active_id), None)


def ensure_registry(settings: Settings) -> Workspace:
    """First run: record the environment's own paths and banks as the workspace "main"."""
    current = active()
    if current is not None:
        return current
    main = Workspace(
        id="main", name="My deals", kind="main",
        db_path=_rel(settings.db_path), uploads_dir=_rel(settings.uploads_dir),
        bank=settings.hindsight_bank, lessons_bank=settings.hindsight_lessons_bank,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    _, spaces = load()
    _save("main", [main, *[w for w in spaces if w.id != "main"]])
    return main


def apply_workspace(settings: Settings, workspace_id: str | None = None) -> Settings:
    """The settings with the workspace's database, uploads and banks. No registry: unchanged."""
    active_id, spaces = load()
    wanted = workspace_id or active_id
    space = next((w for w in spaces if w.id == wanted), None)
    if space is None:
        return settings
    return replace(
        settings, db_path=_abs(space.db_path), uploads_dir=_abs(space.uploads_dir),
        hindsight_bank=space.bank, hindsight_lessons_bank=space.lessons_bank,
    )


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30].strip("-")
    return slug or "workspace"


def create(name: str, kind: str, base: Settings) -> Workspace:
    """A new, empty workspace with its own folder and banks. Doesn't switch to it."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Give the workspace a name.")
    if kind not in ("demo", "company"):
        raise ValueError("kind must be demo or company")
    active_id, spaces = load()
    taken = {w.id for w in spaces}
    stem = _slug(name)
    workspace_id, n = stem, 2
    while workspace_id in taken:
        workspace_id, n = f"{stem}-{n}", n + 1
    folder = registry_file().parent / "workspaces" / workspace_id  # data/workspaces/<id> beside the registry
    space = Workspace(
        id=workspace_id, name=name, kind=kind, db_path=_rel(folder / "deals.db"), uploads_dir=_rel(folder / "uploads"),
        # Hindsight banks are per workspace, so one team's memory never answers for another and a fresh
        # demo workspace never inherits the observations an earlier demo run consolidated.
        bank=f"{base.hindsight_bank}-{workspace_id}", lessons_bank=f"{base.hindsight_lessons_bank}-{workspace_id}",
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    folder.mkdir(parents=True, exist_ok=True)
    _save(active_id or workspace_id, [*spaces, space])
    return space


def activate(workspace_id: str) -> Workspace:
    _, spaces = load()
    space = next((w for w in spaces if w.id == workspace_id), None)
    if space is None:
        raise KeyError(workspace_id)
    _save(workspace_id, spaces)
    return space


def get(workspace_id: str) -> Workspace | None:
    return next((w for w in load()[1] if w.id == workspace_id), None)


def remove(workspace_id: str) -> Workspace:
    """Delete a workspace: its registry entry, and its local files (database, uploads) if they live
    in the standard per-workspace folder this module creates. Its Hindsight banks aren't touched
    here (this module never talks to Hindsight), so the API layer deletes those separately,
    best-effort, after this succeeds.

    Refuses to remove "main" or the active workspace (switch to another one first, so nothing is
    deleted out from under a running context)."""
    active_id, spaces = load()
    space = next((w for w in spaces if w.id == workspace_id), None)
    if space is None:
        raise KeyError(workspace_id)
    if space.kind == "main":
        raise ValueError("The original workspace can't be removed.")
    if workspace_id == active_id:
        raise ValueError("Switch to another workspace before removing this one.")
    _save(active_id, [w for w in spaces if w.id != workspace_id])
    folder = registry_file().parent / "workspaces" / workspace_id
    if folder.is_dir() and folder == _abs(space.db_path).parent:  # only ever the folder create() made
        shutil.rmtree(folder, ignore_errors=True)
    return space
