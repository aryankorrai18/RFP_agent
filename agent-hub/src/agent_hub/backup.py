"""Backups of everything the three apps keep on disk, and a restore that proves it worked.

    python -m agent_hub.backup create            # a dated folder under backups/, then checked
    python -m agent_hub.backup verify <folder>   # checksums and SQLite integrity of an existing backup
    python -m agent_hub.backup restore <folder>  # put it back (the apps must be stopped)
    python -m agent_hub.backup list

What is saved: each app's data folder (databases, uploads, workspace registries, fact sheets). SQLite files are copied with
SQLite's own online backup, so a backup taken while the apps run is still consistent. What is NOT saved: `.env` files (they
hold your keys; keep those in a password manager) and anything in Hindsight Cloud (that is a copy of what is in these
databases and is rebuilt from them).

A backup you have not restored is a hope, not a backup: `verify` re-reads every file against its recorded SHA-256 and runs
`PRAGMA integrity_check` on every database, and `create` runs it for you straight away."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import sqlite3
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
TARGETS = {"hub": REPO / "agent-hub" / "data", "deals": REPO / "deal-intelligence" / "data", "rfp": REPO / "rfp-v0" / "data"}
PORTS = {"rfp": 8001, "deals": 8002, "hub": 8003}
SKIP_DIRS = {"__pycache__"}
SKIP_SUFFIXES = (".tmp", ".lock", "-wal", "-shm", "-journal")
MANIFEST = "manifest.json"
DEFAULT_KEEP = 14


class BackupError(Exception):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sqlite(path: Path) -> bool:
    if path.suffix != ".db":
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def _copy_sqlite(src: Path, dst: Path) -> None:
    """A consistent snapshot of a database that may be in use."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(dst)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def _integrity(path: Path) -> str:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    except sqlite3.DatabaseError as exc:
        return f"not a readable database ({exc})"
    finally:
        conn.close()


def _files(root: Path) -> list[Path]:
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_dir() or any(part in SKIP_DIRS for part in relative.parts) or path.name.endswith(SKIP_SUFFIXES):
            continue
        found.append(path)
    return found


def create(dest_root: Path, targets: dict[str, Path] | None = None, keep: int = DEFAULT_KEEP, now: datetime | None = None) -> Path:
    targets = TARGETS if targets is None else targets
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    folder = dest_root / stamp
    if folder.exists():
        raise BackupError(f"{folder} already exists; wait a second and try again.")
    manifest: dict = {"version": 1, "created": stamp, "targets": {}}
    for name, root in targets.items():
        entries = []
        if root.is_dir():
            for path in _files(root):
                relative = path.relative_to(root)
                out = folder / name / relative
                if _is_sqlite(path):
                    _copy_sqlite(path, out)
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, out)
                entries.append({"path": relative.as_posix(), "sha256": _sha256(out), "size": out.stat().st_size,
                                "sqlite": _is_sqlite(out)})
        manifest["targets"][name] = {"root": str(root), "files": entries}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    problems = verify(folder)
    if problems:
        raise BackupError("The backup was written but did not verify: " + "; ".join(problems[:5]))
    _prune(dest_root, keep)
    return folder


def _prune(dest_root: Path, keep: int) -> None:
    folders = sorted(p for p in dest_root.iterdir() if p.is_dir() and (p / MANIFEST).exists())
    for old in folders[: max(0, len(folders) - max(1, keep))]:
        shutil.rmtree(old, ignore_errors=True)


def _manifest(folder: Path) -> dict:
    try:
        return json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackupError(f"{folder} is not a backup (no readable {MANIFEST}).") from exc


def verify(folder: Path) -> list[str]:
    """Everything wrong with a backup; an empty list means it is whole and its databases are healthy."""
    problems: list[str] = []
    for name, target in _manifest(folder)["targets"].items():
        for entry in target["files"]:
            path = folder / name / entry["path"]
            if not path.exists():
                problems.append(f"{name}/{entry['path']} is missing")
            elif _sha256(path) != entry["sha256"]:
                problems.append(f"{name}/{entry['path']} does not match its checksum")
            elif entry["sqlite"] and (result := _integrity(path)) != "ok":
                problems.append(f"{name}/{entry['path']} failed the database check: {result}")
    return problems


def busy_ports(only: list[str] | None = None) -> list[str]:
    busy = []
    for name, port in PORTS.items():
        if only and name not in only:
            continue
        with socket.socket() as probe:
            probe.settimeout(0.3)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                busy.append(f"{name} (port {port})")
    return busy


def restore(folder: Path, targets: dict[str, Path] | None = None, only: list[str] | None = None, force: bool = False,
            now: datetime | None = None, ports_busy: Callable[[list[str] | None], list[str]] = busy_ports) -> list[Path]:
    """Put a backup back. What was there is moved aside (never deleted), so a wrong restore can be undone."""
    problems = verify(folder)
    if problems:
        raise BackupError("Refusing to restore a backup that does not verify: " + "; ".join(problems[:5]))
    manifest = _manifest(folder)
    names = [n for n in manifest["targets"] if not only or n in only]
    if not names:
        raise BackupError("Nothing to restore: no matching parts in that backup.")
    if not force and (busy := ports_busy(names)):
        raise BackupError("Stop these first (run.ps1 stop), or a running app will overwrite or lock the data: " + ", ".join(busy))
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    aside: list[Path] = []
    for name in names:
        root = (targets or {name: Path(manifest["targets"][name]["root"]) for name in manifest["targets"]})[name]
        if root.exists():
            moved = root.with_name(f"{root.name}.before-restore-{stamp}")
            root.rename(moved)
            aside.append(moved)
        shutil.copytree(folder / name, root) if (folder / name).exists() else root.mkdir(parents=True)
        for entry in manifest["targets"][name]["files"]:
            path = root / entry["path"]
            if _sha256(path) != entry["sha256"]:
                raise BackupError(f"{name}/{entry['path']} did not restore correctly.")
            if entry["sqlite"] and _integrity(path) != "ok":
                raise BackupError(f"{name}/{entry['path']} restored but failed the database check.")
    return aside


def listing(dest_root: Path) -> list[dict]:
    if not dest_root.is_dir():
        return []
    rows = []
    for folder in sorted(p for p in dest_root.iterdir() if p.is_dir() and (p / MANIFEST).exists()):
        manifest = _manifest(folder)
        count = sum(len(t["files"]) for t in manifest["targets"].values())
        size = sum(e["size"] for t in manifest["targets"].values() for e in t["files"])
        rows.append({"name": folder.name, "files": count, "megabytes": round(size / 1048576, 1)})
    return rows


def run(argv: list[str], *, say: Callable[[str], None] = print, targets: dict[str, Path] | None = None,
        dest_root: Path | None = None, ports_busy: Callable[[list[str] | None], list[str]] = busy_ports) -> int:
    parser = argparse.ArgumentParser(prog="agent_hub.backup", description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", help="where backups live (default: backups/ in the repository)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    make = sub.add_parser("create")
    make.add_argument("--keep", type=int, default=DEFAULT_KEEP, help=f"how many backups to keep (default {DEFAULT_KEEP})")
    sub.add_parser("list")
    check = sub.add_parser("verify")
    check.add_argument("folder")
    back = sub.add_parser("restore")
    back.add_argument("folder")
    back.add_argument("--only", help="comma-separated parts: hub, deals, rfp")
    back.add_argument("--force", action="store_true", help="restore even if an app looks like it is running")
    args = parser.parse_args(argv)
    root = Path(args.dest) if args.dest else (dest_root or REPO / "backups")
    try:
        if args.cmd == "create":
            folder = create(root, targets, args.keep)
            manifest = _manifest(folder)
            say(f"Backup written to {folder} and verified: "
                + ", ".join(f"{n} {len(t['files'])} files" for n, t in manifest["targets"].items()) + ".")
            say("Your .env files (keys) are not included; keep those somewhere safe yourself.")
        elif args.cmd == "list":
            for row in listing(root):
                say(f"{row['name']}  {row['files']} files  {row['megabytes']} MB")
        elif args.cmd == "verify":
            problems = verify(Path(args.folder))
            for p in problems:
                say("PROBLEM: " + p)
            say("The backup is whole and its databases are healthy." if not problems else f"{len(problems)} problem(s).")
            return 1 if problems else 0
        else:
            only = [p.strip() for p in args.only.split(",")] if args.only else None
            aside = restore(Path(args.folder), targets, only, args.force, ports_busy=ports_busy)
            say("Restored and verified. The data that was there is kept beside it: " + ", ".join(str(p) for p in aside)
                if aside else "Restored and verified.")
    except BackupError as exc:
        say(f"Error: {exc}")
        return 1
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
