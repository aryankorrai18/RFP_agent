"""Backups: consistent copies, honest verification, and a restore that keeps what it replaces."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agent_hub import backup


def make_db(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS t (v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?)", [(r,) for r in rows])
    conn.commit()
    conn.close()


def rows(path):
    conn = sqlite3.connect(path)
    try:
        return [r[0] for r in conn.execute("SELECT v FROM t ORDER BY v")]
    finally:
        conn.close()


@pytest.fixture
def world(tmp_path):
    hub, deals = tmp_path / "hub" / "data", tmp_path / "deals" / "data"
    make_db(hub / "hub.db", ["chat-1", "chat-2"])
    (hub / "uploads").mkdir(parents=True)
    (hub / "uploads" / "a.txt").write_text("an uploaded email", encoding="utf-8")
    make_db(deals / "workspaces" / "demo" / "deals.db", ["deal-1"])
    (deals / "workspaces.json").write_text('{"active": "demo"}', encoding="utf-8")
    (deals / "__pycache__").mkdir()
    (deals / "__pycache__" / "x.pyc").write_bytes(b"junk")
    return {"hub": hub, "deals": deals}, tmp_path / "backups"


def test_a_backup_copies_databases_uploads_and_registries_and_verifies(world):
    targets, dest = world
    folder = backup.create(dest, targets)
    assert rows(folder / "hub" / "hub.db") == ["chat-1", "chat-2"] and (folder / "hub" / "uploads" / "a.txt").exists()
    assert (folder / "deals" / "workspaces.json").exists() and backup.verify(folder) == []
    manifest = backup._manifest(folder)
    assert {e["path"] for e in manifest["targets"]["deals"]["files"]} == {"workspaces/demo/deals.db", "workspaces.json"}


def test_cache_files_are_left_out(world):
    targets, dest = world
    folder = backup.create(dest, targets)
    assert "x.pyc" not in {p.name for p in folder.rglob("*") if p.is_file()}


def test_a_backup_of_a_database_in_use_is_still_consistent(world):
    targets, dest = world
    conn = sqlite3.connect(targets["hub"] / "hub.db")
    conn.execute("INSERT INTO t VALUES ('uncommitted')")  # an open, uncommitted write
    folder = backup.create(dest, targets)
    conn.rollback()
    conn.close()
    assert rows(folder / "hub" / "hub.db") == ["chat-1", "chat-2"] and backup.verify(folder) == []


def test_verify_notices_a_changed_or_missing_file_and_a_corrupt_database(world):
    targets, dest = world
    folder = backup.create(dest, targets)
    (folder / "hub" / "uploads" / "a.txt").write_text("tampered", encoding="utf-8")
    assert any("checksum" in p for p in backup.verify(folder))
    (folder / "deals" / "workspaces.json").unlink()
    assert any("missing" in p for p in backup.verify(folder))
    folder2 = backup.create(dest, targets, now=datetime.now(timezone.utc) + timedelta(seconds=5))
    db = folder2 / "hub" / "hub.db"
    data = bytearray(db.read_bytes())
    data[100:140] = b"\xff" * 40
    db.write_bytes(bytes(data))
    assert backup.verify(folder2)


def test_the_integrity_check_itself_reports_a_damaged_database(tmp_path):
    db = tmp_path / "x.db"
    make_db(db, [f"row-{n}" for n in range(500)])
    assert backup._integrity(db) == "ok"
    db.write_bytes(db.read_bytes()[:5000])  # a file cut short mid-way, as a failed copy would leave it
    assert backup._integrity(db) != "ok"


def test_a_folder_that_is_not_a_backup_is_refused(tmp_path):
    with pytest.raises(backup.BackupError):
        backup.verify(tmp_path)


def test_restore_puts_everything_back_and_keeps_what_it_replaced(world):
    targets, dest = world
    folder = backup.create(dest, targets)
    make_db(targets["hub"] / "hub.db", ["added-later"])
    (targets["hub"] / "uploads" / "b.txt").write_text("new", encoding="utf-8")
    aside = backup.restore(folder, targets, ports_busy=lambda _o: [])
    assert rows(targets["hub"] / "hub.db") == ["chat-1", "chat-2"] and not (targets["hub"] / "uploads" / "b.txt").exists()
    kept = next(p for p in aside if (p / "hub.db").exists())
    assert "added-later" in rows(kept / "hub.db") and (kept / "uploads" / "b.txt").exists()  # nothing was destroyed


def test_restore_refuses_while_the_apps_are_running_unless_forced(world):
    targets, dest = world
    folder = backup.create(dest, targets)
    with pytest.raises(backup.BackupError, match="Stop these first"):
        backup.restore(folder, targets, ports_busy=lambda _o: ["hub (port 8003)"])
    assert backup.restore(folder, targets, force=True, ports_busy=lambda _o: ["hub (port 8003)"])


def test_restore_refuses_a_backup_that_does_not_verify(world):
    targets, dest = world
    folder = backup.create(dest, targets)
    (folder / "hub" / "uploads" / "a.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(backup.BackupError, match="does not verify"):
        backup.restore(folder, targets, ports_busy=lambda _o: [])
    assert (targets["hub"] / "uploads" / "a.txt").read_text(encoding="utf-8") == "an uploaded email"  # untouched


def test_restore_can_be_limited_to_one_part(world):
    targets, dest = world
    folder = backup.create(dest, targets)
    make_db(targets["hub"] / "hub.db", ["hub-changed"])
    make_db(targets["deals"] / "workspaces" / "demo" / "deals.db", ["deals-changed"])
    backup.restore(folder, targets, only=["deals"], ports_busy=lambda _o: [])
    assert "hub-changed" in rows(targets["hub"] / "hub.db")
    assert "deals-changed" not in rows(targets["deals"] / "workspaces" / "demo" / "deals.db")


def test_old_backups_are_pruned_keeping_the_newest(world):
    targets, dest = world
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for n in range(5):
        backup.create(dest, targets, keep=3, now=start + timedelta(days=n))
    assert [r["name"] for r in backup.listing(dest)] == ["20260103T000000Z", "20260104T000000Z", "20260105T000000Z"]


def test_a_missing_data_folder_is_recorded_as_empty_not_an_error(tmp_path):
    folder = backup.create(tmp_path / "b", {"hub": tmp_path / "nowhere"})
    assert backup._manifest(folder)["targets"]["hub"]["files"] == []


def test_the_command_line_round_trip(world, tmp_path):
    targets, dest = world
    said: list[str] = []
    assert backup.run(["create", "--keep", "5"], say=said.append, targets=targets, dest_root=dest) == 0
    assert "verified" in said[0] and ".env files" in said[1]
    folder = next(p for p in dest.iterdir() if p.is_dir())
    assert backup.run(["verify", str(folder)], say=said.append) == 0 and "whole" in said[-1]
    assert backup.run(["list"], say=said.append, dest_root=dest) == 0 and folder.name in said[-1]
    make_db(targets["hub"] / "hub.db", ["later"])
    assert backup.run(["restore", str(folder)], say=said.append, targets=targets, ports_busy=lambda _o: []) == 0
    assert rows(targets["hub"] / "hub.db") == ["chat-1", "chat-2"]
    assert backup.run(["verify", str(tmp_path)], say=said.append) == 1 and "Error" in said[-1]
