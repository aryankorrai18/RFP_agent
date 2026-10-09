"""scripts/reset_dev.py: the plan, the dry run, the refusals, and the file operations, on a fake project tree.
The app-side steps (removing workspaces and seeding through each app's own code) are rehearsed on a copy, not here."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reset_dev.py"
spec = importlib.util.spec_from_file_location("reset_dev", SCRIPT)
reset = importlib.util.module_from_spec(spec)
sys.modules["reset_dev"] = reset  # dataclasses look the module up by name
spec.loader.exec_module(reset)


def make_db(path: Path, **tables: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    for name, rows in tables.items():
        con.execute(f"create table {name} (id integer primary key, v text)")
        con.executemany(f"insert into {name} (v) values (?)", [("x",)] * rows)
    con.commit()
    con.close()


def registry(project: Path, active: str, spaces: list[dict]) -> None:
    (project / "data").mkdir(parents=True, exist_ok=True)
    (project / "data" / "workspaces.json").write_text(json.dumps({"active": active, "workspaces": spaces}), encoding="utf-8")


def w(id_: str, kind: str, db: str, uploads: str) -> dict:
    return {"id": id_, "name": id_.title(), "kind": kind, "db_path": db, "uploads_dir": uploads, "bank": f"b-{id_}", "lessons_bank": f"l-{id_}"}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    rfp, deal, hub = tmp_path / "rfp-v0", tmp_path / "deal-intelligence", tmp_path / "agent-hub"
    registry(rfp, "virtusa-demo", [w("main", "main", "data/rfp.db", "data/uploads"), w("virtusa", "company", "data/workspaces/virtusa/rfp.db", "data/workspaces/virtusa/uploads"),
                                   w("virtusa-demo", "company", "data/workspaces/virtusa-demo/rfp.db", "data/workspaces/virtusa-demo/uploads")])
    make_db(rfp / "data" / "rfp.db", answers=5)
    (rfp / "data" / "uploads").mkdir()
    (rfp / "data" / "uploads" / "a.docx").write_bytes(b"x")
    (rfp / "data" / "fact_sheet.json").write_text("{}", encoding="utf-8")
    make_db(rfp / "data" / "workspaces" / "virtusa" / "rfp.db", answers=3)
    make_db(rfp / "data" / "workspaces" / "virtusa-demo" / "rfp.db", answers=2)
    make_db(rfp / "data" / "workspaces" / "virtusa-2" / "rfp.db")  # empty orphan
    make_db(rfp / "data" / "workspaces" / "stray" / "rfp.db", answers=1)  # orphan with data
    registry(deal, "main", [w("main", "main", "data/deals.db", "data/uploads"), w("halcyon", "demo", "data/workspaces/halcyon/deals.db", "data/workspaces/halcyon/uploads")])
    make_db(deal / "data" / "deals.db")  # the user's own, empty
    make_db(deal / "data" / "workspaces" / "halcyon" / "deals.db", deals=4)
    make_db(hub / "data" / "hub.db", conversations=2, events=5, jobs=1, uploads=1, users=2, companies=2, audit=3)
    (hub / "data" / "uploads").mkdir()
    (hub / "data" / "uploads" / "t.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".run").mkdir()
    (tmp_path / ".run" / "hub.out.log").write_text("log", encoding="utf-8")
    (tmp_path / ".run" / "pids.json").write_text("{}", encoding="utf-8")
    (rfp / "evaluation" / "results").mkdir(parents=True)
    (rfp / "evaluation" / "results" / "REPORT.md").write_text("evidence", encoding="utf-8")
    (tmp_path / "backups" / "2026").mkdir(parents=True)
    return tmp_path


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def keys(actions) -> list[str]:  # noqa: ANN001
    return [a.key for a in actions]


def test_the_plan_lists_exactly_what_the_owner_allowed(tree):
    k = keys(reset.plan(tree, demo=False))
    assert k[0] == "backup"
    assert "remove_workspace:rfp:virtusa-demo" in k and "remove_workspace:deal:halcyon" in k and "clear_rfp_main" in k
    assert "clear_hub" in k and "clear_logs" in k
    assert not any(x.endswith(":virtusa") or x == "remove_workspace:rfp:main" for x in k)  # a company workspace and main are never removed
    assert any(x.startswith("archive_orphan:") and x.endswith("virtusa-2") for x in k)
    assert any(x.startswith("report_orphan:") and x.endswith("stray") for x in k) and not any(x.startswith("archive_orphan:") and x.endswith("stray") for x in k)
    assert "seed_rfp_main" not in k and "create_deal_demo" not in k


def test_demo_mode_adds_the_two_seed_steps_and_empty_mode_does_not(tree):
    assert {"seed_rfp_main", "create_deal_demo"} <= set(keys(reset.plan(tree, demo=True)))
    assert not {"seed_rfp_main", "create_deal_demo"} & set(keys(reset.plan(tree, demo=False)))


def test_a_dry_run_changes_nothing(tree, capsys):
    before = snapshot(tree)
    assert reset.main(["--root", str(tree)]) == 0
    assert snapshot(tree) == before and "Dry run" in capsys.readouterr().out


def test_it_refuses_while_the_apps_run_and_changes_nothing(tree, monkeypatch, capsys):
    monkeypatch.setattr(reset, "apps_running", lambda: ["rfp (port 8001)"])
    monkeypatch.setattr(reset, "ROOT", tree)  # the guard protects the real project root only
    before = snapshot(tree)
    assert reset.main(["--root", str(tree), "--yes"]) == 2
    assert snapshot(tree) == before and "Refusing" in capsys.readouterr().out


def test_clearing_the_hub_keeps_accounts_and_the_audit_log(tree):
    reset.clear_hub(tree)
    con = sqlite3.connect(tree / "agent-hub" / "data" / "hub.db")
    left = {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in ("conversations", "events", "jobs", "uploads", "users", "companies", "audit")}
    con.close()
    assert left == {"conversations": 0, "events": 0, "jobs": 0, "uploads": 0, "users": 2, "companies": 2, "audit": 3}
    assert not list((tree / "agent-hub" / "data" / "uploads").iterdir())


def test_emptying_rfp_main_keeps_its_fact_sheet_and_registry_entry(tree):
    reset.clear_rfp_main(tree)
    data = tree / "rfp-v0" / "data"
    assert not (data / "rfp.db").exists() and not list((data / "uploads").iterdir())
    assert (data / "fact_sheet.json").exists() and "main" in (data / "workspaces.json").read_text(encoding="utf-8")


def test_an_orphan_is_moved_aside_not_deleted(tree):
    folder = tree / "rfp-v0" / "data" / "workspaces" / "virtusa-2"
    target = reset.archive_orphan(folder, "20260101T000000Z")
    assert not folder.exists() and (target / "rfp.db").exists() and "_archive" in str(target)


def test_evaluation_results_backups_and_other_workspaces_are_never_in_the_plan(tree):
    text = " ".join(a.text for a in reset.plan(tree, demo=True))
    assert "evaluation" not in text and "backups" not in text.replace("verified backup of all three data folders (backup.ps1", "") and "'virtusa'" not in text
    reset.clear_hub(tree)
    reset.clear_rfp_main(tree)
    assert (tree / "rfp-v0" / "evaluation" / "results" / "REPORT.md").read_text(encoding="utf-8") == "evidence"
    assert (tree / "rfp-v0" / "data" / "workspaces" / "virtusa" / "rfp.db").exists() and (tree / "backups" / "2026").is_dir()


def test_an_already_clean_tree_plans_only_the_backup(tmp_path):
    registry(tmp_path / "rfp-v0", "main", [w("main", "main", "data/rfp.db", "data/uploads")])
    registry(tmp_path / "deal-intelligence", "main", [w("main", "main", "data/deals.db", "data/uploads")])
    assert keys(reset.plan(tmp_path, demo=False)) == ["backup"]
