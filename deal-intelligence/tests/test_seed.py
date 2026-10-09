"""The demo dataset and its loader. Offline: no model, no Hindsight."""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import func, select

from deal_intelligence.api.v1 import demo
from deal_intelligence.api.v1.db import (
    Deal, DealSignals, Interaction, MemoryEvent, Play, PlayStats, Stakeholder, deal_code, parse_interaction_code,
)
from deal_intelligence.parsing.parser import parse_document
from tests.builders import make_context

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
TODAY = date(2026, 10, 3)


@pytest.fixture(scope="module")
def seed() -> dict:
    return demo.load_seed()


def closed(seed: dict) -> list[dict]:
    return [d for d in seed["deals"] if d["result"] != "open"]


def opened(seed: dict) -> list[dict]:
    return [d for d in seed["deals"] if d["result"] == "open"]


def by_account(seed: dict, account: str, result: str | None = None) -> list[dict]:
    return [d for d in seed["deals"] if d["account"] == account and (result is None or d["result"] == result)]


def objection(deal: dict, objection_type: str) -> dict | None:
    return next((o for o in (deal["signals"] or {"objections": []})["objections"] if o["type"] == objection_type), None)


def plays(deal: dict) -> set[str]:
    return set(deal["signals"]["plays_used"]) if deal["signals"] else set()


def role(seed: dict, name: str) -> dict:
    return next(d for d in seed["deals"] if d.get("role") == name)


def no_model(ctx):
    calls: list[str] = []

    def refuse(_settings):
        calls.append("llm")
        raise AssertionError("the demo seed must not use the model")

    ctx.llm_provider = refuse
    return calls


def test_seed_validates_and_is_disclosed(seed):
    assert "synthetic" in seed["disclosure"].lower()
    assert seed["vendor"] == "Halcyon Software"
    assert len(seed["plays"]) == 10
    assert {p["code"] for p in seed["plays"]} == {f"PLAY-{n:02d}" for n in range(1, 11)}


def test_counts(seed):
    assert len(closed(seed)) == 19
    assert sum(d["result"] == "won" for d in seed["deals"]) == 9
    assert sum(d["result"] == "lost" for d in seed["deals"]) == 10
    assert len(opened(seed)) == 5
    assert all(d["loss_reason"] for d in seed["deals"] if d["result"] == "lost")
    assert all(2 <= len(d["stakeholders"]) <= 4 for d in closed(seed))
    assert all(3 <= len(d["interactions"]) <= 4 for d in closed(seed))
    assert all(date.fromisoformat("2024-03-01") <= date.fromisoformat(d["closed_on"]) <= date.fromisoformat("2026-08-31")
               for d in closed(seed))


def test_interaction_text_is_substantial(seed):
    for deal in seed["deals"]:
        for item in deal["interactions"]:
            assert 50 <= len(item["text"].split()) <= 140, item["key"]


def test_keys_are_unique(seed):
    keys = [i["key"] for d in seed["deals"] for i in d["interactions"]]
    assert len(keys) == len(set(keys))


def test_validation_names_the_problem(seed, tmp_path):
    def broken(mutate) -> str:
        bad = copy.deepcopy(seed)
        mutate(bad)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError) as caught:
            demo.load_seed(path)
        return str(caught.value)

    assert "segment" in broken(lambda s: s["deals"][0].update(segment="huge"))
    assert "evidence key 'nope'" in broken(lambda s: s["deals"][0]["signals"]["objections"][0]["evidence"].append("nope"))
    assert "unknown play" in broken(lambda s: s["deals"][0]["signals"]["plays_used"].append("PLAY-99"))
    assert "not unique" in broken(lambda s: s["deals"][1]["interactions"][0].update(key=s["deals"][0]["interactions"][0]["key"]))
    assert "loss_reason" in broken(lambda s: s["deals"][2].update(loss_reason="vibes"))
    assert "role 'demo'" in broken(lambda s: next(d for d in s["deals"] if d.get("role") == "demo").pop("role"))
    assert "days_ago" in broken(lambda s: s["deals"][-1]["interactions"][0].pop("days_ago"))


def test_loader_inserts_everything(tmp_path, seed):
    ctx = make_context(tmp_path, llm=None)
    calls = no_model(ctx)
    result = demo.seed_demo(ctx, today=TODAY)

    expected_interactions = sum(len(d["interactions"]) for d in seed["deals"])
    assert result == {
        "plays": 10, "deals": 24, "closed": 19, "open": 5, "interactions": expected_interactions, "already_there": 0,
        "demo_deal": "D-015", "closable_deal": "D-018",
    }
    assert calls == []

    with ctx.db.session() as session:
        assert session.scalar(select(func.count()).select_from(Play)) == 10
        assert session.scalar(select(func.count()).select_from(PlayStats)) == 10
        assert session.scalar(select(func.count()).select_from(Deal)) == 24
        assert session.scalar(select(func.count()).select_from(Interaction)) == expected_interactions
        assert session.scalar(select(func.count()).select_from(Stakeholder)) == sum(len(d["stakeholders"]) for d in seed["deals"])
        assert session.scalar(select(func.count()).select_from(DealSignals)) == 23
        events = session.scalars(select(MemoryEvent)).all()
        assert len(events) == 24 and {e.kind for e in events} == {"deal_added"}
        assert all(row.times_used == 0 for row in session.scalars(select(PlayStats)))

        for item, deal in zip(seed["deals"], session.scalars(select(Deal).order_by(Deal.id)), strict=True):
            assert (deal.account, deal.name) == (item["account"], item["name"])
            assert deal.is_open == (item["result"] == "open")
            assert deal.stage == ("closed" if not deal.is_open else item["stage"])
            assert deal.signals_status == ("none" if item["signals"] is None else "ready")
            assert deal.hindsight_status == ("skipped" if deal.is_open else "pending")
            assert (deal.signals is None) == (item["signals"] is None)
            if deal.signals is not None:
                assert deal.signals.source == "seed"

        tidewater = session.scalars(select(Deal).where(Deal.account == "Tidewater Freight")).one()
        assert tidewater.signals_status == "none" and tidewater.signals is None and len(tidewater.interactions) == 2
        assert len(tidewater.stakeholders) == 1

        # Closed deals keep their interactions out of Hindsight unless the account also has an open deal.
        for deal in session.scalars(select(Deal)):
            statuses = {i.hindsight_status for i in deal.interactions}
            if deal.is_open or deal.account == "Cedarline Bank":
                assert statuses == {"pending"}, deal.name
            else:
                assert statuses == {"skipped"}, deal.name


def test_evidence_resolves_to_interaction_codes(tmp_path):
    ctx = make_context(tmp_path, llm=None)
    demo.seed_demo(ctx, today=TODAY)
    with ctx.db.session() as session:
        for deal in session.scalars(select(Deal)):
            if deal.signals is None:
                continue
            own = {i.code for i in deal.interactions}
            cited = [e for o in deal.signals.objections for e in o["evidence"]] + [
                e for p in deal.signals.promises for e in p["evidence"]]
            assert set(cited) <= own, deal.name
            assert all(parse_interaction_code(code) for code in cited)
        first = session.get(Deal, 1)
        assert [i.code for i in first.interactions] == ["INT-0001", "INT-0002", "INT-0003", "INT-0004"]


def test_reseed_is_idempotent(tmp_path):
    ctx = make_context(tmp_path, llm=None)
    first = demo.seed_demo(ctx, today=TODAY)
    again = demo.seed_demo(ctx, today=TODAY)
    assert again == {
        "plays": 0, "deals": 0, "closed": 0, "open": 0, "interactions": 0, "already_there": 24,
        "demo_deal": first["demo_deal"], "closable_deal": first["closable_deal"],
    }
    with ctx.db.session() as session:
        assert session.scalar(select(func.count()).select_from(Deal)) == 24
        assert session.scalar(select(func.count()).select_from(Interaction)) == first["interactions"]
        assert session.scalar(select(func.count()).select_from(MemoryEvent)) == 24


def test_open_deals_use_relative_dates(tmp_path_factory):
    def snapshot(today: date) -> dict:
        ctx = make_context(tmp_path_factory.mktemp("rel"), llm=None)
        result = demo.seed_demo(ctx, today=today)
        with ctx.db.session() as session:
            deal = session.get(Deal, parse_deal(result["demo_deal"]))
            return {
                "opened": deal.opened_on,
                "interactions": [i.occurred_on for i in deal.interactions],
                "first_seen": [date.fromisoformat(o["first_seen_on"]) for o in deal.signals.objections],
                "promise_due": [date.fromisoformat(p["due_on"]) for p in deal.signals.promises],
                "closed_example": session.get(Deal, 1).closed_on,
            }

    now, later = snapshot(date(2026, 10, 3)), snapshot(date(2026, 12, 1))
    shift = (date(2026, 12, 1) - date(2026, 10, 3)).days
    assert (later["opened"] - now["opened"]).days == shift
    assert all((b - a).days == shift for a, b in zip(now["interactions"], later["interactions"], strict=True))
    assert all((b - a).days == shift for a, b in zip(now["promise_due"], later["promise_due"], strict=True))
    assert all((b - a).days == shift for a, b in zip(now["first_seen"], later["first_seen"], strict=True))
    assert now["closed_example"] == later["closed_example"] == date(2024, 3, 27)
    assert max(now["interactions"]) <= date(2026, 10, 3)
    assert any(due < date(2026, 10, 3) for due in now["promise_due"])  # the whitepaper promise is overdue


def parse_deal(code: str) -> int:
    return int(code.removeprefix("D-"))


def test_closed_demo_ids_are_stable(tmp_path):
    ctx = make_context(tmp_path, llm=None)
    result = demo.seed_demo(ctx, today=TODAY)
    with ctx.db.session() as session:
        assert session.get(Deal, parse_deal(result["demo_deal"])).account == "Cedarline Bank"
        assert session.get(Deal, parse_deal(result["closable_deal"])).account == "Larkfield Credit Union"
    assert deal_code(parse_deal(result["demo_deal"])) == result["demo_deal"]


def test_planted_patterns_hold(seed):
    done = closed(seed)

    fintech_sso_unresolved = [d for d in done if d["industry"] == "fintech" and (objection(d, "sso") or {}).get("status") == "unresolved"]
    assert len(fintech_sso_unresolved) == 3
    assert sum(d["result"] == "lost" for d in fintech_sso_unresolved) == 2

    both_security_plays = [d for d in done if (objection(d, "sso") or {}).get("status") == "addressed"
                           and {"PLAY-01", "PLAY-10"} <= plays(d)]
    assert len(both_security_plays) == 2 and all(d["result"] == "won" for d in both_security_plays)

    price_losses = [d for d in done if d["result"] == "lost" and d["loss_reason"] == "price"]
    assert price_losses and all("PLAY-05" in plays(d) for d in price_losses)

    assert any(d["result"] == "won" and any(o["status"] == "unresolved" for o in d["signals"]["objections"]) for d in done)
    assert any(d["result"] == "lost" and "PLAY-01" in plays(d) for d in done)
    assert all(d["result"] == "lost" for d in done if d["loss_reason"] == "security_compliance")

    cedar_old = by_account(seed, "Cedarline Bank", "lost")
    assert len(cedar_old) == 1 and cedar_old[0]["loss_reason"] == "feature_gap" and cedar_old[0]["segment"] == "smb"
    assert cedar_old[0]["closed_on"].startswith("2025-02")


def test_pricing_story_is_planted_and_stays_clear_of_the_demo_deals(seed):
    new_closed, new_open = seed["deals"][18:23], seed["deals"][23]
    assert [d["result"] for d in new_closed] == ["lost", "lost", "lost", "won", "won"]
    assert sorted(d["loss_reason"] for d in new_closed if d["result"] == "lost") == ["competitor", "no_decision", "price"]
    for deal in new_closed:
        assert deal["industry"] in ("retail", "logistics") and deal["segment"] in ("mid_market", "smb")
        assert [o["type"] for o in deal["signals"]["objections"]] == ["pricing"]
        assert date.fromisoformat("2025-03-01") <= date.fromisoformat(deal["closed_on"]) <= date.fromisoformat("2026-08-31")
    lost, won = new_closed[:3], new_closed[3:]
    assert all(objection(d, "pricing")["status"] == "unresolved" and plays(d) == {"PLAY-05"} for d in lost)
    assert sum("Orbitly" in d["signals"]["competitors"] for d in lost) == 1
    assert all(objection(d, "pricing")["status"] == "addressed" and "PLAY-05" not in plays(d) for d in won)
    assert [plays(d) for d in won] == [{"PLAY-03", "PLAY-02"}, {"PLAY-03", "PLAY-08"}]

    assert (new_open["result"], new_open["industry"], new_open["segment"], new_open["stage"]) == ("open", "retail", "mid_market", "negotiation")
    assert 120000 <= new_open["amount"] <= 140000 and "role" not in new_open and 4 <= len(new_open["interactions"]) <= 6
    assert objection(new_open, "pricing")["status"] == "raised" and new_open["signals"]["pricing"]["discount_requested"]
    assert new_open["signals"]["competitors"] == ["Orbitly"] and new_open["signals"]["plays_used"] == []
    assert any(s["stance"] == "champion" and s["engaged"] for s in new_open["stakeholders"])
    assert any(s["economic_buyer"] and s["engaged"] for s in new_open["stakeholders"])
    assert any("revised quote" in p["text"].lower() and 0 <= p["due_in_days"] <= 7 for p in new_open["signals"]["promises"])
    for deal in new_closed + [new_open]:
        assert not {o["type"] for o in deal["signals"]["objections"]} & {"sso", "security_review"}
        assert "Brightline" not in deal["signals"]["competitors"]


def test_demo_and_closable_deals_overlap(seed):
    demo_deal, closable = role(seed, "demo"), role(seed, "closable")
    for deal in (demo_deal, closable):
        assert deal["industry"] == "fintech" and deal["segment"] == "mid_market"
        assert objection(deal, "sso")["status"] == "unresolved"
        assert not any(s["engaged"] for s in deal["stakeholders"] if s["economic_buyer"])
        assert deal["signals"]["plays_used"] == []
    assert demo_deal["amount"] == 180000 and demo_deal["stage"] == "evaluation"
    assert 6 <= len(demo_deal["interactions"]) <= 8
    assert closable["stage"] == "negotiation" and closable["signals"]["pricing"]["discount_requested"]
    assert len(closable["interactions"]) == 5
    assert "Brightline" in demo_deal["signals"]["competitors"]
    assert any(p["due_in_days"] < 0 and "whitepaper" in p["text"] for p in demo_deal["signals"]["promises"])
    assert any(s["engaged"] and s["stance"] == "champion" for s in demo_deal["stakeholders"])


def test_other_open_deals(seed):
    ashgrove = by_account(seed, "Ashgrove Health Network")[0]
    assert ashgrove["segment"] == "enterprise" and 5 <= len(ashgrove["interactions"]) <= 6
    assert objection(ashgrove, "security_review") and objection(ashgrove, "data_residency")
    assert ashgrove["signals"]["plays_used"] == ["PLAY-01"] and "Brightline" in ashgrove["signals"]["competitors"]
    tidewater = by_account(seed, "Tidewater Freight")[0]
    assert tidewater["signals"] is None and tidewater["stage"] == "discovery" and len(tidewater["interactions"]) == 2


def test_sample_files_parse():
    files = sorted(p for p in SAMPLES.iterdir() if p.suffix in {".txt", ".md"} and p.name != "README.md")
    assert len(files) == 3
    for path in files:
        parsed = parse_document(path.name, path.read_bytes(), 50_000)
        assert 80 <= len(parsed.text.split()) <= 200, path.name
        assert "Tidewater" in parsed.text or "Nadia" in parsed.text
    assert "synthetic" in (SAMPLES / "README.md").read_text(encoding="utf-8").lower()
