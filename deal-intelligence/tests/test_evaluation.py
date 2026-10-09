"""The evaluation harness itself: the generated world, the marking, the statistics and the report. Offline and quick."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from deal_intelligence.api.v1 import demo
from evaluation import brief_eval, charts, harness, report, retrieval_eval, stats
from evaluation.world import SITUATIONS, make_world, play_class


def run(coro):
    return asyncio.run(coro)


# -- the world ----------------------------------------------------------------------------------------------

def test_the_same_seed_makes_the_same_world_and_another_seed_a_different_one():
    a, b, c = make_world(5, 20, 10), make_world(5, 20, 10), make_world(6, 20, 10)
    assert a.deals == b.deals and a.deals != c.deals


def test_the_world_has_the_requested_size_and_is_spread_over_every_situation():
    world = make_world(1, 60, 20)
    assert (len(world.train), len(world.test)) == (60, 20) and len(world.deals) == 80
    assert {d.situation for d in world.test} == set(SITUATIONS)
    assert all(sum(1 for d in world.test if d.situation == name) == 4 for name in SITUATIONS)
    assert len({d["account"] for d in world.deals}) == 80  # no two deals share an account


def test_the_generated_world_passes_the_products_own_seed_validation(tmp_path):
    path = make_world(2, 30, 10).write(tmp_path / "world.json")
    seed = demo.load_seed(path)
    assert len(seed["deals"]) == 40 and all(d["signals"] is not None for d in seed["deals"])


def test_the_planted_rules_are_consistent():
    for sit in SITUATIONS.values():
        assert not set(sit.good) & set(sit.trap) and sit.obvious in sit.good + sit.trap
        if sit.kind == "counterintuitive":
            assert sit.obvious in sit.trap  # the catalogue's obvious play is the trap here
        else:
            assert sit.obvious in sit.good
        assert play_class(sit, sit.good[0]) == "good" and play_class(sit, sit.trap[0]) == "trap"


def test_good_plays_win_much_more_often_than_traps_in_the_generated_history():
    wins = {"good": [0, 0], "trap": [0, 0]}
    for seed in range(1, 8):
        for d in make_world(seed, 60, 5).train:
            if d.play_class in wins:
                wins[d.play_class][0] += d.result == "won"
                wins[d.play_class][1] += 1
    rate = {k: v[0] / v[1] for k, v in wins.items()}
    assert rate["good"] > 0.75 and rate["trap"] < 0.25


def test_truncating_keeps_the_test_deals_and_the_first_closed_deals():
    world = make_world(3, 40, 10)
    small = harness.truncate(world, 12)
    assert len(small.train) == 12 and small.test == world.test and small.deals[:12] == world.deals[:12] and len(small.deals) == 22


# -- marking and baselines --------------------------------------------------------------------------------------

def test_marking_follows_the_planted_rules():
    pricing = SITUATIONS["pricing"]
    assert harness.score("pricing", ["PLAY-03", "PLAY-02", "PLAY-09"])["hit1"] is True
    wrong = harness.score("pricing", ["PLAY-05", "PLAY-03"], ["PLAY-03"])
    assert wrong["hit1"] is False and wrong["hit3"] is True and wrong["trap1"] is True and wrong["trap3"] is True
    assert wrong["false_alarm"] is True  # it flagged a good play as one to avoid
    assert harness.score("pricing", [])["empty"] is True and harness.score("pricing", ["PLAY-02"] * 3)["hit3"] is False
    assert harness.score("pricing", ["PLAY-02", "PLAY-08", "PLAY-09", "PLAY-05"])["trap3"] is False  # only the top three count
    assert pricing.trap == ("PLAY-05",)


def test_the_baselines_do_what_their_descriptions_say():
    from evaluation.world import DealTruth

    def deal(sit, result, *plays):
        return DealTruth("a", "a", sit, result, plays[0], None, list(plays))

    train = [deal("pricing", "lost", "PLAY-05"), deal("pricing", "lost", "PLAY-05"), deal("pricing", "won", "PLAY-05"),
             deal("pricing", "won", "PLAY-03"), deal("sso", "won", "PLAY-10")]
    assert harness.naive_rag(train, "pricing")[0] == "PLAY-05"  # most used, whatever happened
    assert harness.neighbour_wins(train, "pricing") == ["PLAY-05", "PLAY-03"] or harness.neighbour_wins(train, "pricing")[0] in ("PLAY-05", "PLAY-03")
    assert harness.neighbour_wins(train, "sso") == ["PLAY-10"] and harness.naive_rag(train, "legal_terms") == []
    assert harness.popularity(train)[0] in ("PLAY-10", "PLAY-03")  # smoothed win rate, not use count


# -- the product is run on the world --------------------------------------------------------------------------

def test_a_world_loads_into_the_product_and_its_retrieval_answers_for_every_test_deal():
    rows = run(retrieval_eval.evaluate_world(1, 24, 5))
    assert len(rows) == 5 * len(retrieval_eval.SYSTEMS)
    assert {r["system"] for r in rows} == set(retrieval_eval.SYSTEMS) and all(isinstance(r["hit3"], bool) for r in rows)
    ours = [r for r in rows if r["system"] == "ours_similar"]
    assert any(r["ranked"] for r in ours)  # the product recommended something


def test_the_ranking_change_can_be_switched_off_to_reproduce_the_old_behaviour():
    from deal_intelligence.api.v1 import ranking

    try:
        retrieval_eval.set_variant("before")
        assert ranking.LABEL_OVERRULED_MIN_USED > 10**6
        retrieval_eval.set_variant("after")
        assert ranking.LABEL_OVERRULED_MIN_USED == 3
    finally:
        retrieval_eval.set_variant("after")


def test_the_brief_evaluation_estimates_its_calls_without_any_model():
    estimate = run(brief_eval.estimate(1, 20, 5, ("none", "similar")))
    assert estimate["calls"] == 10 and estimate["input_tokens"] > 0 and estimate["usd_upper_bound"] > 0


def test_a_brief_is_marked_from_its_recommended_steps_and_avoid_list():
    marks = brief_eval.mark_brief("integration", {"next_steps": [{"play_code": "PLAY-06"}, {"play_code": "PLAY-04"}],
                                                   "avoid": [{"play_code": "PLAY-06"}]})
    assert marks["hit3"] and marks["trap3"] and marks["flagged"] and marks["plays"] == ["PLAY-06", "PLAY-04"] and not marks["hit1"]


def test_resuming_skips_what_is_already_written(tmp_path):
    path = tmp_path / "briefs.jsonl"
    path.write_text(json.dumps({"seed": 1, "deal": "A", "arm": "none", "status": "ready"}) + "\n"
                    + json.dumps({"seed": 1, "deal": "B", "arm": "none", "status": "failed"}) + "\n", encoding="utf-8")
    assert brief_eval.load_done(path) == {(1, "A", "none")}  # a failed brief is tried again


# -- statistics -----------------------------------------------------------------------------------------------

def test_wilson_interval_behaves_at_the_edges_and_in_the_middle():
    rate, low, high = stats.wilson(50, 100)
    assert rate == 0.5 and 0.40 < low < 0.41 and 0.59 < high < 0.60
    assert stats.wilson(0, 10)[1] == 0.0 and stats.wilson(10, 10)[2] == 1.0 and stats.wilson(0, 0) == (0.0, 0.0, 0.0)


def test_the_sign_test_is_exact_and_ignores_agreement():
    assert stats.sign_test([True] * 8 + [False] * 2, [False] * 8 + [False] * 2) == (8, 0, pytest.approx(2 * 0.5**8))
    assert stats.sign_test([True, False], [True, False]) == (0, 0, 1.0)
    assert stats.sign_test([True, False], [False, True])[2] == 1.0


def test_bootstrap_intervals_are_repeatable_and_contain_the_mean():
    values = [1.0] * 30 + [0.0] * 70
    a, b = stats.bootstrap_mean(values, seed=3), stats.bootstrap_mean(values, seed=3)
    assert a == b and a[1] < 0.3 < a[2]
    diff = stats.paired_difference([1.0] * 10, [0.0] * 10)
    assert diff[0] == 1.0 and diff[1] == 1.0
    with pytest.raises(ValueError):
        stats.paired_difference([1.0], [1.0, 0.0])


# -- charts and the report ---------------------------------------------------------------------------------------

def test_charts_are_wellformed_svg_with_every_series_and_escape_their_text():
    svg = charts.bar_chart("A <b> chart", "sub", ["g1", "g2"], {"One & two": [(0.5, 0.4, 0.6), (0.9, 0.8, 1.0)], "Three": [(0.1, 0.0, 0.2), (0.2, 0.1, 0.3)]})
    assert svg.startswith("<svg") and svg.endswith("</svg>") and "&lt;b&gt;" in svg and "One &amp; two" in svg and svg.count("<rect") >= 6
    line = charts.line_chart("L", "s", [10, 20], {"x": [(0.1, 0.0, 0.2), (0.5, 0.4, 0.6)]})
    assert "<polyline" in line and "<polygon" in line


def test_the_report_is_built_from_result_files_and_says_it_is_synthetic(tmp_path, monkeypatch):
    data = run(retrieval_eval.run([1], 24, 5, curve=False, say=lambda _m: None))
    data["curve_rows"] = data["rows"]
    data["design"]["curve_sizes"] = [24]
    for name in ("heldout_after.json", "heldout_before.json", "dev_after.json", "dev_before.json"):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    report.main()
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "synthetic" in text.lower() and "rules planted" in text and "Layer 1" in text and "What this evaluation found" in text
    assert "| Deal Intelligence |" in text and (tmp_path / "learning_curve.svg").exists() and Path(tmp_path / "retrieval_control.svg").exists()
