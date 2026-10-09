"""Recording an outcome: validation, play credit, the learning journal and what gets queued for Hindsight."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select

from deal_intelligence.api.v1.db import Deal, DealSignals, Lesson, MemoryEvent, PlayStats
from deal_intelligence.api.v1.outcomes import memory_journal, outcome_credit, rebuild_play_stats, record_outcome
from deal_intelligence.errors import PipelineError

from .builders import add_deal, make_context
from .fakes import ENGAGED, SILENT, add_catalogue, closed


def setup(tmp_path, plays=("PLAY-01", "PLAY-02")):
    ctx = make_context(tmp_path)
    add_catalogue(ctx.db)
    scheduled: list[int] = []
    ctx.schedule_sync = lambda: scheduled.append(1)  # type: ignore[method-assign]
    deal_id = add_deal(ctx.db, name="Acme", account="Acme Pay", objections=[("sso", "unresolved")], stakeholders=[SILENT],
                       plays=list(plays))
    return ctx, deal_id, scheduled


def stats(ctx):
    with ctx.db.session() as s:
        return {r.play_code: (r.times_used, r.won, r.lost_quality, r.lost_other, round(r.outcome_credit, 4))
                for r in s.scalars(select(PlayStats)) if r.times_used or r.outcome_credit}


def test_credit_rules():
    assert outcome_credit("won", None) == 0.1
    assert outcome_credit("lost", "security_compliance") == -0.1
    assert outcome_credit("lost", "feature_gap") == -0.1
    assert outcome_credit("lost", "unresolved_objection") == -0.1
    for reason in ("price", "no_decision", "timing", "champion_left", "competitor", None):
        assert outcome_credit("lost", reason) == 0.0


def test_a_win_credits_only_the_ticked_plays_and_queues_everything_for_hindsight(tmp_path):
    ctx, deal_id, scheduled = setup(tmp_path, plays=("PLAY-01", "PLAY-02", "PLAY-04"))  # PLAY-04 was planned, not done
    result = record_outcome(ctx, deal_id, "won", None, ["PLAY-01", "PLAY-02"], date(2026, 9, 30))
    assert result["credit"] == 0.1 and result["credit_applied"] == {"PLAY-01": 0.1, "PLAY-02": 0.1}
    assert result["plays_used"] == ["PLAY-01", "PLAY-02"] and result["previous"] is None and result["lessons_changed"] == 3
    assert stats(ctx) == {"PLAY-01": (1, 1, 0, 0, 0.1), "PLAY-02": (1, 1, 0, 0, 0.1)}

    with ctx.db.session() as s:
        deal = s.get(Deal, deal_id)
        assert (deal.result, deal.stage, deal.closed_on, deal.loss_reason) == ("won", "closed", date(2026, 9, 30), None)
        assert deal.hindsight_status == "pending" and deal.signals.plays_used == ["PLAY-01", "PLAY-02"]
        events = list(s.scalars(select(MemoryEvent).order_by(MemoryEvent.id)))
        assert [e.kind for e in events] == ["outcome_recorded", "play_credit", "play_credit"]
        assert events[0].deal_id == deal_id and "D-001 recorded as won" in events[0].detail
        assert [e.play_code for e in events[1:]] == ["PLAY-01", "PLAY-02"]
        lessons = {row.key: row for row in s.scalars(select(Lesson))}
        assert set(lessons) == {"outcome:D-001", "play:D-001:PLAY-01", "play:D-001:PLAY-02"}
        assert all(row.hindsight_status == "pending" for row in lessons.values())
    assert scheduled == [1]


def test_a_security_loss_costs_the_plays_and_a_price_loss_does_not(tmp_path):
    ctx, deal_id, _ = setup(tmp_path)
    record_outcome(ctx, deal_id, "lost", "security_compliance", ["PLAY-01"], date(2026, 9, 30))
    assert stats(ctx) == {"PLAY-01": (1, 0, 1, 0, -0.1)}

    ctx2, deal2, _ = setup(tmp_path / "second")
    result = record_outcome(ctx2, deal2, "lost", "price", ["PLAY-01"], date(2026, 9, 30))
    assert result["credit"] == 0.0 and result["credit_applied"] == {}
    assert stats(ctx2) == {"PLAY-01": (1, 0, 0, 1, 0.0)}
    with ctx2.db.session() as s:
        assert [e.kind for e in s.scalars(select(MemoryEvent))] == ["outcome_recorded"]  # nothing to credit, only the outcome
        assert s.get(Lesson, 1).signal == "neutral"


def test_re_recording_a_different_outcome_applies_only_the_difference(tmp_path):
    ctx, deal_id, _ = setup(tmp_path)
    record_outcome(ctx, deal_id, "won", None, ["PLAY-01", "PLAY-02"], date(2026, 9, 1))
    record_outcome(ctx, deal_id, "won", None, ["PLAY-01", "PLAY-02"], date(2026, 9, 1))  # same again: no change
    assert stats(ctx) == {"PLAY-01": (1, 1, 0, 0, 0.1), "PLAY-02": (1, 1, 0, 0, 0.1)}

    result = record_outcome(ctx, deal_id, "lost", "feature_gap", ["PLAY-01"], date(2026, 9, 2))
    assert result["previous"] == {"result": "won", "loss_reason": None}
    assert result["credit_applied"] == {"PLAY-01": -0.2, "PLAY-02": -0.1}  # the win is taken back, then the loss applied
    assert stats(ctx) == {"PLAY-01": (1, 0, 1, 0, -0.1)}  # PLAY-02 was unticked: back to nothing
    with ctx.db.session() as s:
        deal = s.get(Deal, deal_id)
        assert (deal.result, deal.loss_reason, deal.signals.plays_used) == ("lost", "feature_gap", ["PLAY-01"])
        recorded = [e.detail for e in s.scalars(select(MemoryEvent).order_by(MemoryEvent.id)) if e.kind == "outcome_recorded"]
        assert "previously won" in recorded[-1] and "recorded as lost (feature_gap)" in recorded[-1]
        lessons = {row.key: row for row in s.scalars(select(Lesson))}
        assert set(lessons) == {"outcome:D-001", "play:D-001:PLAY-01"} and lessons["outcome:D-001"].signal == "negative"
        assert lessons["outcome:D-001"].hindsight_status == "pending"  # the rewritten lesson is sent again


def test_the_interactions_of_a_closing_deal_keep_their_state_but_the_summary_is_queued(tmp_path):
    ctx = make_context(tmp_path)
    add_catalogue(ctx.db)
    ctx.schedule_sync = lambda: None  # type: ignore[method-assign]
    deal_id = add_deal(ctx.db, name="Acme", account="Acme", interactions=[("email", date(2026, 6, 1), "hi")])
    with ctx.db.session() as s:
        deal = s.get(Deal, deal_id)
        deal.hindsight_status = "retained"
        deal.interactions[0].hindsight_status = "retained"
        s.commit()
    record_outcome(ctx, deal_id, "won", None, ["PLAY-01"], date(2026, 9, 1))
    with ctx.db.session() as s:
        deal = s.get(Deal, deal_id)
        assert deal.hindsight_status == "pending" and deal.interactions[0].hindsight_status == "retained"


@pytest.mark.parametrize("kwargs, code, status", [
    ({"result": "maybe"}, "invalid_request", 422),
    ({"result": "lost"}, "invalid_request", 422),  # a loss needs a reason
    ({"result": "lost", "loss_reason": "boredom"}, "invalid_request", 422),
    ({"result": "won", "plays_used": ["PLAY-99"]}, "invalid_request", 422),  # not in the catalogue
])
def test_invalid_input_raises_pipeline_errors_and_changes_nothing(tmp_path, kwargs, code, status):
    ctx, deal_id, scheduled = setup(tmp_path)
    with pytest.raises(PipelineError) as exc:
        record_outcome(ctx, deal_id, **kwargs)
    assert (exc.value.code, exc.value.http_status) == (code, status)
    assert stats(ctx) == {} and scheduled == []
    with ctx.db.session() as s:
        assert s.get(Deal, deal_id).result == "open" and not list(s.scalars(select(MemoryEvent)))


def test_unknown_and_deleted_deals_are_not_found(tmp_path):
    ctx, deal_id, _ = setup(tmp_path)
    with pytest.raises(PipelineError) as exc:
        record_outcome(ctx, 999, "won")
    assert exc.value.http_status == 404
    with ctx.db.session() as s:
        s.get(Deal, deal_id).status = "deleted"
        s.commit()
    with pytest.raises(PipelineError) as exc:
        record_outcome(ctx, deal_id, "won")
    assert exc.value.code == "not_found"


def test_a_lost_deal_is_remembered_even_without_ticked_plays(tmp_path):
    ctx, deal_id, _ = setup(tmp_path)
    record_outcome(ctx, deal_id, "lost", "price", [], date(2026, 9, 1))
    with ctx.db.session() as s:
        assert s.get(DealSignals, deal_id).plays_used == []
    assert stats(ctx) == {}


def test_rebuild_matches_the_incremental_counts_and_ignores_deleted_deals(tmp_path):
    ctx, deal_id, _ = setup(tmp_path)
    record_outcome(ctx, deal_id, "won", None, ["PLAY-01"], date(2026, 9, 1))
    other = closed(ctx.db, "Old", "Old Co", "lost", loss_reason="security_compliance", plays=["PLAY-01", "PLAY-03"], stakeholders=[ENGAGED])
    gone = closed(ctx.db, "Gone", "Gone Co", "won", plays=["PLAY-03"])
    with ctx.db.session() as s:
        s.get(Deal, gone).status = "deleted"
        s.commit()
    assert other and rebuild_play_stats(ctx.db) == 2
    assert stats(ctx) == {"PLAY-01": (2, 1, 1, 0, 0.0), "PLAY-03": (1, 0, 1, 0, -0.1)}
    assert rebuild_play_stats(ctx.db) == 2 and stats(ctx)["PLAY-01"] == (2, 1, 1, 0, 0.0)  # idempotent


def test_the_journal_lists_newest_first_with_deal_codes(tmp_path):
    ctx, deal_id, _ = setup(tmp_path)
    record_outcome(ctx, deal_id, "won", None, ["PLAY-01"], date(2026, 9, 1))
    journal = memory_journal(ctx.db, limit=10)
    assert [e["kind"] for e in journal] == ["play_credit", "outcome_recorded"]
    assert journal[0]["deal_code"] == "D-001" and journal[0]["play_code"] == "PLAY-01" and "+0.1" in journal[0]["detail"]
    assert len(memory_journal(ctx.db, limit=1)) == 1
