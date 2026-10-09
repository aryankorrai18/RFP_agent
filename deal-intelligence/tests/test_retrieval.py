"""Retrieval end to end with fake Hindsight: modes, the structural gate, SQLite validation, degrade paths,
the close-the-loop demo moment and a leave-one-out check of the play ranking."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from deal_intelligence.api.v1.db import Deal, Interaction
from deal_intelligence.api.v1.memory import RecallHit
from deal_intelligence.api.v1.outcomes import record_outcome
from deal_intelligence.api.v1.retrieval import build_recommendations, closed_facts, deal_timeline, n_closed
from deal_intelligence.errors import PipelineError

from .builders import add_deal, make_context
from .fakes import BUYER, ENGAGED, SILENT, FakeLessons, FakeMemory, add_catalogue, add_open_fintech, seed_fintech_history


def world(tmp_path, *, lessons: bool = True, sync: bool = True, **settings):
    memory, fake_lessons = FakeMemory(), FakeLessons() if lessons else None
    ctx = make_context(tmp_path, memory=memory, lessons=fake_lessons, **settings)
    ids = seed_fintech_history(ctx.db)
    open_id = add_open_fintech(ctx.db)
    if sync:
        asyncio.run(ctx.sync())
    return ctx, memory, fake_lessons, ids, open_id


def run(ctx, deal_id, mode=None):
    return asyncio.run(build_recommendations(ctx, deal_id, mode))


def codes(items):
    return [i.code for i in items]


# ---- modes ------------------------------------------------------------------------------------------------------


def test_none_mode_brings_nothing_from_other_deals(tmp_path):
    ctx, memory, _lessons, _ids, open_id = world(tmp_path)
    rec = run(ctx, open_id, "none")
    assert (rec.mode, rec.n_closed, rec.similar, rec.plays, rec.warnings, rec.degraded) == ("none", 5, [], [], [], None)
    assert rec.memory_state and memory.recalls == []  # no search at all


def test_longctx_returns_every_closed_deal_by_id_without_plays_or_warnings(tmp_path):
    ctx, memory, _lessons, _ids, open_id = world(tmp_path)
    rec = run(ctx, open_id, "longctx")
    assert codes(rec.similar) == ["D-001", "D-002", "D-003", "D-004", "D-005"]
    assert [s.rank for s in rec.similar] == [1, 2, 3, 4, 5] and all(s.relevance == 0.0 for s in rec.similar)
    assert rec.plays == [] and rec.warnings == [] and rec.lessons_used is False and memory.recalls == []
    assert rec.similar[0].shared_keys == ["competitor:brightline", "objection:security_review", "objection:sso",
                                          "industry:fintech", "segment:mid_market"]
    assert rec.similar[4].shared_keys == []  # included, but it is plainly unrelated
    assert rec.similar[0].summary.startswith("A mid-market fintech deal.")


def test_invalid_mode_and_unknown_deal_are_pipeline_errors(tmp_path):
    ctx, _m, _l, _ids, open_id = world(tmp_path)
    with pytest.raises(PipelineError) as exc:
        run(ctx, open_id, "psychic")
    assert exc.value.http_status == 422
    with pytest.raises(PipelineError) as exc:
        run(ctx, 404)
    assert exc.value.http_status == 404


def test_default_mode_comes_from_settings(tmp_path):
    ctx, _m, _l, _ids, open_id = world(tmp_path, retrieval_mode="none")
    assert run(ctx, open_id).mode == "none"


# ---- the gate ---------------------------------------------------------------------------------------------------


def test_similar_mode_applies_the_gate_and_counts_plays_and_warnings_from_the_survivors(tmp_path):
    ctx, memory, lessons, ids, open_id = world(tmp_path)
    rec = run(ctx, open_id, "similar")
    assert set(codes(rec.similar)) == {"D-001", "D-002", "D-003", "D-004"}  # the off-topic retail deal is gone
    assert set(codes(rec.similar[:2])) == {"D-001", "D-003"}  # three shared problem keys each; the lost deal is not hidden
    assert set(rec.similar[0].shared_keys) >= {"objection:sso", "objection:security_review", "competitor:brightline"}
    assert rec.lessons_used is False and lessons.recalls == [] and rec.degraded is None

    by_code = {p.play_code: p for p in rec.plays}
    assert set(by_code) == {"PLAY-01", "PLAY-02"}  # PLAY-03 only appears in a lost deal, PLAY-06 is off-topic
    assert (by_code["PLAY-01"].used_in_similar, by_code["PLAY-01"].won_in_similar) == (2, 2)
    assert by_code["PLAY-01"].source_deals == ["D-001", "D-002"]
    assert by_code["PLAY-01"].reasons[0] == "used in 2 similar deals, 2 won" and by_code["PLAY-01"].lesson_signal == 0.0

    warnings = {w.kind: w for w in rec.warnings}
    sso = warnings["unresolved_objection"]
    assert (sso.objection_type, sso.similar_lost, sso.similar_total, sso.source_deals) == ("sso", 1, 2, ["D-003", "D-004"])  # D-004 won anyway
    assert (warnings["silent_sponsor"].similar_lost, warnings["silent_sponsor"].similar_total) == (1, 1)
    assert (warnings["competitor"].similar_lost, warnings["competitor"].similar_total) == (1, 2)


def test_an_off_topic_deal_with_strong_positive_lessons_never_appears_or_outranks_relevant_deals(tmp_path):
    ctx, _m, lessons, ids, open_id = world(tmp_path)
    for n in range(20):  # Hindsight is full of glowing lessons about the retail deal's play
        lessons.items.append({"content": f"PLAY-06 won again {n}", "document_id": f"lesson-extra-{n}",
                              "tags": ["play:PLAY-06", "signal:positive", "deal:D-005", "kind:play_result"]})
    rec = run(ctx, open_id, "hindsight")
    assert "D-005" not in codes(rec.similar) and "PLAY-06" not in [p.play_code for p in rec.plays]
    assert rec.lessons_used is True and {p.play_code for p in rec.plays} == {"PLAY-01", "PLAY-02"}


def test_hindsight_mode_adds_validated_lesson_evidence_to_plays(tmp_path):
    ctx, _m, lessons, ids, open_id = world(tmp_path)
    lessons.items.append({"content": "stale", "document_id": "lesson-not-in-sqlite",
                          "tags": ["play:PLAY-01", "signal:positive", "deal:D-001"]})  # SQLite has no such lesson
    rec = run(ctx, open_id, "hindsight")
    play = {p.play_code: p for p in rec.plays}["PLAY-01"]
    assert play.lesson_signal > 0 and play.lesson_evidence
    assert all(doc.startswith("lesson-play:") for doc in play.lesson_evidence) and "lesson-not-in-sqlite" not in play.lesson_evidence
    assert any(reason.startswith("Hindsight lessons: ") for reason in play.reasons)
    assert sorted(lessons.recalls[0][1]) == ["play:PLAY-01", "play:PLAY-02"]  # asked about the candidate plays only


def test_similar_deals_are_capped_at_k_with_the_best_structural_match_first(tmp_path):
    ctx, _m, _l, _ids, open_id = world(tmp_path, similar_deals_k=2)
    rec = run(ctx, open_id, "similar")
    assert set(codes(rec.similar)) == {"D-001", "D-003"}  # three shared problem keys beat everything else


# ---- the query and validation -------------------------------------------------------------------------------------


def test_the_recall_query_is_built_from_signals_not_the_account_name(tmp_path):
    ctx, memory, _l, _ids, open_id = world(tmp_path)
    memory.recalls.clear()
    run(ctx, open_id, "similar")
    query, tags, limit = memory.recalls[0]
    assert "SSO (unresolved)" in query and "security review (raised)" in query and "Brightline" in query
    assert "mid-market fintech" in query and "champion has gone silent" in query
    assert "Acme" not in query and "Payments" not in query
    assert tags == ["kind:deal_summary"] and limit == max(ctx.settings.similar_deals_k * 4, 12)


class ScriptedMemory(FakeMemory):
    def __init__(self, codes: list[str]):
        super().__init__()
        self.script = codes

    async def recall(self, query, tags, limit):
        return [RecallHit(code=c, rank=n, final=1.0 / n) for n, c in enumerate(self.script, start=1)]


def test_hits_are_validated_against_sqlite_deleted_unknown_self_open_and_duplicate_hits_are_dropped(tmp_path):
    memory = ScriptedMemory([])
    ctx = make_context(tmp_path, memory=memory)
    ids = seed_fintech_history(ctx.db)
    open_id = add_open_fintech(ctx.db)
    other_open = add_open_fintech(ctx.db, "Other", "Other Co")
    with ctx.db.session() as s:
        for deal in s.query(Deal):
            deal.hindsight_status = "retained"  # everything is "in Hindsight": no SQLite fallback for unsynced deals
        s.get(Deal, ids["lost_sso"]).status = "deleted"  # D-003 deleted in SQLite, still in Hindsight
        s.commit()
    memory.script = ["D-099", "D-003", "D-006", "D-007", "D-004", "D-004", "D-001", "INT-0001"]
    rec = run(ctx, open_id, "similar")
    assert codes(rec.similar) == ["D-001", "D-004"]  # structural order; D-002 was not recalled so it is not here
    assert [s.rank for s in rec.similar] == [7, 5]  # Hindsight's own rank is kept, hits that were dropped included
    assert other_open  # a different open deal is never similar either
    assert {w.kind for w in rec.warnings} >= {"unresolved_objection"}
    sso = next(w for w in rec.warnings if w.kind == "unresolved_objection")
    assert (sso.similar_lost, sso.similar_total) == (0, 1)  # the deleted loss no longer counts


def test_a_deal_closed_a_moment_ago_counts_before_it_reaches_hindsight(tmp_path):
    ctx, memory, _l, _ids, open_id = world(tmp_path, sync=False)  # nothing retained yet
    assert memory.docs == {}
    rec = run(ctx, open_id, "similar")
    assert set(codes(rec.similar)) == {"D-001", "D-002", "D-003", "D-004"} and rec.degraded is None


# ---- degrade paths ----------------------------------------------------------------------------------------------


def test_hindsight_down_falls_back_to_the_same_gate_over_sqlite_with_a_plain_note(tmp_path):
    ctx, memory, _l, _ids, open_id = world(tmp_path)
    normal = run(ctx, open_id, "similar")
    memory.available = False
    rec = run(ctx, open_id, "similar")
    assert rec.degraded and "unavailable" in rec.degraded and "local database" in rec.degraded
    assert set(codes(rec.similar)) == set(codes(normal.similar)) and "D-005" not in codes(rec.similar)
    assert sorted((p.play_code, p.used_in_similar, p.won_in_similar) for p in rec.plays) == sorted(
        (p.play_code, p.used_in_similar, p.won_in_similar) for p in normal.plays)  # equal ties may order differently
    assert [(w.kind, w.similar_lost, w.similar_total) for w in rec.warnings] == [(w.kind, w.similar_lost, w.similar_total) for w in normal.warnings]


def test_lessons_down_behaves_like_similar_mode_with_a_note(tmp_path):
    ctx, _m, lessons, _ids, open_id = world(tmp_path)
    similar = run(ctx, open_id, "similar")
    lessons.available = False
    rec = run(ctx, open_id, "hindsight")
    assert rec.lessons_used is False and "lessons" in rec.degraded and "unavailable" in rec.degraded
    assert codes(rec.similar) == codes(similar.similar) and [p.play_code for p in rec.plays] == [p.play_code for p in similar.plays]
    assert all(p.lesson_signal == 0.0 and p.lesson_evidence == [] for p in rec.plays)


def test_lessons_turned_off_is_a_note_not_an_error(tmp_path):
    ctx, _m, _l, _ids, open_id = world(tmp_path, lessons=False)
    rec = run(ctx, open_id, "hindsight")
    assert rec.lessons_used is False and "turned off" in rec.degraded and rec.plays
    ctx2, *_rest = world(tmp_path / "off", lessons=True, lessons_enabled=False)
    assert "turned off" in run(ctx2, 6, "hindsight").degraded


def test_both_banks_down_reports_both(tmp_path):
    ctx, memory, lessons, _ids, open_id = world(tmp_path)
    memory.available = lessons.available = False
    rec = run(ctx, open_id, "hindsight")
    assert "Hindsight is unavailable" in rec.degraded and "lessons" in rec.degraded and rec.similar and rec.plays


# ---- memory_state, timeline ---------------------------------------------------------------------------------------


def test_memory_state_is_stable_and_changes_with_the_memory(tmp_path):
    ctx, _m, _l, _ids, open_id = world(tmp_path)
    first, again = run(ctx, open_id, "hindsight"), run(ctx, open_id, "hindsight")
    assert first.memory_state == again.memory_state and len(first.memory_state) == 64
    assert run(ctx, open_id, "similar").memory_state != first.memory_state  # lesson ids are part of it
    add_deal(ctx.db, name="New", account="N", result="won", closed_on=date(2026, 5, 1), industry="fintech", plays=["PLAY-07"])
    assert run(ctx, open_id, "hindsight").memory_state != first.memory_state


def test_timeline_is_recalled_by_deal_tag_and_checked_against_sqlite(tmp_path):
    memory = FakeMemory()
    ctx = make_context(tmp_path, memory=memory)
    add_catalogue(ctx.db)
    deal_id = add_deal(ctx.db, name="Acme", account="Acme", interactions=[
        ("email", date(2026, 6, 10), "Second email about SSO"), ("call_note", date(2026, 6, 1), "First call about SSO")])
    other = add_deal(ctx.db, name="Other", account="Other", interactions=[("email", date(2026, 6, 5), "unrelated SSO")])
    asyncio.run(ctx.sync())
    with ctx.db.session() as s:  # INT-0001 removed from SQLite while Hindsight still holds it
        s.delete(s.get(Interaction, 1))
        s.commit()
    timeline = asyncio.run(deal_timeline(ctx, deal_id))
    assert [i["code"] for i in timeline["items"]] == ["INT-0002"] and timeline["degraded"] is None and other
    assert memory.recalls[-1][1] == ["kind:interaction", "deal:D-001"]

    memory.available = False
    fallback = asyncio.run(deal_timeline(ctx, deal_id))
    assert [i["code"] for i in fallback["items"]] == ["INT-0002"] and "unavailable" in fallback["degraded"]
    assert n_closed(ctx.db) == 0


# ---- close the loop: the demo moment ------------------------------------------------------------------------------


def test_recording_a_similar_loss_changes_the_warning_counts_and_the_plays(tmp_path):
    ctx, _m, _l, _ids, open_id = world(tmp_path)
    before = run(ctx, open_id, "hindsight")
    sso = next(w for w in before.warnings if w.kind == "unresolved_objection")
    assert (sso.similar_lost, sso.similar_total, sso.source_deals) == (1, 2, ["D-003", "D-004"])
    play02_before = next(p for p in before.plays if p.play_code == "PLAY-02")
    assert (play02_before.used_in_similar, play02_before.won_in_similar, play02_before.lost_quality_in_similar) == (2, 2, 0)

    # A similar deal is lost on a security objection after the team used the security pack and no SSO workshop.
    zenith = add_deal(ctx.db, name="Zenith", account="Zenith Pay", industry="fintech", segment="mid_market",
                      objections=[("sso", "unresolved")], stakeholders=[SILENT], plays=["PLAY-02"])
    record_outcome(ctx, zenith, "lost", "security_compliance", ["PLAY-02"], date(2026, 9, 30))
    asyncio.run(ctx.sync())

    after = run(ctx, open_id, "hindsight")
    sso2 = next(w for w in after.warnings if w.kind == "unresolved_objection")
    assert (sso2.similar_lost, sso2.similar_total) == (2, 3) and "D-007" in sso2.source_deals
    silent = next(w for w in after.warnings if w.kind == "silent_sponsor")
    assert (silent.similar_lost, silent.similar_total) == (2, 2)
    play02 = next(p for p in after.plays if p.play_code == "PLAY-02")
    assert (play02.used_in_similar, play02.won_in_similar, play02.lost_quality_in_similar) == (3, 2, 1)
    assert "D-007" in play02.source_deals and after.plays[0].play_code == "PLAY-01"  # the pack slipped behind the workshop
    assert after.n_closed == before.n_closed + 1 and after.memory_state != before.memory_state
    assert "D-007" in {s.code for s in after.similar}


# ---- leave-one-out ------------------------------------------------------------------------------------------------

# (archetype, deal kwargs). Contamination on purpose: C1 shares a competitor with the A deals.
HISTORY = [
    ("A", dict(result="won", objections=[("sso", "addressed"), ("security_review", "addressed")], competitors=["Brightline"], plays=["PLAY-01", "PLAY-02"], stakeholders=[ENGAGED, BUYER])),
    ("A", dict(result="won", objections=[("sso", "addressed")], plays=["PLAY-01"], stakeholders=[ENGAGED])),
    ("A", dict(result="won", objections=[("sso", "addressed"), ("security_review", "addressed")], plays=["PLAY-01", "PLAY-02"], stakeholders=[ENGAGED])),
    ("A", dict(result="lost", loss_reason="security_compliance", objections=[("sso", "unresolved")], plays=["PLAY-03"], stakeholders=[SILENT])),
    ("B", dict(industry="healthcare", segment="enterprise", result="won", objections=[("data_residency", "addressed"), ("legal_terms", "addressed")], plays=["PLAY-04", "PLAY-07"], stakeholders=[ENGAGED])),
    ("B", dict(industry="healthcare", segment="enterprise", result="won", objections=[("data_residency", "addressed")], plays=["PLAY-04"], stakeholders=[ENGAGED])),
    ("B", dict(industry="healthcare", segment="enterprise", result="lost", loss_reason="price", objections=[("data_residency", "unresolved")], plays=["PLAY-05"], stakeholders=[SILENT])),
    ("C", dict(industry="retail", segment="smb", result="won", objections=[("pricing", "addressed")], competitors=["Brightline"], plays=["PLAY-06"], stakeholders=[ENGAGED])),
    ("C", dict(industry="retail", segment="smb", result="won", objections=[("pricing", "addressed")], plays=["PLAY-06"], stakeholders=[ENGAGED])),
    ("C", dict(industry="retail", segment="smb", result="lost", loss_reason="no_decision", objections=[("pricing", "unresolved")], plays=["PLAY-07"], stakeholders=[SILENT])),
]
LEAVE_ONE_OUT_TARGET = 0.8


def test_leave_one_out_the_top_play_is_one_that_won_in_analogous_deals(tmp_path):
    """For each closed deal, pretend it is open with the signals it ended with, rank plays from the
    other nine, and check the top play is one that won in analogous deals (same archetype)."""
    hits, misses = 0, []
    for left_out, (archetype, _kw) in enumerate(HISTORY):
        memory = FakeMemory()
        ctx = make_context(tmp_path / f"loo{left_out}", memory=memory)
        add_catalogue(ctx.db)
        expected: set[str] = set()
        for index, (arch, kw) in enumerate(HISTORY):
            if index == left_out:
                continue
            kw = dict(kw)
            add_deal(ctx.db, name=f"deal {index}", account=f"acct {index}", closed_on=date(2026, 2, 1),
                     **{"industry": "fintech", "segment": "mid_market", **kw})
            if arch == archetype and kw["result"] == "won":
                expected |= set(kw["plays"])
        kw = {"industry": "fintech", "segment": "mid_market", **HISTORY[left_out][1]}
        clone = add_deal(ctx.db, name="clone", account="clone", **{k: v for k, v in kw.items()
                                                                  if k not in ("result", "loss_reason", "plays", "stakeholders")},
                         result="open", stakeholders=kw.get("stakeholders"), plays=[])
        asyncio.run(ctx.sync())  # summaries retained, so recall (word overlap here) supplies the Hindsight rank
        rec = asyncio.run(build_recommendations(ctx, clone, "similar"))
        top = rec.plays[0].play_code if rec.plays else None
        if top in expected:
            hits += 1
        else:
            misses.append((left_out, archetype, top, sorted(expected)))
    assert hits / len(HISTORY) >= LEAVE_ONE_OUT_TARGET, f"misses: {misses}"
    assert len(closed_facts(ctx.db)) == 9  # last world: 9 closed deals and the open clone
