"""Plays to avoid: the rule, retrieval, the prompt, brief validation, the comparison and the seeded pricing story.
Offline: fakes for Hindsight, the lessons bank and the model."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from deal_intelligence.api.v1 import demo, experiment, outcomes
from deal_intelligence.api.v1.brief_prompts import candidate_plays
from deal_intelligence.api.v1.briefs import _catalogue, generate_brief, latest_brief
from deal_intelligence.api.v1.contracts import AvoidPlay, Recommendations
from deal_intelligence.api.v1.db import Brief
from deal_intelligence.api.v1.experiment import diff_briefs, latest_comparison, start_comparison
from deal_intelligence.api.v1.ranking import DealFacts, avoid_plays
from deal_intelligence.api.v1.retrieval import build_recommendations
from deal_intelligence.schemas import BriefStep, DealBriefResult

from .builders import add_deal, make_context
from .fake_llm import FakeCall, FakeLLM, default_brief
from .fakes import SILENT, FakeLessons, FakeMemory, add_open_fintech, seed_fintech_history
from .test_briefs import World, fields

NAMES = {"PLAY-05": "Discount offer", "PLAY-03": "ROI workbook"}


def facts(deal_id: int, *, result: str = "lost", plays=(), **kw) -> DealFacts:
    return DealFacts(
        deal_id=deal_id, code=f"D-{deal_id:03d}", account=f"Acct {deal_id}", industry="retail", segment="mid_market",
        stage="closed", result=result, loss_reason="price" if result == "lost" else None, plays_used=tuple(plays), **kw,
    )


OPEN = facts(1, result="open")


# ---- the rule -------------------------------------------------------------------------------------------------------


def test_a_play_used_in_three_similar_deals_with_no_win_is_avoided_with_honest_counts():
    similar = [facts(n, plays=["PLAY-05"]) for n in (4, 2, 3)] + [facts(5, result="won", plays=["PLAY-03"])]
    [item] = avoid_plays(OPEN, similar, NAMES)
    assert (item.play_code, item.name) == ("PLAY-05", "Discount offer")
    assert (item.used_in_similar, item.won_in_similar, item.lost_in_similar) == (3, 0, 3)
    assert item.source_deals == ["D-002", "D-003", "D-004"]
    assert item.text == "Discount offer: used in 3 similar deals, 0 won."
    assert item.to_dict()["play_code"] == "PLAY-05"


def test_below_the_threshold_or_with_a_single_win_a_play_is_not_avoided():
    two = [facts(n, plays=["PLAY-05"]) for n in (2, 3)]
    assert avoid_plays(OPEN, two, NAMES) == []
    assert [a.play_code for a in avoid_plays(OPEN, two, NAMES, min_used=2)] == ["PLAY-05"]
    one_win = [facts(n, plays=["PLAY-05"]) for n in (2, 3, 4)] + [facts(5, result="won", plays=["PLAY-05"])]
    assert avoid_plays(OPEN, one_win, NAMES) == []  # honest: it won once, so it is not "never won"


def test_order_is_most_used_first_then_code_and_a_play_the_open_deal_used_is_still_listed():
    similar = [facts(n, plays=["PLAY-09", "PLAY-05"]) for n in (2, 3, 4)] + [facts(5, plays=["PLAY-05"])]
    open_deal = facts(1, result="open", plays=["PLAY-09"])
    assert [(a.play_code, a.used_in_similar) for a in avoid_plays(open_deal, similar, NAMES)] == [("PLAY-05", 4), ("PLAY-09", 3)]
    assert avoid_plays(open_deal, similar, NAMES)[1].name == "PLAY-09"  # unknown names fall back to the code


def test_a_deal_counts_a_play_once_and_the_deal_itself_never_counts():
    similar = [facts(2, plays=["PLAY-05", "PLAY-05"]), facts(3, plays=["PLAY-05"]), facts(1, plays=["PLAY-05"])]
    assert avoid_plays(OPEN, similar, NAMES) == []  # D-001 is the open deal; D-002 counts once
    assert avoid_plays(OPEN, [facts(2, plays=["PLAY-05"])], NAMES, min_used=1)[0].text == "Discount offer: used in 1 similar deal, 0 won."


# ---- retrieval --------------------------------------------------------------------------------------------------------


def avoid_world(tmp_path, *, memory=None, lessons=None, extra_losses: int = 2):
    """The fintech history plus extra similar losses that all used PLAY-03 (the discount in this catalogue)."""
    memory = memory if memory is not None else FakeMemory()
    ctx = make_context(tmp_path, memory=memory, lessons=lessons if lessons is not None else FakeLessons())
    seed_fintech_history(ctx.db)
    for n in range(extra_losses):
        add_deal(ctx.db, name=f"Lost {n}", account=f"Lost Co {n}", result="lost", loss_reason="price", closed_on=date(2026, 3, 1),
                 industry="fintech", segment="mid_market", objections=[("sso", "unresolved")], plays=["PLAY-03"], stakeholders=[SILENT])
    open_id = add_open_fintech(ctx.db)
    asyncio.run(ctx.sync())
    return ctx, memory, open_id


def run(ctx, deal_id, mode="hindsight", **kw):
    return asyncio.run(build_recommendations(ctx, deal_id, mode, **kw))


def test_similar_and_hindsight_fill_avoid_and_none_and_longctx_leave_it_empty(tmp_path):
    ctx, _memory, open_id = avoid_world(tmp_path)
    for mode in ("similar", "hindsight"):
        rec = run(ctx, open_id, mode)
        assert [(a.play_code, a.used_in_similar, a.won_in_similar) for a in rec.avoid] == [("PLAY-03", 3, 0)]
        assert rec.avoid[0].name == "Discount offer" and rec.avoid[0].source_deals == ["D-003", "D-006", "D-007"]
        assert "PLAY-03" not in {p.play_code for p in rec.plays} and {p.play_code for p in rec.plays} == {"PLAY-01", "PLAY-02"}
    for mode in ("none", "longctx"):
        assert run(ctx, open_id, mode).avoid == []


def test_avoid_is_part_of_the_memory_state(tmp_path):
    ctx, _memory, open_id = avoid_world(tmp_path)
    with_avoid = run(ctx, open_id, "similar")
    ctx2, _m2, open2 = avoid_world(tmp_path / "fewer", extra_losses=1)
    without = run(ctx2, open2, "similar")
    assert without.avoid == [] and with_avoid.memory_state != without.memory_state
    assert with_avoid.memory_state == run(ctx, open_id, "similar").memory_state


def test_source_sqlite_never_calls_memory_or_lessons_and_is_not_degraded(tmp_path):
    class Boom(FakeMemory):
        async def recall(self, query, tags, limit):
            raise AssertionError("source='sqlite' must not recall")

    class BoomLessons(FakeLessons):
        async def recall(self, query, tags, limit=20):
            raise AssertionError("source='sqlite' must not recall lessons")

    ctx, _memory, open_id = avoid_world(tmp_path / "ok")
    normal = run(ctx, open_id, "hindsight")
    ctx2, _m, open2 = avoid_world(tmp_path / "boom", memory=Boom(), lessons=BoomLessons())
    # the world was synced with a memory whose retain works; only recall raises
    for mode in ("similar", "hindsight"):
        rec = run(ctx2, open2, mode, source="sqlite")
        assert rec.degraded is None and rec.lessons_used is False
        assert {s.code for s in rec.similar} == {s.code for s in normal.similar}
        assert [(a.play_code, a.used_in_similar) for a in rec.avoid] == [(a.play_code, a.used_in_similar) for a in normal.avoid]
        assert {p.play_code for p in rec.plays} == {p.play_code for p in normal.plays}
    with pytest.raises(AssertionError):  # the default source does recall, so the guard is real
        asyncio.run(build_recommendations(ctx2, open2, "similar", source="memory"))


def test_an_unknown_source_is_rejected(tmp_path):
    from deal_intelligence.errors import PipelineError

    ctx, _memory, open_id = avoid_world(tmp_path)
    with pytest.raises(PipelineError):
        run(ctx, open_id, "similar", source="telepathy")


def test_exclude_deal_ids_removes_candidates_and_their_counts(tmp_path):
    ctx, _memory, open_id = avoid_world(tmp_path)
    full = run(ctx, open_id, "similar", source="sqlite")
    left_out = run(ctx, open_id, "similar", source="sqlite", exclude_deal_ids=frozenset({6}))
    assert "D-006" in {s.code for s in full.similar} and "D-006" not in {s.code for s in left_out.similar}
    assert left_out.n_closed == full.n_closed - 1
    assert left_out.avoid == [] and [a.used_in_similar for a in full.avoid] == [3]  # two uses left: below the threshold


def test_a_closed_deal_can_be_the_target_and_is_never_its_own_neighbour(tmp_path):
    ctx, _memory, _open = avoid_world(tmp_path)
    won_id = 1  # D-001: won with PLAY-01 and PLAY-02
    rec = run(ctx, won_id, "similar", source="sqlite", exclude_deal_ids=frozenset({won_id}))
    assert "D-001" not in {s.code for s in rec.similar} and rec.n_closed == 6  # seven closed deals, minus the target
    assert rec.degraded is None
    plain = run(ctx, won_id, "similar", source="sqlite")
    assert "D-001" not in {s.code for s in plain.similar}
    # its own plays are what is being predicted, so they are not filtered out as "already done"
    assert "PLAY-01" in {p.play_code for p in rec.plays}


# ---- the prompt and the brief -----------------------------------------------------------------------------------------


def avoid_item(world: World, code: str = "PLAY-03") -> AvoidPlay:
    return AvoidPlay(code, "Pricing workshop", 3, 0, 3, [world.lost_code, "D-050", "D-051"],
                     "Pricing workshop: used in 3 similar deals, 0 won.")


def test_prompt_has_the_avoid_block_inside_evidence_and_the_hash_is_unchanged(tmp_path):
    world = World(tmp_path)
    recs = world.recs(avoid=[avoid_item(world)])
    prompts = {mode: world.prompt(mode, recs) for mode in ("none", "longctx", "hindsight", "similar")}
    assert len({p[2] for p in prompts.values()}) == 1  # prompt_hash is identical across modes
    for mode in ("none", "longctx"):
        assert "<avoid>" not in prompts[mode][1]
    for mode in ("hindsight", "similar"):
        user = prompts[mode][1]
        assert user.index("<evidence>") < user.index("<avoid>") < user.index("</avoid>") < user.index("</evidence>")
        block = user.split("<avoid>")[1].split("</avoid>")[0]
        assert 'code="PLAY-03"' in block and 'used="3"' in block and 'won="0"' in block
        assert world.lost_code in block and "D-050" in block and "Do not recommend" in block
        assert "lesson" not in user.lower() and "score" not in user.lower()
    assert world.prompt("hindsight", world.recs())[2] == prompts["hindsight"][2]  # avoid or not, the hash is the same


def test_avoided_plays_are_not_offered_as_candidates(tmp_path):
    world = World(tmp_path)
    recs = world.recs(avoid=[avoid_item(world, "PLAY-02")])  # PLAY-02 is also in the ranked plays: the avoid list wins
    view_catalogue = _catalogue(world.ctx.db)
    offered = [p["code"] for p in candidate_plays(recs, view_catalogue, {"plays_used": []}, "hindsight")]
    assert offered == ["PLAY-01"]
    assert "PLAY-02" in [p["code"] for p in candidate_plays(recs, view_catalogue, {"plays_used": []}, "none")]  # none: no memory, no avoid
    empty = world.recs(plays=[], avoid=[avoid_item(world, "PLAY-03")])
    fallback = [p["code"] for p in candidate_plays(empty, view_catalogue, {"plays_used": []}, "hindsight")]
    assert "PLAY-03" not in fallback and fallback


def test_a_next_step_on_an_avoided_play_is_dropped_and_flagged(tmp_path):
    world = World(tmp_path)
    recs = world.recs(avoid=[avoid_item(world, "PLAY-03")])
    world.llm.scripts[DealBriefResult] = fields(next_steps=[
        BriefStep(play_code="PLAY-03", rationale="Offer a discount.", source_ids=[world.lost_code]),
        BriefStep(play_code="PLAY-01", rationale="Security pack.", source_ids=[world.won_code]),
    ])
    brief = world.generate("hindsight", recs)
    assert [s["play_code"] for s in brief.content["next_steps"]] == ["PLAY-01"]
    flagged = [f for f in brief.flags if f["code"] == "avoided_play"]
    assert len(flagged) == 1 and "PLAY-03" in flagged[0]["detail"] and flagged[0]["where"] == "next_steps[0]"
    assert "play_not_offered" not in [f["code"] for f in brief.flags]


def test_a_brief_with_only_avoided_steps_is_asked_again_and_stores_the_avoid_list(tmp_path):
    world = World(tmp_path)
    recs = world.recs(avoid=[avoid_item(world, "PLAY-03")])
    bad = fields(next_steps=[BriefStep(play_code="PLAY-03", rationale="Discount.", source_ids=[])])
    good = fields(next_steps=[BriefStep(play_code="PLAY-01", rationale="Security pack.", source_ids=[world.won_code])])
    world.llm.scripts[DealBriefResult] = [bad, good]
    brief = world.generate("hindsight", recs)
    assert len(world.llm.calls) == 2 and "<correction>" in world.llm.calls[1].user
    assert "PLAY-03" in world.llm.calls[1].user.split("<correction>")[1] and "avoided" in world.llm.calls[1].user.split("<correction>")[1]
    assert [s["play_code"] for s in brief.content["next_steps"]] == ["PLAY-01"]
    codes = [f["code"] for f in brief.flags]
    assert "avoided_play" in codes and "retried" in codes
    assert brief.content["avoid"] == [avoid_item(world, "PLAY-03").to_dict()]
    assert brief.evidence["avoid"] == brief.content["avoid"]
    # none and longctx never show or store an avoid list
    for mode in ("none", "longctx"):
        assert world.generate(mode, recs).content["avoid"] == []


# ---- the comparison -----------------------------------------------------------------------------------------------------


def discount_blind_script(call: FakeCall) -> DealBriefResult:
    """The 'obvious' model: with no avoid block it recommends the discount; with one it recommends PLAY-03."""
    play = "PLAY-03" if "<avoid>" in call.user else "PLAY-05"
    step = BriefStep(play_code=play, rationale="The obvious next step.", source_ids=[])
    return default_brief(call).model_copy(update={"next_steps": [step]})


def seeded_world(tmp_path, llm=None):
    ctx = make_context(tmp_path, memory=FakeMemory(), lessons=FakeLessons(), llm=llm or FakeLLM(brief=discount_blind_script))
    info = demo.seed_demo(ctx)
    outcomes.rebuild_play_stats(ctx.db)
    asyncio.run(ctx.sync())
    return ctx, info


def compare(ctx, deal_id, arms=None) -> dict:
    async def go() -> None:
        job = start_comparison(ctx, deal_id, arms)
        await ctx.jobs.wait(job.id)

    asyncio.run(go())
    return latest_comparison(ctx, deal_id)


def test_comparison_names_the_arms_that_recommended_an_avoided_play(tmp_path):
    ctx, _info = seeded_world(tmp_path)
    deal_id = 24
    comparison = compare(ctx, deal_id)
    assert comparison["job"]["status"] == "completed"
    differs = comparison["differs"]
    assert differs["avoided_play_recommended_by"] == ["none", "longctx"]
    assert differs["play_codes"]["hindsight"] == ["PLAY-03"] and differs["play_codes"]["none"] == ["PLAY-05"]
    hindsight_evidence = comparison["arms"]["hindsight"]["evidence"]
    assert [a["play_code"] for a in hindsight_evidence["avoid"]] == ["PLAY-05"]
    assert comparison["prompt_hash_same"] is True


def test_comparison_without_a_hindsight_arm_or_without_avoid_is_empty(tmp_path):
    ctx, _info = seeded_world(tmp_path)
    assert compare(ctx, 24, ["none"])["differs"]["avoided_play_recommended_by"] == []
    # the Cedarline demo deal has no avoided play, so nothing can be recommended against it
    assert compare(ctx, 15)["differs"]["avoided_play_recommended_by"] == []


# ---- brief diff ----------------------------------------------------------------------------------------------------------


def test_diff_briefs_reports_added_removed_and_changed_avoid_plays(tmp_path):
    ctx = make_context(tmp_path)
    deal_id = add_deal(ctx.db, name="Acme", account="Acme")

    def avoid(code: str, used: int) -> dict:
        return AvoidPlay(code, code, used, 0, used, ["D-002"], f"{code}: used in {used} similar deals, 0 won.").to_dict()

    with ctx.db.session() as session:
        before = Brief(deal_id=deal_id, mode="hindsight", evidence={"avoid": [avoid("PLAY-05", 3), avoid("PLAY-09", 3)]}, content={})
        after = Brief(deal_id=deal_id, mode="hindsight", evidence={"avoid": [avoid("PLAY-05", 4), avoid("PLAY-07", 3)]}, content={})
        session.add_all([before, after])
        session.commit()
        ids = (before.id, after.id)
    diff = diff_briefs(ctx, *ids)
    assert [a["play_code"] for a in diff["added_avoid"]] == ["PLAY-07"]
    assert [a["play_code"] for a in diff["removed_avoid"]] == ["PLAY-09"]
    assert diff["changed_avoid"] == [{"play_code": "PLAY-05", "from": {"used": 3, "won": 0}, "to": {"used": 4, "won": 0}}]
    empty = diff_briefs(ctx, ids[0], ids[0])
    assert empty["added_avoid"] == empty["removed_avoid"] == empty["changed_avoid"] == []


# ---- the seeded pricing story, end to end ----------------------------------------------------------------------------


def test_the_open_pricing_deal_is_told_not_to_discount_and_led_to_the_workbook(tmp_path):
    ctx, _info = seeded_world(tmp_path, llm=FakeLLM())
    rec = run(ctx, 24, "hindsight")
    assert {"D-004", "D-008", "D-019", "D-020", "D-021", "D-022", "D-023"} <= {s.code for s in rec.similar}
    [avoid] = rec.avoid
    assert (avoid.play_code, avoid.name, avoid.used_in_similar, avoid.won_in_similar) == ("PLAY-05", "Discount offer", 4, 0)
    assert avoid.source_deals == ["D-008", "D-019", "D-020", "D-021"]
    assert avoid.text == "Discount offer: used in 4 similar deals, 0 won."
    assert [p.play_code for p in rec.plays[:2]] == ["PLAY-03", "PLAY-02"]
    assert "PLAY-05" not in {p.play_code for p in rec.plays}
    assert rec.lessons_used is True and rec.degraded is None

    asyncio.run(generate_brief(ctx, 24, "hindsight", recommendations=rec, today=date.today()))
    brief = latest_brief(ctx.db, 24, "hindsight")
    assert brief.status == "ready" and [a["play_code"] for a in brief.content["avoid"]] == ["PLAY-05"]
    assert brief.content["next_steps"][0]["play_code"] == "PLAY-03"  # the default model picks the first offered play


def test_the_new_pricing_deals_leave_the_cedarline_sso_evidence_alone(tmp_path):
    ctx, info = seeded_world(tmp_path, llm=FakeLLM())
    cedarline = int(info["demo_deal"][2:])
    rec = run(ctx, cedarline, "hindsight")
    sso = next(w for w in rec.warnings if w.objection_type == "sso")
    assert (sso.similar_lost, sso.similar_total) == (2, 3)
    assert not {"D-019", "D-020", "D-021", "D-022", "D-023", "D-024"} & {s.code for s in rec.similar}
    assert rec.avoid == []
    outcomes.record_outcome(ctx, int(info["closable_deal"][2:]), "lost", "unresolved_objection", ["PLAY-05"], date(2026, 9, 30))
    asyncio.run(ctx.sync())
    after = run(ctx, cedarline, "hindsight")
    sso_after = next(w for w in after.warnings if w.objection_type == "sso")
    assert (sso_after.similar_lost, sso_after.similar_total) == (3, 4)


def test_default_fake_model_still_picks_an_offered_play_when_an_avoid_block_is_present(tmp_path):
    world = World(tmp_path)
    brief = world.generate("hindsight", world.recs(avoid=[avoid_item(world, "PLAY-03")]))
    assert brief.content["next_steps"][0]["play_code"] == "PLAY-01" and "avoided_play" not in [f["code"] for f in brief.flags]
    assert Recommendations(deal_id=1, mode="none", n_closed=0).avoid == []
