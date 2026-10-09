"""The structural gate, contrast and warnings are pure functions: tested without a database."""

from __future__ import annotations

from deal_intelligence.api.v1.ranking import (
    DealFacts, LessonEvidence, contrast_candidates, gate_and_order, passes_gate, rank_plays, recall_query,
    shared_keys, situation_text, warnings,
)

_ids = iter(range(1, 1000))


def facts(code_id: int | None = None, *, result="won", loss_reason=None, industry="fintech", segment="mid_market",
          objections=(), competitors=(), plays=(), sponsor="champion_engaged", buyer="engaged", account="Acct") -> DealFacts:
    deal_id = code_id or next(_ids) + 100
    return DealFacts(
        deal_id=deal_id, code=f"D-{deal_id:03d}", account=account, industry=industry, segment=segment, stage="closed",
        result=result, loss_reason=loss_reason, objections=tuple(objections), competitors=tuple(competitors),
        plays_used=tuple(plays), sponsor_state=sponsor, economic_buyer=buyer,
    )


OPEN = facts(1, result="open", objections=[("sso", "unresolved")], competitors=["Brightline"], sponsor="champion_silent")


def test_gate_passes_on_a_shared_problem_key():
    other = facts(industry="retail", segment="smb", objections=[("sso", "addressed")])
    assert shared_keys(OPEN, other) == ["objection:sso"]
    assert passes_gate(OPEN, other)


def test_gate_passes_on_same_segment_and_industry_alone():
    other = facts(objections=[("pricing", "addressed")])
    assert shared_keys(OPEN, other) == ["industry:fintech", "segment:mid_market"]
    assert passes_gate(OPEN, other)


def test_gate_rejects_one_context_match_without_a_shared_problem():
    assert not passes_gate(OPEN, facts(segment="enterprise", objections=[("pricing", "raised")]))  # same industry only
    assert not passes_gate(OPEN, facts(industry="retail", objections=[("pricing", "raised")]))  # same segment only


def test_gate_honours_min_shared_keys_and_never_matches_the_deal_itself():
    one_key = facts(industry="retail", segment="smb", objections=[("sso", "addressed")])
    two_keys = facts(industry="retail", segment="smb", objections=[("sso", "addressed")], competitors=["brightline"])
    assert not passes_gate(OPEN, one_key, min_shared_keys=2)
    assert passes_gate(OPEN, two_keys, min_shared_keys=2)
    assert not passes_gate(OPEN, OPEN)


def test_competitor_keys_are_slugged():
    other = facts(industry="retail", segment="smb", competitors=["BRIGHTLINE"])
    assert shared_keys(OPEN, other) == ["competitor:brightline"]


def test_gate_orders_by_problem_keys_and_uses_hindsight_rank_only_as_the_tiebreak():
    weak = facts(10, industry="retail", segment="smb", objections=[("sso", "addressed")])
    strong = facts(11, industry="retail", segment="smb", objections=[("sso", "addressed")], competitors=["Brightline"])
    same_strength = facts(12, industry="retail", segment="smb", objections=[("sso", "addressed")])
    offtopic = facts(13, industry="retail", segment="smb", objections=[("pricing", "addressed")])
    kept = gate_and_order(OPEN, [(weak, 1), (offtopic, 2), (same_strength, 3), (strong, 9)])
    assert [f.deal_id for f, _r, _k in kept] == [11, 10, 12]  # strong first despite rank 9; off-topic is gone


def test_contrast_uses_won_deals_excludes_plays_already_done_and_counts_all_users():
    done = facts(1, result="open", objections=[("sso", "unresolved")], plays=["PLAY-02"])
    similar = [
        facts(2, objections=[("sso", "addressed")], plays=["PLAY-01", "PLAY-02"]),
        facts(3, objections=[("sso", "addressed")], plays=["PLAY-01"]),
        facts(4, result="lost", loss_reason="security_compliance", objections=[("sso", "unresolved")], plays=["PLAY-01", "PLAY-09"]),
        facts(5, result="lost", loss_reason="price", objections=[("sso", "unresolved")], plays=["PLAY-01"]),
    ]
    rows = {c.play_code: c for c in contrast_candidates(done, similar)}
    assert set(rows) == {"PLAY-01"}  # PLAY-02 already done; PLAY-09 only appears in a lost deal
    row = rows["PLAY-01"]
    assert (row.used, row.won, row.lost_quality, row.lost_other) == (4, 2, 1, 1)
    assert row.source_deals == ["D-002", "D-003", "D-004", "D-005"]


def test_warning_counts_are_honest_including_a_win_despite_the_pattern():
    similar = [
        facts(2, result="lost", loss_reason="security_compliance", objections=[("sso", "unresolved")], sponsor="champion_silent"),
        facts(3, result="lost", loss_reason="price", objections=[("sso", "unresolved")]),
        facts(4, result="won", objections=[("sso", "unresolved")]),  # noise: won anyway
        facts(5, result="won", objections=[("sso", "addressed")]),  # resolved: not part of the unresolved count
    ]
    by_kind = {w.kind: w for w in warnings(OPEN, similar)}
    sso = by_kind["unresolved_objection"]
    assert (sso.similar_lost, sso.similar_total, sso.objection_type) == (2, 3, "sso")
    assert sso.source_deals == ["D-002", "D-003", "D-004"]
    assert "2 of 3" in sso.text and "1 won despite it" in sso.text
    silent = by_kind["silent_sponsor"]
    assert (silent.similar_lost, silent.similar_total, silent.source_deals) == (1, 1, ["D-002"])


def test_no_warning_without_evidence_among_similar_deals():
    assert warnings(OPEN, [facts(2, objections=[("pricing", "addressed")])]) == []


def test_competitor_and_no_champion_warnings():
    open_deal = facts(1, result="open", competitors=["Brightline"], sponsor="none")
    similar = [facts(2, result="lost", loss_reason="competitor", competitors=["Brightline"], sponsor="none"),
               facts(3, result="won", competitors=["Brightline"], sponsor="champion_engaged")]
    by_kind = {w.kind: w for w in warnings(open_deal, similar)}
    assert (by_kind["competitor"].similar_lost, by_kind["competitor"].similar_total) == (1, 2)
    assert (by_kind["no_champion"].similar_lost, by_kind["no_champion"].similar_total) == (1, 1)


def test_plays_order_by_problem_keys_then_win_ratio_then_volume_then_lessons():
    similar = [
        facts(2, objections=[("sso", "addressed")], competitors=["Brightline"], plays=["PLAY-A"]),  # 2 problem keys
        facts(3, objections=[("sso", "addressed")], plays=["PLAY-B", "PLAY-C", "PLAY-D"]),
        facts(4, objections=[("sso", "addressed")], plays=["PLAY-B", "PLAY-C"]),
        facts(5, result="lost", loss_reason="price", objections=[("sso", "unresolved")], plays=["PLAY-C"]),
    ]
    recs = rank_plays(contrast_candidates(OPEN, similar), {"PLAY-A": "Alpha"})
    # A wins on problem keys; B (2/2) beats C (2/3); D (1/1, used once) trails B on volume.
    assert [r.play_code for r in recs] == ["PLAY-A", "PLAY-B", "PLAY-D", "PLAY-C"]
    assert recs[1].reasons[0] == "used in 2 similar deals, 2 won"
    assert recs[0].name == "Alpha"


def test_lesson_signal_only_breaks_exact_ties_and_hindsight_rank_comes_last():
    similar = [facts(2, objections=[("sso", "addressed")], plays=["PLAY-A", "PLAY-B", "PLAY-C"])]
    contrast = contrast_candidates(OPEN, similar)
    plain = rank_plays(contrast, {}, hit_ranks={"D-002": 1})
    assert [r.play_code for r in plain] == ["PLAY-A", "PLAY-B", "PLAY-C"]
    boosted = rank_plays(contrast, {}, lessons={"PLAY-C": LessonEvidence(net=2.0, positive=2, evidence=["lesson-x"])})
    assert boosted[0].play_code == "PLAY-C" and boosted[0].lesson_evidence == ["lesson-x"]
    assert any("Hindsight lessons: 2 positive" in reason for reason in boosted[0].reasons)


def test_lessons_cannot_rescue_a_play_the_gate_never_produced():
    # The off-topic deal is gated out, so its play is not a candidate however strong its lessons are.
    offtopic = facts(2, industry="retail", segment="smb", objections=[("pricing", "addressed")], plays=["PLAY-X"])
    relevant = facts(3, objections=[("sso", "addressed")], plays=["PLAY-A"])
    kept = [f for f, _r, _k in gate_and_order(OPEN, [(offtopic, 1), (relevant, 2)])]
    recs = rank_plays(contrast_candidates(OPEN, kept), {}, lessons={"PLAY-X": LessonEvidence(net=50.0, positive=50)})
    assert [r.play_code for r in recs] == ["PLAY-A"]


def test_recall_query_is_built_from_signals_not_the_account():
    query = recall_query(facts(1, result="open", account="Zebra Industries", objections=[("sso", "unresolved"), ("pricing", "raised")],
                               competitors=["Brightline"], sponsor="champion_silent", buyer="none"))
    assert "SSO (unresolved)" in query and "pricing (raised)" in query and "Brightline" in query
    assert "fintech" in query and "mid-market" in query and "champion has gone silent" in query
    assert "Zebra" not in query


def test_situation_text_is_the_same_vocabulary_for_open_and_closed_deals():
    closed_text = situation_text(facts(objections=[("sso", "unresolved")]), closed=True)
    assert "SSO (stayed unresolved)" in closed_text and closed_text.startswith("A mid-market fintech deal.")


def test_the_catalogue_label_stops_boosting_a_play_this_companys_history_contradicts():
    similar = [
        facts(2, result="lost", loss_reason="unresolved_objection", objections=[("sso", "unresolved")], plays=["PLAY-OBVIOUS"]),
        facts(3, result="lost", loss_reason="unresolved_objection", objections=[("sso", "unresolved")], plays=["PLAY-OBVIOUS"]),
        facts(4, result="lost", loss_reason="unresolved_objection", objections=[("sso", "unresolved")], plays=["PLAY-OBVIOUS"]),
        facts(5, objections=[("sso", "addressed")], plays=["PLAY-OBVIOUS"]),
        facts(6, objections=[("sso", "addressed")], plays=["PLAY-BETTER"]),
        facts(7, objections=[("sso", "addressed")], plays=["PLAY-BETTER"]),
    ]
    labels = {"PLAY-OBVIOUS": frozenset({"sso"})}
    ranked = rank_plays(contrast_candidates(OPEN, similar), {}, addresses=labels, open_objections=frozenset({"sso"}))
    assert [r.play_code for r in ranked][0] == "PLAY-BETTER"  # 1 of 4 won with the labelled play, 2 of 2 with the other
    few = rank_plays(contrast_candidates(OPEN, similar[2:]), {}, addresses=labels, open_objections=frozenset({"sso"}))
    assert [r.play_code for r in few][0] == "PLAY-OBVIOUS"  # used in only 2 deals here: too little to overrule the label
