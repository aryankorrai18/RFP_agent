"""The evaluation harness itself: the generated world, the marking, the local memory, the statistics and the report.
Offline and quick."""

from __future__ import annotations

import asyncio
import json

import pytest

from evaluation import charts, draft_eval, harness, report, retrieval_eval, stats
from evaluation.memory import LexicalLessons, LexicalMemory
from evaluation.world import KIND_COUNTS, KINDS, PROPOSALS, REVIEW_PLAN, TOPICS, Question, make_world
from rfp_assistant.api.v1.lessons import REVIEW
from rfp_assistant.api.v1.memory import MemoryItem


def run(coro):
    return asyncio.run(coro)


# -- the world ----------------------------------------------------------------------------------------------

def test_the_same_seed_makes_the_same_world_and_another_seed_a_different_one():
    a, b, c = make_world(5), make_world(5), make_world(6)
    assert a.questions == b.questions and a.pairs == b.pairs and a.questions != c.questions


def test_a_world_has_every_kind_of_question_the_requested_number_of_times_on_distinct_topics():
    world = make_world(1)
    assert len(world.questions) == sum(KIND_COUNTS.values())
    for kind in KINDS:
        assert sum(1 for q in world.questions if q.kind == kind) == KIND_COUNTS[kind]
    assert len({q.topic for q in world.questions}) == len(world.questions)
    assert len({q.id for q in world.questions}) == len(world.questions)
    assert sum(KIND_COUNTS.values()) <= len(TOPICS)


def test_the_planted_rules_are_consistent():
    topics = {t.key: t for t in TOPICS}
    for seed in range(1, 8):
        world = make_world(seed)
        facts = {f.id: f for f in world.facts}
        for q in world.questions:
            if q.kind == "unanswerable":
                assert q.gold_value is None and q.gold_ref is None and q.gold_fact is None
                continue
            assert q.gold_value in topics[q.topic].values
            assert all(v in topics[q.topic].values and v != q.gold_value for v in q.forbidden)
            if q.gold_ref:  # the right answer is in the library with the right value, in a proposal that was won
                proposal, topic = q.gold_ref
                assert PROPOSALS[proposal][3] == "won"
                assert any(t == topic and q.gold_value in text for t, _q, text in world.pairs[proposal])
            if q.bad_ref:
                proposal, topic = q.bad_ref
                assert any(t == topic and q.forbidden[0] in text for t, _q, text in world.pairs[proposal])
            if q.gold_fact:
                assert q.gold_value in facts[q.gold_fact].statement
        assert all(PROPOSALS[q.bad_ref[0]][3] == "lost" for q in world.questions if q.kind == "lost_trap")
        stale = [q for q in world.questions if q.kind == "stale"]
        assert all(PROPOSALS[q.bad_ref[0]][2] < PROPOSALS[q.gold_ref[0]][2] for q in stale)  # the wrong one is the older one


def test_values_inside_one_topic_are_not_confusable_with_each_other():
    for topic in TOPICS:
        for a in topic.values:
            for b in topic.values:
                if a != b:
                    assert not harness.value_in(topic.answer.format(v=a), b), (topic.key, a, b)


def test_review_history_has_the_same_shape_the_products_review_path_writes():
    world = make_world(2)
    events = world.history()
    assert len(events) == len(REVIEW_PLAN) * KIND_COUNTS["reviewed"] and len(world.history(2)) == 2 * KIND_COUNTS["reviewed"]
    refs = {r for q in world.questions for r in (q.gold_ref, q.bad_ref)}
    for ref, action, tags in events:
        detail = f"ANS-0001 was {action}" + (f" ({', '.join(tags)})" if tags else "") + "."
        match = REVIEW.search(detail)  # the lessons collector must be able to read it back
        assert match and match.group(1) == action and ref in refs


# -- marking --------------------------------------------------------------------------------------------------

def _question(**kw):
    base = dict(id="q", kind="stale", topic="uptime", text="t", project="neutral", gold_value="99.9%", forbidden=["99.5%"], gold_ref=("a", "uptime"))
    return Question(**(base | kw))


def test_a_value_is_found_only_as_a_whole_phrase():
    assert harness.value_in("We commit to 99.9% uptime.", "99.9%") and not harness.value_in("We commit to 99.99% uptime.", "99.9%")
    assert harness.value_in("a SOC 2 Type II report", "Type II") and not harness.value_in("a SOC 2 Type II report", "Type I")
    assert harness.value_in("within 24 hours.", "24 hours") and not harness.value_in("within 124 hours", "24 hours")


def test_marking_follows_the_planted_rules():
    q = _question()
    good = {"status": "drafted", "answer": "We commit to 99.9% uptime.", "sources": ["ANS-0002"], "flags": [], "word_count": 6}
    ok = harness.mark_draft(q, good, "99.9% uptime 99.5%", "ANS-0002")
    assert ok["correct"] and not ok["stale_value"] and ok["cites_gold"]
    stale = harness.mark_draft(q, good | {"answer": "We commit to 99.5% uptime."}, "99.9% 99.5%", "ANS-0002")
    assert not stale["correct"] and stale["stale_value"] and stale["wrong_value"]
    both = harness.mark_draft(q, good | {"answer": "99.9% (formerly 99.5%)"}, "", "ANS-0002")
    assert not both["correct"]  # stating the wrong value next to the right one is still wrong
    uncited = harness.mark_draft(q, good | {"sources": ["ANS-0001"]}, "", "ANS-0002")
    assert not uncited["correct"] and not uncited["cites_gold"]
    assert not harness.mark_draft(q, good | {"flags": ["invalid_citation"]}, "", "ANS-0002")["correct"]
    handoff = {"status": "needs_sme", "answer": "SME COMPLETION TEMPLATE", "sources": [], "flags": [], "word_count": 3}
    abstained = harness.mark_draft(q, handoff, "", "ANS-0002")
    assert abstained["abstained"] and not abstained["correct"] and not abstained["wrong_value"]


def test_an_unanswerable_question_is_right_only_when_handed_to_an_expert_without_inventing_numbers():
    q = _question(kind="unanswerable", gold_value=None, forbidden=[], gold_ref=None)
    handed = {"status": "needs_sme", "answer": "SME COMPLETION TEMPLATE - fill in", "sources": [], "flags": [], "word_count": 5}
    assert harness.mark_draft(q, handed, "", None)["correct"]
    invented = {"status": "drafted", "answer": "It is 42 minutes.", "sources": [], "flags": [], "word_count": 4}
    marks = harness.mark_draft(q, invented, "nothing numeric here", None)
    assert not marks["correct"] and marks["fabricated"] and marks["invented_number"]
    assert not harness.mark_draft(q, invented | {"answer": "It is 15 minutes."}, "the record says 15 minutes", None)["invented_number"]


def test_retrieval_marking_counts_the_right_and_wrong_answers_in_the_top_three():
    world = make_world(1)
    q = next(q for q in world.questions if q.kind == "stale")
    refs = {q.gold_ref: "ANS-0010", q.bad_ref: "ANS-0011"}
    first = harness.mark_retrieval(q, ["ANS-0010", "ANS-0011"], refs)
    assert first["hit1"] and first["hit3"] and first["rr"] == 1.0 and not first["bad1"] and first["bad_shown"]
    second = harness.mark_retrieval(q, ["ANS-0011", "ANS-0010", "ANS-0001"], refs)
    assert not second["hit1"] and second["hit3"] and second["rr"] == 0.5 and second["bad1"]
    assert harness.mark_retrieval(q, ["ANS-0001", "ANS-0002", "ANS-0003", "ANS-0010"], refs)["hit3"] is False  # only the top three count
    assert harness.mark_retrieval(q, [], refs)["empty"]


# -- the local memory banks ----------------------------------------------------------------------------------

def test_the_lexical_memory_ranks_by_rare_word_overlap_and_breaks_ties_by_code():
    memory = LexicalMemory()
    for code, question, answer in (("ANS-0002", "How long is retention?", "Retained for 90 days."),
                                   ("ANS-0001", "How long is retention?", "Retained for 30 days."),
                                   ("ANS-0003", "Which encryption standard?", "AES-256.")):
        run(memory.retain(MemoryItem(code, question, answer, None, None, None)))
    hits = run(memory.recall("What retention period applies?", 3))
    assert [h.answer_code for h in hits][:2] == ["ANS-0001", "ANS-0002"] and hits[0].final == hits[1].final > hits[2].final
    run(memory.delete("ANS-0001"))
    assert "ANS-0001" not in [h.answer_code for h in run(memory.recall("retention", 5))]


def test_the_lexical_lessons_bank_recalls_only_what_carries_a_wanted_tag():
    lessons = LexicalLessons()
    run(lessons.retain([{"document_id": "a", "content": "Answer ANS-0001 was rewritten", "tags": ["answer:ANS-0001"]},
                        {"document_id": "b", "content": "Answer ANS-0002 was accepted", "tags": ["answer:ANS-0002"]}]))
    assert [h.document_id for h in run(lessons.recall("answer", ["answer:ANS-0002"]))] == ["b"]
    assert run(lessons.recall("answer", ["answer:ANS-0009"])) == []


# -- the product is run on the world ---------------------------------------------------------------------------

def test_a_world_loads_into_the_product_and_its_retrieval_answers_every_library_question():
    rows = run(retrieval_eval.evaluate_world(1))
    assert len(rows) == 16 * len(retrieval_eval.SYSTEMS) and {r["system"] for r in rows} == set(retrieval_eval.SYSTEMS)
    assert any(r["ranked"] for r in rows if r["system"] == "outcome")


def test_review_history_changes_the_rankings_the_way_the_product_says_it_should():
    cold = run(retrieval_eval.evaluate_world(1, history=0, systems=("plain", "outcome")))
    warm = run(retrieval_eval.evaluate_world(1, history=5, systems=("plain", "outcome")))

    def rate(rows, system):
        rs = [r for r in rows if r["system"] == system and r["kind"] == "reviewed"]
        return sum(r["hit1"] for r in rs) / len(rs)

    assert rate(warm, "outcome") > rate(cold, "outcome") and rate(warm, "outcome") > rate(warm, "plain")
    assert rate(cold, "outcome") == rate(cold, "plain")  # nothing to learn from yet: memory changes nothing


def test_the_context_diagnostic_arithmetic_and_wrapper_leave_the_product_untouched():
    from evaluation import context_diagnostic as cd
    from rfp_assistant.api.v1 import ranking, retrieval

    table = {a["position"]: a for a in cd.arithmetic()}
    assert table[1]["bonus_needed"] == 2.0 and not table[1]["enough"] and not table[6]["enough"] and table[7]["enough"]
    try:
        cd.set_power(4.0)
        assert retrieval.rank_factors is not ranking.rank_factors
    finally:
        retrieval.rank_factors = ranking.rank_factors
    assert retrieval.rank_factors is ranking.rank_factors


def test_the_flat_diagnostic_can_be_switched_on_and_off():
    from rfp_assistant.api.v1 import ranking, retrieval

    try:
        retrieval_eval.set_variant("flat")
        assert retrieval.order_candidates is not ranking.order_candidates
    finally:
        retrieval_eval.set_variant("default")
    assert retrieval.order_candidates is ranking.order_candidates


def test_the_draft_evaluation_runs_the_products_drafting_and_marks_each_draft(tmp_path):
    out = tmp_path / "drafts.jsonl"
    written = run(draft_eval.run_seed(1, ("none", "outcome"), draft_eval.ScriptedLLM(), None, out, set(), 2, say=lambda _m: None))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert written == len(rows) == 2 * sum(KIND_COUNTS.values()) and {r["arm"] for r in rows} == {"none", "outcome"}
    none = [r for r in rows if r["arm"] == "none"]
    assert all(r["shown"] == [] for r in none)  # memory off: no past answers reach the model
    assert all(r["correct"] for r in none if r["kind"] == "unanswerable")  # nothing to say, so it is handed to an expert
    assert not any(r["correct"] for r in none if r["kind"] == "stale")
    assert all("input_tokens" in r and r["model"] == "scripted" for r in rows)
    assert any(r["correct"] for r in rows if r["arm"] == "outcome" and r["kind"] == "stale")


class QuotaGoneLLM(draft_eval.ScriptedLLM):
    async def draft_answer(self, *args, **kwargs):  # noqa: ANN002, ANN003
        from rfp_assistant.providers.base import LLMError

        raise LLMError("api_error", "drafting: Gemini quota used up for the model (You exceeded your current quota)")


def test_a_run_stops_at_the_first_quota_failure_and_writes_nothing_for_it(tmp_path):
    from rfp_assistant.providers.base import LLMError

    out = tmp_path / "drafts.jsonl"
    with pytest.raises(LLMError, match="stopped after"):
        run(draft_eval.run_seed(1, ("none",), QuotaGoneLLM(), None, out, set(), 1, say=lambda _m: None))
    assert not out.exists() or out.read_text(encoding="utf-8").strip() == ""


def test_the_canary_passes_a_model_that_uses_relevant_answers_and_fails_one_that_never_does(tmp_path):
    assert run(draft_eval.canary(draft_eval.ScriptedLLM(), None, 1, say=lambda _m: None))

    class Refuses(draft_eval.ScriptedLLM):
        async def draft_answer(self, company, facts, requirement, past_answers=None, instructions=None, *, temperature=None):  # noqa: ANN001, ANN201
            return await super().draft_answer(company, facts, requirement, None, instructions)

    assert not run(draft_eval.canary(Refuses(), None, 1, say=lambda _m: None))


def test_the_draft_estimate_counts_its_calls_without_any_model():
    estimate = run(draft_eval.estimate(1, ("none", "plain"), first_seed=1))
    assert estimate["calls"] == 2 * sum(KIND_COUNTS.values()) and estimate["input_tokens"] > 0 and estimate["usd_upper_bound"] > 0


def test_resuming_skips_what_is_already_written(tmp_path):
    path = tmp_path / "drafts.jsonl"
    path.write_text(json.dumps({"seed": 1, "question": "a", "arm": "none", "status": "drafted"}) + "\n"
                    + json.dumps({"seed": 1, "question": "b", "arm": "none", "status": "failed"}) + "\n", encoding="utf-8")
    assert draft_eval.load_done(path) == {(1, "a", "none")}  # a failed draft is tried again


# -- statistics, charts and the report ----------------------------------------------------------------------------

def test_wilson_interval_and_sign_test_behave():
    rate, low, high = stats.wilson(50, 100)
    assert rate == 0.5 and 0.40 < low < 0.41 and 0.59 < high < 0.60 and stats.wilson(0, 0) == (0.0, 0.0, 0.0)
    assert stats.sign_test([True] * 8 + [False] * 2, [False] * 10) == (8, 0, pytest.approx(2 * 0.5**8))
    assert stats.sign_test([True, False], [False, True])[2] == 1.0


def test_charts_are_wellformed_svg_and_escape_their_text():
    svg = charts.bar_chart("A <b> chart", "sub", ["g1"], {"One & two": [(0.5, 0.4, 0.6)]})
    assert svg.startswith("<svg") and svg.endswith("</svg>") and "&lt;b&gt;" in svg and "One &amp; two" in svg


def test_the_report_is_built_from_result_files_and_says_it_is_synthetic(tmp_path, monkeypatch):
    data = run(retrieval_eval.run([1], curve=False, say=lambda _m: None))
    data["curve_rows"] = [r | {"history": h} for h in data["design"]["curve_steps"] for r in data["rows"] if r["system"] in ("plain", "outcome", "hindsight")]
    (tmp_path / "heldout.json").write_text(json.dumps(data), encoding="utf-8")
    drafts = tmp_path / "x.jsonl"
    run(draft_eval.run_seed(1, ("none", "plain", "outcome"), draft_eval.ScriptedLLM(), None, drafts, set(), 2, say=lambda _m: None))
    (tmp_path / "drafts.jsonl").write_text(drafts.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    report.main()
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "synthetic" in text.lower() and "Layer 1" in text and "Layer 2" in text and "lexical" in text.lower()
    for name in ("retrieval_first.svg", "drafts_correct.svg", "learning_curve.svg"):
        assert (tmp_path / name).exists()
