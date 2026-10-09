"""Memory may change the preference among relevant answers; it must not turn an irrelevant one into
the top answer. Offline."""

from __future__ import annotations

import asyncio
import random

from rfp_assistant.api.v1.ranking import Scored, order_candidates
from rfp_assistant.api.v1.retrieval import retrieve
from tests.test_hindsight_lessons import library_with_competing_answers
from tests.v1_fakes import FakeLessons, FakeV1LLM, make_context, pair


def s(name, score, final, rank, relevance=None):
    return Scored(name, score, relevance if relevance is not None else 1.0 / rank, final, rank)


def test_rank_policy_is_the_pre_fix_rule():
    items = [s("relevant", 0.4, 1.0, 1), s("off-topic", 0.9, 0.001, 2)]
    assert order_candidates(items, policy="rank") == ["off-topic", "relevant"]


def test_gated_policy_keeps_an_off_topic_answer_below_the_relevant_group():
    items = [s("scim-lost", 0.3, 1.065, 1), s("scim-won", 0.9, 0.031, 2), s("mfa", 5.0, 0.001, 3)]
    # scim-won is in the first group (0.031 >= 1% of 1.065), so its lessons can lift it; mfa isn't.
    assert order_candidates(items, policy="gated", min_share=0.01) == ["scim-won", "scim-lost", "mfa"]


def test_gated_order_never_lets_a_later_group_overtake_an_earlier_one():
    rng = random.Random(7)
    for _ in range(500):
        n = rng.randint(2, 12)
        finals = sorted((rng.random() ** 3 for _ in range(n)), reverse=True)
        items = [s(i, rng.uniform(0, 3), finals[i], i + 1) for i in range(n)]
        order = order_candidates(items, policy="gated", min_share=0.01)
        assert sorted(order) == list(range(n))
        # Rebuild the groups and check every item of an earlier group comes before any later one.
        remaining, groups = list(items), []
        while remaining:
            best = max(x.final for x in remaining)
            group = [x.item for x in remaining if x.final >= 0.01 * best]
            groups.append(group)
            remaining = [x for x in remaining if x.item not in group]
        position = {item: i for i, item in enumerate(order)}
        for earlier, later in zip(groups, groups[1:]):
            assert max(position[i] for i in earlier) < min(position[i] for i in later)


def test_a_strong_lesson_cannot_lift_an_off_topic_answer_to_the_top(tmp_path):
    lessons = FakeLessons()
    llm = FakeV1LLM()
    ctx = make_context(tmp_path, llm, lessons=lessons)
    question = "Describe your penetration testing practices."

    async def scenario():
        vague, specific = await library_with_competing_answers(ctx, llm, lost_on="2025-06-01", won_on="2025-06-01")
        # An unrelated answer from a won proposal, praised by many lessons.
        from datetime import date

        from tests.test_hindsight_lessons import import_proposal

        [off_topic] = await import_proposal(ctx, llm, [pair("What is your refund policy for annual plans?",
                                                            "Refunds are prorated for annual plans.")],
                                            client="Other", industry="retail", submitted_on=date(2025, 6, 1),
                                            result="won", marker="c")
        await ctx.sync()
        for i in range(6):
            lessons.items.append({"content": f"Reviewers praised {off_topic.code} for penetration testing practices.",
                                  "tags": ["signal:positive", f"answer:{off_topic.code}"], "document_id": f"extra-{i}"})
        results = {}
        for policy in ("rank", "gated"):
            found = await retrieve(question, memory=ctx.memory, db=ctx.db, k=3, mode="hindsight", lessons=lessons,
                                   relevance=policy, min_share=0.01)
            results[policy] = [p.id for p in found.past_answers]
        return off_topic.code, specific, results

    off_topic, specific, results = asyncio.run(scenario())
    assert results["rank"][0] == off_topic  # the pre-fix rule lets six lessons lift an unrelated answer to the top
    assert results["gated"][0] == specific  # memory still prefers the won, specific answer among relevant ones
    assert results["gated"].index(off_topic) > results["gated"].index(specific)
