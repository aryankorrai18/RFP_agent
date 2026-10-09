"""The free local memory backend (SQLite FTS5). A shared contract runs against the in-memory fakes and the
local classes so they stay interchangeable; the rest covers what is specific to the local backend."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from deal_intelligence.api.v1 import demo, retrieval
from deal_intelligence.api.v1.context import build_context
from deal_intelligence.api.v1.db import parse_deal_code
from deal_intelligence.api.v1.lessons import LessonBrief, LessonHit
from deal_intelligence.api.v1.memory import DEAL_SUMMARY_TAG, MemoryItem
from deal_intelligence.api.v1.memory_local import LOCAL_FOOTER, LocalLessons, LocalMemory, fts_query
from deal_intelligence.config import Settings, resolve_memory_backend

from .fakes import FakeLessons, FakeMemory

HOSTILE_QUERIES = [
    'AND OR "quote (paren', "NOT NEAR/3 * ^ - :", "", "   ", "!!! ??? ...", "a an the of", "col:thing AND (x OR y",
    "SELECT * FROM memory_fts; DROP TABLE x;--", "word " * 5000, "naïve café 日本語",
]


def run(coro):  # noqa: ANN001, ANN202
    return asyncio.run(coro)


@pytest.fixture(params=["fake", "local"])
def memory(request, tmp_path):
    if request.param == "fake":
        return FakeMemory()
    return LocalMemory(tmp_path / "memory.db", "bank-a")


@pytest.fixture(params=["fake", "local"])
def lessons(request, tmp_path):
    if request.param == "fake":
        return FakeLessons()
    return LocalLessons(tmp_path / "memory.db", "lessons-a")


def item(code: str, content: str, *tags: str) -> MemoryItem:
    return MemoryItem(code=code, content=content, tags=list(tags))


def lesson(document_id: str, content: str, *tags: str) -> dict:
    return {"document_id": document_id, "content": content, "tags": list(tags)}


def codes(hits) -> list[str]:  # noqa: ANN001
    return [h.code for h in hits]


# ---- contract: interactions bank ---------------------------------------------------------------------


def test_memory_retain_then_recall(memory):
    run(memory.ensure_bank())
    run(memory.retain(item("D-001", "The customer asked for single sign on and a security review", DEAL_SUMMARY_TAG)))
    hits = run(memory.recall("customer security review", [DEAL_SUMMARY_TAG], 5))
    assert codes(hits) == ["D-001"]
    assert hits[0].rank == 1 and hits[0].final is not None


def test_memory_same_code_replaces_the_old_text(memory):
    run(memory.retain(item("D-001", "alpha bravo charlie", "t")))
    run(memory.retain(item("D-002", "alpha golf hotel", "t")))
    run(memory.retain(item("D-001", "delta echo foxtrot", "t")))
    assert codes(run(memory.recall("alpha", ["t"], 5)))[0] == "D-002"
    assert codes(run(memory.recall("delta", ["t"], 5)))[0] == "D-001"
    assert sorted(codes(run(memory.recall("alpha delta", ["t"], 5)))) == ["D-001", "D-002"]


def test_memory_delete_removes_it(memory):
    run(memory.retain(item("D-001", "alpha bravo", "t")))
    run(memory.delete("D-001"))
    run(memory.delete("D-404"))  # deleting what is not there is fine
    assert run(memory.recall("alpha", ["t"], 5)) == []


def test_memory_tag_filter_is_all_strict(memory):
    run(memory.retain(item("D-001", "alpha bravo", "a", "b")))
    assert codes(run(memory.recall("alpha", ["a"], 5))) == ["D-001"]
    assert codes(run(memory.recall("alpha", ["a", "b"], 5))) == ["D-001"]
    assert run(memory.recall("alpha", ["a", "c"], 5)) == []
    assert run(memory.recall("alpha", ["c"], 5)) == []


@pytest.mark.parametrize("query", HOSTILE_QUERIES)
def test_memory_odd_queries_do_not_raise(memory, query):
    run(memory.retain(item("D-001", "alpha bravo word", "t")))
    assert isinstance(run(memory.recall(query, ["t"], 5)), list)


def test_memory_limit_is_respected(memory):
    for n in range(8):
        run(memory.retain(item(f"D-{n:03}", f"shared words number {n}", "t")))
    assert len(run(memory.recall("shared words", ["t"], 3))) == 3
    assert [h.rank for h in run(memory.recall("shared words", ["t"], 3))] == [1, 2, 3]


def test_memory_ranking_prefers_the_more_relevant_document(memory):
    run(memory.retain(item("D-001", "pricing came up once in a long conversation about onboarding schedules", "t")))
    run(memory.retain(item("D-002", "pricing pricing discount pricing negotiation about the price", "t")))
    run(memory.retain(item("D-003", "completely unrelated holiday party planning", "t")))
    hits = run(memory.recall("pricing discount negotiation", ["t"], 5))
    assert codes(hits)[:2] == ["D-002", "D-001"]


# ---- contract: lessons bank --------------------------------------------------------------------------


def test_lessons_recall_by_any_tag(lessons):
    run(lessons.retain([
        lesson("lesson-1", "sso workshop won the deal", "play:PLAY-01", "signal:positive"),
        lesson("lesson-2", "security pack lost the deal", "play:PLAY-02", "signal:negative"),
        lesson("lesson-3", "pricing walkthrough deal", "play:PLAY-06"),
    ]))
    hits = run(lessons.recall("deal", ["play:PLAY-01", "play:PLAY-02"], 10))
    assert sorted(h.document_id for h in hits) == ["lesson-1", "lesson-2"]
    assert all(isinstance(h, LessonHit) and h.type for h in hits)
    assert run(lessons.recall("deal", ["play:PLAY-99"], 10)) == []
    assert [h.rank for h in hits] == [1, 2]


def test_lessons_replace_by_document_id(lessons):
    run(lessons.retain([lesson("lesson-1", "alpha bravo", "t")]))
    run(lessons.retain([lesson("lesson-1", "delta echo", "t", "u"), lesson("lesson-2", "alpha foxtrot", "t")]))
    assert [h.document_id for h in run(lessons.recall("alpha", ["t"], 10))][0] == "lesson-2"
    hits = run(lessons.recall("delta alpha", ["t"], 10))
    assert sorted(h.document_id for h in hits) == ["lesson-1", "lesson-2"]
    assert run(lessons.recall("delta", ["t"], 10))[0].document_id == "lesson-1"
    assert next(h for h in hits if h.document_id == "lesson-1").tags == ["t", "u"]


@pytest.mark.parametrize("query", HOSTILE_QUERIES)
def test_lessons_odd_queries_do_not_raise(lessons, query):
    run(lessons.retain([lesson("lesson-1", "alpha bravo", "t")]))
    assert isinstance(run(lessons.recall(query, ["t"], 5)), list)


def test_lessons_limit_and_ranking(lessons):
    run(lessons.retain([lesson(f"lesson-{n}", f"deal outcome number {n}", "t") for n in range(6)]
                       + [lesson("lesson-best", "sso sso sso objection resolved", "t")]))
    assert len(run(lessons.recall("deal outcome", ["t"], 2))) == 2
    assert run(lessons.recall("sso objection resolved", ["t"], 5))[0].document_id == "lesson-best"


# ---- local only --------------------------------------------------------------------------------------


def test_query_sanitiser_keeps_content_words_only():
    assert fts_query("") == "" and fts_query("the of and") == "" and fts_query("!!! (") == ""
    assert fts_query('AND OR "quote (paren') == '"quote" OR "paren"'
    assert fts_query("SSO, SSO! Security-review") == '"sso" OR "security" OR "review"'
    assert len(fts_query("word" * 1 + " ".join(f"w{n}" for n in range(500))).split(" OR ")) <= 48


def test_local_memory_basics(tmp_path):
    local = LocalMemory(tmp_path / "memory.db", "b")
    assert run(local.extraction_mode()) == "local"
    assert run(local.healthy()) is True
    run(local.close())


def test_memory_persists_across_instances(tmp_path):
    path = tmp_path / "memory.db"
    run(LocalMemory(path, "b").retain(item("D-001", "alpha bravo", "t")))
    run(LocalLessons(path, "l").retain([lesson("lesson-1", "alpha bravo", "t")]))
    assert codes(run(LocalMemory(path, "b").recall("alpha", ["t"], 5))) == ["D-001"]
    assert [h.document_id for h in run(LocalLessons(path, "l").recall("alpha", ["t"], 5))] == ["lesson-1"]


def test_banks_in_one_file_do_not_leak(tmp_path):
    path = tmp_path / "memory.db"
    one, two = LocalMemory(path, "deal-interactions-one"), LocalMemory(path, "deal-interactions-two")
    run(one.retain(item("D-001", "alpha bravo", "t")))
    run(two.retain(item("D-002", "alpha bravo", "t")))
    assert codes(run(one.recall("alpha", ["t"], 5))) == ["D-001"]
    assert codes(run(two.recall("alpha", ["t"], 5))) == ["D-002"]
    run(one.delete("D-002"))  # the other bank's document of the same name is untouched
    assert codes(run(two.recall("alpha", ["t"], 5))) == ["D-002"]

    lessons_one, lessons_two = LocalLessons(path, "l-one"), LocalLessons(path, "l-two")
    run(lessons_one.retain([lesson("lesson-1", "alpha", "t")]))
    assert run(lessons_two.recall("alpha", ["t"], 5)) == []
    assert run(lessons_two.reflect("alpha", ["t"])).based_on == []


def test_two_folders_do_not_leak(tmp_path):
    first, second = LocalMemory(tmp_path / "w1" / "memory.db", "b"), LocalMemory(tmp_path / "w2" / "memory.db", "b")
    run(first.retain(item("D-001", "alpha bravo", "t")))
    assert run(second.recall("alpha", ["t"], 5)) == []


def test_folder_can_be_deleted_after_use(tmp_path):
    import shutil

    local = LocalMemory(tmp_path / "w1" / "memory.db", "b")
    run(local.retain(item("D-001", "alpha", "t")))
    shutil.rmtree(tmp_path / "w1")
    assert not (tmp_path / "w1").exists()


OUTCOMES = [
    lesson("lesson-outcome:D-001", "Deal D-001, a mid-market fintech deal, was won. An SSO objection was addressed.",
           "kind:deal_outcome", "signal:positive", "deal:D-001", "result:won", "objection:sso", "industry:fintech"),
    lesson("lesson-outcome:D-002", "Deal D-002, a mid-market fintech deal, was lost because of security and compliance. "
           "An SSO objection was left unresolved.",
           "kind:deal_outcome", "signal:negative", "deal:D-002", "result:lost", "objection:sso", "industry:fintech",
           "loss:security_compliance"),
    lesson("lesson-outcome:D-003", "Deal D-003, a mid-market retail deal, was lost because of price.",
           "kind:deal_outcome", "signal:neutral", "deal:D-003", "result:lost", "industry:retail", "loss:price"),
    lesson("lesson-play:D-001:PLAY-01", "Play PLAY-01 (SSO integration workshop) was used in deal D-001, a fintech deal. "
           "The deal was won, which counts in favour of the play.",
           "kind:play_result", "signal:positive", "deal:D-001", "play:PLAY-01", "objection:sso"),
    lesson("lesson-play:D-002:PLAY-01", "Play PLAY-01 (SSO integration workshop) was used in deal D-002, a fintech deal. "
           "The deal was lost, which counts against the play.",
           "kind:play_result", "signal:negative", "deal:D-002", "play:PLAY-01", "objection:sso"),
]


def test_reflect_is_deterministic_and_states_counts(tmp_path):
    local = LocalLessons(tmp_path / "memory.db", "l")
    run(local.retain(OUTCOMES))
    first = run(local.reflect("an sso objection in a fintech deal", ["objection:sso"]))
    second = run(LocalLessons(tmp_path / "memory.db", "l").reflect("an sso objection in a fintech deal", ["objection:sso"]))
    assert isinstance(first, LessonBrief) and first == second
    assert "2 recorded outcomes" in first.text and "1 won" in first.text and "1 lost for a reason a play could have changed" in first.text
    assert first.text.endswith("This was summarised locally without a model.")
    assert 1 <= len(first.based_on) <= 3
    assert set(first.based_on[0]) == {"id", "text", "type"}
    assert all(row["id"].startswith("lesson-") for row in first.based_on)


def test_reflect_without_matches_is_honest(tmp_path):
    local = LocalLessons(tmp_path / "memory.db", "l")
    brief = run(local.reflect("anything", ["objection:sso"]))
    assert brief.based_on == [] and "No recorded outcomes" in brief.text and "locally without a model" in brief.text


def test_playbook_shape_counts_and_footer(tmp_path):
    local = LocalLessons(tmp_path / "memory.db", "l")
    run(local.retain(OUTCOMES))
    book = run(local.playbook())
    assert set(book) == {"content", "last_refreshed_at", "is_stale"} and book["is_stale"] is False
    assert datetime_ok(book["last_refreshed_at"])
    content = book["content"]
    assert content.rstrip().endswith("Computed locally from recorded outcomes, not by Hindsight.") and LOCAL_FOOTER in content
    assert "3 closed deals: 1 won, 1 lost for a reason a play could have changed, 1 with no verdict" in content
    assert "PLAY-01 (SSO integration workshop): 1 won, 1 lost (fixable), 0 no verdict (2 deals)" in content
    assert "- sso: 1 won, 1 lost (fixable), 0 no verdict (2 deals)" in content
    assert "## Loss reasons" in content and "- price: 0 won, 0 lost (fixable), 1 no verdict (1 deal)" in content
    assert run(local.playbook(refresh=True))["content"] == content


def test_playbook_with_nothing_recorded(tmp_path):
    book = run(LocalLessons(tmp_path / "memory.db", "l").playbook())
    assert "No closed deals" in book["content"] and LOCAL_FOOTER in book["content"]


def datetime_ok(value: str) -> bool:
    from datetime import datetime

    return datetime.fromisoformat(value) is not None


# ---- backend choice ----------------------------------------------------------------------------------

CLOUD = "https://api.hindsight.vectorize.io"
LOCAL_URL = "http://127.0.0.1:8888"


@pytest.mark.parametrize(("backend", "url", "key", "expected"), [
    ("auto", CLOUD, None, "local"),
    ("auto", CLOUD, "hs-key", "hindsight"),
    ("auto", LOCAL_URL, None, "hindsight"),
    ("auto", LOCAL_URL, "hs-key", "hindsight"),
    ("local", CLOUD, "hs-key", "local"),
    ("hindsight", CLOUD, None, "hindsight"),
    ("local", LOCAL_URL, None, "local"),
])
def test_resolve_memory_backend(backend, url, key, expected):
    settings = replace(Settings(), memory_backend=backend, hindsight_url=url, hindsight_api_key=key)
    assert resolve_memory_backend(settings) == expected


def test_resolve_memory_backend_rejects_unknown_values():
    with pytest.raises(ValueError, match="DEAL_MEMORY_BACKEND"):
        resolve_memory_backend(replace(Settings(), memory_backend="redis"))


def test_env_var_is_read_and_validated(monkeypatch):
    monkeypatch.setenv("DEAL_MEMORY_BACKEND", "Local")
    assert Settings.from_env().memory_backend == "local"
    monkeypatch.setenv("DEAL_MEMORY_BACKEND", "bogus")
    with pytest.raises(ValueError, match="DEAL_MEMORY_BACKEND"):
        Settings.from_env()
    monkeypatch.delenv("DEAL_MEMORY_BACKEND")
    assert Settings.from_env().memory_backend == "auto"


# ---- the whole app on the local backend, with no network ---------------------------------------------


def test_build_context_uses_local_memory_and_recommends_from_the_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("DEAL_MEMORY_BACKEND", "local")
    settings = replace(Settings.from_env(), db_path=tmp_path / "ws" / "deals.db", uploads_dir=tmp_path / "ws" / "up",
                       memory_backend="local")
    ctx = build_context(lambda: settings, lambda _s: None)
    assert ctx.memory_backend == "local"
    assert isinstance(ctx.memory, LocalMemory) and isinstance(ctx.lessons, LocalLessons)

    async def scenario():
        seeded = demo.seed_demo(ctx)
        await ctx.sync()
        rec = await retrieval.build_recommendations(ctx, _deal_id(ctx, seeded["demo_deal"]), "hindsight")
        pending = ctx.last_lesson_sync.pending
        return rec, pending, await ctx.lessons.playbook()

    rec, pending, playbook = run(scenario())
    assert (tmp_path / "ws" / "memory.db").exists()
    assert pending == 0
    assert rec.degraded is None
    assert rec.similar, "similar closed deals should come back from local recall"
    assert rec.plays, "plays should be ranked from the similar deals"
    assert rec.lessons_used is True
    assert all(s.relevance > 0 for s in rec.similar if s.rank < retrieval.UNSYNCED_RANK)
    assert "closed deals" in playbook["content"] and LOCAL_FOOTER in playbook["content"]
    run(ctx.shutdown())


def test_build_context_leaves_lessons_off_when_disabled(tmp_path):
    settings = replace(Settings(), db_path=tmp_path / "deals.db", memory_backend="local", lessons_enabled=False)
    ctx = build_context(lambda: settings, lambda _s: None)
    assert isinstance(ctx.memory, LocalMemory) and ctx.lessons is None


def test_build_context_keeps_hindsight_for_a_keyed_cloud(tmp_path):
    from deal_intelligence.api.v1.memory import HindsightMemory

    settings = replace(Settings(), db_path=tmp_path / "deals.db", hindsight_url=CLOUD, hindsight_api_key="k")
    ctx = build_context(lambda: settings, lambda _s: None)
    assert ctx.memory_backend == "hindsight" and isinstance(ctx.memory, HindsightMemory)


def _deal_id(ctx, code: str) -> int:  # noqa: ANN001
    return parse_deal_code(code)


def test_status_reports_the_local_backend_without_calling_hindsight(tmp_path):
    from deal_intelligence.api.v1.router import status

    settings = replace(Settings(), db_path=tmp_path / "deals.db", memory_backend="local", hindsight_url=CLOUD)
    ctx = build_context(lambda: settings, lambda _s: None)
    block = run(status(ctx))["hindsight"]
    assert block["backend"] == "local" and block["healthy"] is True and block["mode"] == "local"
    assert block["cloud"] is False and block["api_key_set"] is False
    assert {"url", "bank", "lessons_bank"} <= set(block)
