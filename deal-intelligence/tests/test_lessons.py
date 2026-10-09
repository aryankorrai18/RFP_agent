"""Outcome lessons: built deterministically from closed deals, retained in batches, recalled as per-play
evidence, and the HindsightLessons call shapes. Offline, with fakes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from hindsight_client_api.exceptions import ApiException
from sqlalchemy import select

from deal_intelligence.api.v1.db import Deal, Lesson
from deal_intelligence.api.v1.lessons import (
    BANK_MISSION, PLAYBOOK_ID, PLAYBOOK_QUERY, REFLECT_MISSION, HindsightLessons, LessonHit, collect_lessons,
    deal_type_tags, pending_lessons, play_signals, reflect_deal_type, sync_lessons,
)
from deal_intelligence.api.v1.memory import MemoryUnavailable
from deal_intelligence.api.v1.ranking import deal_facts

from .builders import add_deal, make_context
from .fakes import BUYER, ENGAGED, SILENT, FakeLessons, add_catalogue, closed


def lessons_by_key(ctx):
    with ctx.db.session() as s:
        return {row.key: row for row in s.scalars(select(Lesson))}


def test_outcomes_become_tagged_lessons_with_the_signal_rules(tmp_path):
    ctx = make_context(tmp_path)
    add_catalogue(ctx.db)
    closed(ctx.db, "Won", "A", "won", objections=[("sso", "addressed")], plays=["PLAY-01"], stakeholders=[ENGAGED])
    closed(ctx.db, "Security loss", "B", "lost", loss_reason="security_compliance", objections=[("sso", "unresolved")],
           competitors=["Brightline"], plays=["PLAY-02"], stakeholders=[SILENT, BUYER])
    closed(ctx.db, "Price loss", "C", "lost", loss_reason="price", objections=[("pricing", "unresolved")], plays=["PLAY-03"])
    add_deal(ctx.db, name="Open", account="D", plays=["PLAY-04"])  # open deals teach nothing yet

    assert collect_lessons(ctx.db) == 6
    rows = lessons_by_key(ctx)
    assert sorted(rows) == ["outcome:D-001", "outcome:D-002", "outcome:D-003",
                            "play:D-001:PLAY-01", "play:D-002:PLAY-02", "play:D-003:PLAY-03"]
    assert rows["outcome:D-001"].signal == "positive"
    assert rows["outcome:D-002"].signal == "negative"  # a quality loss: the plays could have changed it
    assert rows["outcome:D-003"].signal == "neutral"  # price: no verdict on the plays
    assert rows["play:D-003:PLAY-03"].signal == "neutral" and rows["play:D-002:PLAY-02"].signal == "negative"

    lost = rows["outcome:D-002"]
    assert set(lost.tags) == {"kind:deal_outcome", "signal:negative", "deal:D-002", "result:lost", "industry:fintech",
                              "segment:mid_market", "objection:sso", "competitor:brightline", "loss:security_compliance"}
    assert lost.deal_id == 2 and lost.play_code is None and lost.hindsight_status == "pending"
    assert lost.text.startswith("Deal D-002, a mid-market fintech deal, was lost because of security and compliance.")
    assert "An SSO objection stayed unresolved." in lost.text and "Plays used: PLAY-02 (Security review pack)." in lost.text
    play = rows["play:D-002:PLAY-02"]
    assert {"kind:play_result", "play:PLAY-02", "deal:D-002", "signal:negative"} <= set(play.tags)
    assert play.play_code == "PLAY-02" and "counts against the play" in play.text
    assert "says little about the play" in rows["play:D-003:PLAY-03"].text and "counts in favour" in rows["play:D-001:PLAY-01"].text


def test_collection_is_idempotent_and_follows_a_re_recorded_outcome(tmp_path):
    ctx = make_context(tmp_path)
    add_catalogue(ctx.db)
    deal_id = closed(ctx.db, "Deal", "A", "won", plays=["PLAY-01", "PLAY-02"])
    assert collect_lessons(ctx.db) == 3
    assert collect_lessons(ctx.db) == 0
    with ctx.db.session() as s:  # pretend they were all sent
        for row in s.scalars(select(Lesson)):
            row.hindsight_status = "retained"
        deal = s.get(Deal, deal_id)
        deal.result, deal.loss_reason = "lost", "security_compliance"
        deal.signals.plays_used = ["PLAY-01"]
        s.commit()
    assert collect_lessons(ctx.db) == 3  # outcome and play:PLAY-01 rewritten, play:PLAY-02 removed
    rows = lessons_by_key(ctx)
    assert sorted(rows) == ["outcome:D-001", "play:D-001:PLAY-01"]
    assert rows["outcome:D-001"].signal == "negative" and rows["outcome:D-001"].hindsight_status == "pending"
    assert collect_lessons(ctx.db) == 0


def test_lessons_of_a_deleted_deal_are_dropped(tmp_path):
    ctx = make_context(tmp_path)
    deal_id = closed(ctx.db, "Deal", "A", "won", plays=["PLAY-01"])
    collect_lessons(ctx.db)
    with ctx.db.session() as s:
        s.get(Deal, deal_id).status = "deleted"
        s.commit()
    assert collect_lessons(ctx.db) == 2 and lessons_by_key(ctx) == {}


def test_sync_retains_in_batches_of_twenty_with_idempotent_document_ids(tmp_path):
    lessons = FakeLessons()
    ctx = make_context(tmp_path, lessons=lessons)
    for n in range(12):
        closed(ctx.db, f"Deal {n}", f"Account {n}", "won", plays=[f"PLAY-{n:02d}"])
    report = asyncio.run(sync_lessons(ctx.db, lessons, ctx.lessons_lock))
    assert (report.collected, report.retained, report.failed, report.pending) == (24, 24, 0, 0)
    assert lessons.batches == [20, 4]
    item = next(i for i in lessons.items if i["document_id"] == "lesson-play:D-001:PLAY-00")
    assert item["tags"] and item["metadata"] == {"lesson_key": "play:D-001:PLAY-00", "signal": "positive"}
    asyncio.run(sync_lessons(ctx.db, lessons, ctx.lessons_lock))
    assert len(lessons.items) == 24 and lessons.batches == [20, 4]  # nothing new, nothing resent


def test_lesson_outage_keeps_rows_pending_and_the_next_sync_catches_up(tmp_path):
    lessons = FakeLessons()
    ctx = make_context(tmp_path, lessons=lessons)
    closed(ctx.db, "Deal", "A", "won", plays=["PLAY-01"])
    lessons.available = False
    report = asyncio.run(sync_lessons(ctx.db, lessons, ctx.lessons_lock))
    assert (report.retained, report.failed, report.pending) == (0, 2, 2) and "down" in report.errors[0]
    assert all(row.hindsight_attempts == 1 and row.hindsight_error for row in lessons_by_key(ctx).values())
    lessons.available = True
    report = asyncio.run(sync_lessons(ctx.db, lessons, ctx.lessons_lock))
    assert (report.retained, report.pending) == (2, 0) and pending_lessons(ctx.db) == 0


# ---- recalled lessons as per-play evidence ---------------------------------------------------------------------


def hit(doc, tags, rank=1, text=None):
    return LessonHit(text=text or doc, tags=tags, document_id=doc, type="world", rank=rank)


def test_play_signals_net_positive_and_negative_evidence_once_per_lesson():
    hits = [
        hit("lesson-a", ["play:PLAY-01", "signal:positive", "deal:D-001"], rank=1),
        hit("lesson-a", ["play:PLAY-01", "signal:positive", "deal:D-001"], rank=2),  # same lesson, second fact
        hit("lesson-b", ["play:PLAY-01", "signal:negative", "deal:D-002"], rank=3),
        hit("lesson-c", ["play:PLAY-02", "signal:neutral", "deal:D-003"], rank=4),
        hit("lesson-d", ["play:PLAY-09", "signal:positive"], rank=5),  # not a candidate
    ]
    out = play_signals(hits, {"PLAY-01", "PLAY-02"})
    one = out["PLAY-01"]
    assert (one.positive, one.negative) == (1, 1) and one.evidence == ["lesson-a", "lesson-b"]
    assert 0 < one.net < 1  # +1.0 for rank 1, minus about 0.83 for rank 3
    assert out["PLAY-02"].net == 0.0 and out["PLAY-02"].neutral == 1 and "PLAY-09" not in out


def test_play_signals_ignore_lessons_sqlite_no_longer_vouches_for():
    hits = [hit("lesson-play:D-009:PLAY-01", ["play:PLAY-01", "signal:positive", "deal:D-009"]),
            hit("lesson-play:D-001:PLAY-01", ["play:PLAY-01", "signal:positive", "deal:D-001"], rank=2)]
    out = play_signals(hits, {"PLAY-01"}, valid_deals={"D-001"}, valid_documents={"lesson-play:D-001:PLAY-01"})
    assert out["PLAY-01"].evidence == ["lesson-play:D-001:PLAY-01"] and out["PLAY-01"].positive == 1


def test_reflect_is_asked_about_the_deal_type_by_industry_segment_and_objection(tmp_path):
    ctx = make_context(tmp_path)
    deal_id = add_deal(ctx.db, name="Open", account="Zebra", objections=[("sso", "unresolved")])
    with ctx.db.session() as s:
        deal = s.get(Deal, deal_id)
        facts = deal_facts(deal, deal.signals, [])
    lessons = FakeLessons()
    brief = asyncio.run(reflect_deal_type(lessons, facts))
    query, tags = lessons.reflect_calls[0]
    assert tags == ["industry:fintech", "segment:mid_market", "objection:sso"] == deal_type_tags(facts)
    assert "mid-market fintech deal" in query and "Zebra" not in query and brief.text


# ---- HindsightLessons call shapes ----------------------------------------------------------------------------


class FakeLessonClient:
    def __init__(self):
        self.calls: list[tuple] = []
        self.model_exists = False

    async def acreate_bank(self, bank, **kwargs):
        self.calls.append(("acreate_bank", bank, kwargs))

    async def aretain_batch(self, bank, **kwargs):
        self.calls.append(("aretain_batch", bank, kwargs))

    async def arecall(self, bank, **kwargs):
        self.calls.append(("arecall", bank, kwargs))
        return SimpleNamespace(results=[SimpleNamespace(text="t", tags=["play:PLAY-01"], document_id="lesson-x", type="world")])

    async def areflect(self, bank, **kwargs):
        self.calls.append(("areflect", bank, kwargs))
        memory = SimpleNamespace(id="1", text="fact", type="world")
        return SimpleNamespace(text="Advice.", based_on=SimpleNamespace(memories=[memory]))

    async def aget_mental_model(self, bank, model_id, **kwargs):
        self.calls.append(("aget_mental_model", bank, model_id, kwargs))
        if not self.model_exists:
            raise ApiException(status=404, reason="missing")
        return {"content": "Playbook text", "last_refreshed_at": None, "is_stale": False}

    async def acreate_mental_model(self, bank, **kwargs):
        self.calls.append(("acreate_mental_model", bank, kwargs))
        self.model_exists = True

    async def arefresh_mental_model(self, bank, model_id):
        self.calls.append(("arefresh_mental_model", bank, model_id))


def hindsight_lessons(client) -> HindsightLessons:
    lessons = HindsightLessons("http://hindsight.test", "deal-lessons")
    lessons._client = client
    return lessons


def names(client):
    return [c[0] for c in client.calls]


def test_lessons_bank_uses_concise_extraction_observations_and_the_deal_missions():
    client = FakeLessonClient()
    asyncio.run(hindsight_lessons(client).ensure_bank())
    assert client.calls == [("acreate_bank", "deal-lessons", {
        "name": "Deal lessons", "mission": BANK_MISSION, "retain_mission": BANK_MISSION, "retain_extraction_mode": "concise",
        "enable_observations": True, "reflect_mission": REFLECT_MISSION})]


def test_lessons_retain_is_async_batch_and_recall_matches_any_tag():
    client = FakeLessonClient()
    lessons = hindsight_lessons(client)
    items = [{"content": "c", "document_id": "lesson-outcome:D-001", "tags": ["deal:D-001"]}]
    hits = asyncio.run(_both(lessons, items))
    retain = next(c for c in client.calls if c[0] == "aretain_batch")
    assert retain[2] == {"items": items, "retain_async": True}
    recall = next(c for c in client.calls if c[0] == "arecall")
    assert recall[2]["tags"] == ["play:PLAY-01"] and recall[2]["tags_match"] == "any" and recall[2]["query"] == "sso"
    assert [(h.document_id, h.tags, h.rank) for h in hits] == [("lesson-x", ["play:PLAY-01"], 1)]


async def _both(lessons, items):
    await lessons.retain(items)
    return await lessons.recall("sso", ["play:PLAY-01"])


def test_reflect_and_playbook_only_create_or_refresh_when_asked():
    client = FakeLessonClient()
    lessons = hindsight_lessons(client)
    brief = asyncio.run(lessons.reflect("what do we know?", ["industry:fintech"]))
    reflect = next(c for c in client.calls if c[0] == "areflect")
    assert reflect[2]["tags"] == ["industry:fintech"] and reflect[2]["tags_match"] == "any" and reflect[2]["include_facts"] is True
    assert brief.text == "Advice." and brief.based_on == [{"id": "1", "text": "fact", "type": "world"}]

    assert asyncio.run(lessons.playbook()) is None  # reading is free and creates nothing
    assert "acreate_mental_model" not in names(client)
    result = asyncio.run(lessons.playbook(refresh=True))
    create = next(c for c in client.calls if c[0] == "acreate_mental_model")
    assert create[2]["id"] == PLAYBOOK_ID == "deal-playbook" and create[2]["source_query"] == PLAYBOOK_QUERY
    assert "arefresh_mental_model" in names(client) and result["content"] == "Playbook text"
    assert any(c[0] == "aget_mental_model" and c[3] == {"detail": "content"} for c in client.calls)


def test_lesson_failures_surface_as_memory_unavailable():
    class Broken(FakeLessonClient):
        async def aretain_batch(self, bank, **kwargs):
            raise ApiException(status=503, reason="busy")

    try:
        asyncio.run(hindsight_lessons(Broken()).retain([]))
    except MemoryUnavailable as exc:
        assert "503" in str(exc)
    else:
        raise AssertionError("expected MemoryUnavailable")
