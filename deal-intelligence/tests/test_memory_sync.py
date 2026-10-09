"""Documents written to Hindsight (summary text, tags), the HindsightMemory call shapes with a fake
client, and the outbox that keeps Hindsight in step with SQLite. Everything is offline."""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import aiohttp
from hindsight_client_api.exceptions import ApiException

from deal_intelligence.api.v1.db import Deal, Interaction, utcnow
from deal_intelligence.api.v1.memory import (
    HindsightMemory, MemoryItem, MemoryUnavailable, deal_summary_item, interaction_item,
)
from deal_intelligence.api.v1.sync import pending_count, sync_outbox, touch

from .builders import add_deal, make_context
from .fakes import BUYER, ENGAGED, SILENT, FakeMemory, add_catalogue, closed


def summary_for(ctx, deal_id):
    with ctx.db.session() as s:
        deal = s.get(Deal, deal_id)
        return deal_summary_item(deal, deal.signals, list(deal.stakeholders), {"PLAY-01": "SSO integration workshop"})


def lost_fintech(ctx):
    add_catalogue(ctx.db)
    return closed(ctx.db, "Quill", "Quill Capital", "lost", loss_reason="security_compliance",
                  objections=[("sso", "unresolved"), ("security_review", "addressed")], competitors=["Brightline"],
                  plays=["PLAY-01"], stakeholders=[SILENT, BUYER])


# ---- document text and tags ----------------------------------------------------------------------------


def test_summary_is_lead_first_situation_before_outcome_and_never_names_the_account(tmp_path):
    ctx = make_context(tmp_path)
    item = summary_for(ctx, lost_fintech(ctx))
    text = item.content
    assert text.startswith("A mid-market fintech deal.")
    positions = [text.index(part) for part in ("SSO (stayed unresolved)", "The competitor was Brightline", "The champion went silent",
                                               "Plays used: PLAY-01 (SSO integration workshop)", "Outcome: the deal was lost")]
    assert positions == sorted(positions)  # situation, then plays, then outcome
    assert "because of security and compliance" in text.rsplit("Outcome", 1)[1]
    assert "Quill" not in text
    assert item.code == "D-001" and item.metadata == {"deal_id": "D-001"}


def test_summary_tags_cover_the_structured_keys(tmp_path):
    ctx = make_context(tmp_path)
    tags = set(summary_for(ctx, lost_fintech(ctx)).tags)
    assert tags == {"kind:deal_summary", "industry:fintech", "segment:mid_market", "result:lost", "loss:security_compliance",
                    "objection:sso", "objection:security_review", "competitor:brightline"}


def test_interaction_item_is_tagged_by_deal_and_account(tmp_path):
    ctx = make_context(tmp_path)
    deal_id = add_deal(ctx.db, name="Acme", account="Acme Payments", interactions=[("email", date(2026, 6, 12), "We need SSO before signing.")])
    with ctx.db.session() as s:
        deal = s.get(Deal, deal_id)
        interaction = deal.interactions[0]
        item = interaction_item(deal, interaction)
    assert item.code == "INT-0001"
    assert item.tags == ["kind:interaction", "deal:D-001", "account:acme-payments"]
    assert "We need SSO before signing." in item.content and "2026-06-12" in item.content


# ---- HindsightMemory call shapes -----------------------------------------------------------------------------


class FakeDocuments:
    def __init__(self, client):
        self.client = client

    async def delete_document(self, bank, code):
        self.client.calls.append(("delete_document", bank, code))
        if self.client.delete_error:
            raise self.client.delete_error


class FakeClient:
    def __init__(self):
        self.calls: list[tuple] = []
        self.create_error: Exception | None = None
        self.delete_error: Exception | None = None
        self.recall_error: Exception | None = None
        self.results: list = []
        self.documents = FakeDocuments(self)

    async def acreate_bank(self, bank, **kwargs):
        self.calls.append(("acreate_bank", bank, kwargs))
        if self.create_error:
            raise self.create_error

    async def aretain(self, bank, **kwargs):
        self.calls.append(("aretain", bank, kwargs))

    async def arecall(self, bank, **kwargs):
        self.calls.append(("arecall", bank, kwargs))
        if self.recall_error:
            raise self.recall_error
        return SimpleNamespace(results=self.results)

    async def aget_version(self):
        return {"version": "x"}

    async def aget_bank_config(self, bank):
        return {"config": {"retain_extraction_mode": "chunks"}}

    async def aclose(self):
        self.calls.append(("aclose",))


def hindsight(client: FakeClient) -> HindsightMemory:
    memory = HindsightMemory("http://hindsight.test", "deal-interactions")
    memory._client = client  # type: ignore[assignment]
    return memory


def hit(document_id, final=0.5, metadata=None):
    return SimpleNamespace(document_id=document_id, metadata=metadata, scores=SimpleNamespace(final=final, semantic=final, reranker=None, keyword=None))


def test_bank_is_created_once_in_chunks_mode_without_observations():
    client = FakeClient()
    memory = hindsight(client)

    async def scenario():
        await memory.ensure_bank()
        await memory.ensure_bank()

    asyncio.run(scenario())
    assert client.calls == [("acreate_bank", "deal-interactions", {
        "name": "Deal interactions", "retain_extraction_mode": "chunks", "enable_observations": False})]


def test_an_existing_bank_is_fine_but_other_errors_mean_unavailable():
    client = FakeClient()
    client.create_error = ApiException(status=409, reason="exists")
    asyncio.run(hindsight(client).ensure_bank())

    client = FakeClient()
    client.create_error = ApiException(status=500, reason="boom")
    try:
        asyncio.run(hindsight(client).ensure_bank())
    except MemoryUnavailable as exc:
        assert "500" in str(exc)
    else:
        raise AssertionError("expected MemoryUnavailable")


def test_retain_sends_the_document_id_tags_timestamp_context_and_metadata():
    client = FakeClient()
    memory = hindsight(client)
    stamp = utcnow()
    item = MemoryItem(code="D-004", content="A deal.", tags=["kind:deal_summary", "result:lost"], timestamp=stamp,
                      context="summary of a closed sales deal", metadata={"deal_id": "D-004"})
    asyncio.run(memory.retain(item))
    retain = next(c for c in client.calls if c[0] == "aretain")
    assert retain[1] == "deal-interactions"
    assert retain[2] == {"content": "A deal.", "document_id": "D-004", "tags": ["kind:deal_summary", "result:lost"],
                         "timestamp": stamp, "context": "summary of a closed sales deal", "metadata": {"deal_id": "D-004"}}


def test_recall_requires_every_tag_and_dedupes_chunks_by_document_id():
    client = FakeClient()
    client.results = [hit("D-001", 0.9), hit("D-001", 0.8), hit("D-002", 0.4), hit(None, 0.3, {"deal_id": "D-003"}), hit(None, 0.2)]
    hits = asyncio.run(hindsight(client).recall("sso deal", ["kind:deal_summary"], limit=10))
    call = next(c for c in client.calls if c[0] == "arecall")
    assert call[2] == {"query": "sso deal", "tags": ["kind:deal_summary"], "tags_match": "all_strict", "budget": "mid"}
    assert [(h.code, h.rank, h.final) for h in hits] == [("D-001", 1, 0.9), ("D-002", 2, 0.4), ("D-003", 3, 0.3)]
    assert len(asyncio.run(hindsight(client).recall("q", ["kind:deal_summary"], limit=2))) == 2


def test_connection_errors_become_memory_unavailable_and_health_never_raises():
    client = FakeClient()
    client.recall_error = aiohttp.ClientConnectionError("down")
    memory = hindsight(client)
    try:
        asyncio.run(memory.recall("q", ["kind:deal_summary"], 5))
    except MemoryUnavailable:
        pass
    else:
        raise AssertionError("expected MemoryUnavailable")
    assert asyncio.run(memory.healthy()) is True and asyncio.run(memory.extraction_mode()) == "chunks"


def test_delete_goes_through_the_documents_api_and_a_404_is_success():
    client = FakeClient()
    memory = hindsight(client)
    asyncio.run(memory.delete("D-004"))
    client.delete_error = ApiException(status=404, reason="gone")
    asyncio.run(memory.delete("D-005"))
    assert [c for c in client.calls if c[0] == "delete_document"] == [
        ("delete_document", "deal-interactions", "D-004"), ("delete_document", "deal-interactions", "D-005")]
    client.delete_error = ApiException(status=500, reason="boom")
    try:
        asyncio.run(memory.delete("D-006"))
    except MemoryUnavailable:
        pass
    else:
        raise AssertionError("expected MemoryUnavailable")


# ---- the outbox ----------------------------------------------------------------------------------------------


def rows(ctx):
    with ctx.db.session() as s:
        deals = {d.code: (d.hindsight_status, d.hindsight_attempts) for d in s.query(Deal)}
        interactions = {i.code: i.hindsight_status for i in s.query(Interaction)}
    return deals, interactions


def test_outbox_retains_closed_summaries_and_open_interactions_but_never_skipped_ones(tmp_path):
    memory = FakeMemory()
    ctx = make_context(tmp_path, memory=memory)
    add_catalogue(ctx.db)
    closed(ctx.db, "Won", "A", "won", plays=["PLAY-01"], stakeholders=[ENGAGED],
           interactions=[("email", date(2026, 1, 5), "old history")])  # closed deal: interactions are skipped
    add_deal(ctx.db, name="Open", account="B", interactions=[("email", date(2026, 6, 1), "hello"), ("call_note", date(2026, 6, 2), "sso")])

    report = asyncio.run(sync_outbox(ctx.db, memory, ctx.sync_lock))
    assert (report.retained, report.deleted, report.failed, report.pending) == (3, 0, 0, 0)
    assert sorted(memory.docs) == ["D-001", "INT-0002", "INT-0003"]  # the open deal's summary is not due, INT-0001 was skipped
    deals, interactions = rows(ctx)
    assert deals == {"D-001": ("retained", 0), "D-002": ("pending", 0)}
    assert interactions == {"INT-0001": "skipped", "INT-0002": "retained", "INT-0003": "retained"}
    assert asyncio.run(sync_outbox(ctx.db, memory, ctx.sync_lock)).retained == 0  # nothing left to do


def test_outbox_deletes_what_was_marked_pending_delete(tmp_path):
    memory = FakeMemory()
    ctx = make_context(tmp_path, memory=memory)
    add_catalogue(ctx.db)
    deal_id = closed(ctx.db, "Won", "A", "won", plays=["PLAY-01"])
    asyncio.run(sync_outbox(ctx.db, memory, ctx.sync_lock))
    assert "D-001" in memory.docs
    with ctx.db.session() as s:
        deal = s.get(Deal, deal_id)
        deal.status, deal.hindsight_status = "deleted", "pending_delete"
        touch(deal)
        s.commit()
    report = asyncio.run(sync_outbox(ctx.db, memory, ctx.sync_lock))
    assert report.deleted == 1 and "D-001" not in memory.docs and memory.deleted == ["D-001"]
    assert rows(ctx)[0]["D-001"][0] == "deleted"


def test_outage_stops_at_the_first_failure_keeps_rows_pending_and_recovers(tmp_path):
    memory = FakeMemory()
    ctx = make_context(tmp_path, memory=memory)
    add_catalogue(ctx.db)
    closed(ctx.db, "One", "A", "won", plays=["PLAY-01"])
    closed(ctx.db, "Two", "B", "lost", loss_reason="price", plays=["PLAY-03"])

    memory.available = False
    report = asyncio.run(sync_outbox(ctx.db, memory, ctx.sync_lock))
    assert (report.retained, report.failed, report.pending) == (0, 1, 2)  # stopped after the first
    deals, _ = rows(ctx)
    assert deals == {"D-001": ("pending", 1), "D-002": ("pending", 0)}
    with ctx.db.session() as s:
        assert "down" in s.get(Deal, 1).hindsight_error

    memory.available = True
    report = asyncio.run(sync_outbox(ctx.db, memory, ctx.sync_lock))
    assert (report.retained, report.failed, report.pending) == (2, 0, 0)
    with ctx.db.session() as s:
        assert s.get(Deal, 1).hindsight_error is None


def test_a_deal_edited_during_the_sync_stays_pending(tmp_path):
    memory = FakeMemory()
    ctx = make_context(tmp_path, memory=memory)
    add_catalogue(ctx.db)
    deal_id = closed(ctx.db, "One", "A", "won", plays=["PLAY-01"])

    def edit_midway(_item):
        with ctx.db.session() as s:
            touch(s.get(Deal, deal_id))
            s.commit()

    memory.on_retain = edit_midway
    asyncio.run(sync_outbox(ctx.db, memory, ctx.sync_lock))
    assert rows(ctx)[0]["D-001"][0] == "pending"  # the newer version goes out next time
    memory.on_retain = None
    assert asyncio.run(sync_outbox(ctx.db, memory, ctx.sync_lock)).retained == 1


def test_a_closed_deal_without_signals_waits_and_is_not_counted_as_pending(tmp_path):
    memory = FakeMemory()
    ctx = make_context(tmp_path, memory=memory)
    with ctx.db.session() as s:
        s.add(Deal(name="Bare", account="X", result="won", stage="closed", closed_on=date(2026, 1, 1)))
        s.commit()
    assert pending_count(ctx.db) == 0
    assert asyncio.run(sync_outbox(ctx.db, memory, ctx.sync_lock)).retained == 0
