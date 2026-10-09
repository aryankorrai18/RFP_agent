"""The Hindsight Reflect cache and the leave-one-out memory self-check, offline."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from deal_intelligence.api.v1 import demo, lessons, outcomes, quality, reflection
from deal_intelligence.errors import PipelineError
from deal_intelligence.main import app
from tests.builders import make_context
from tests.fake_llm import FakeLLM
from tests.fakes import FakeLessons, FakeMemory


@pytest.fixture
def ctx(tmp_path):
    ctx = make_context(tmp_path, memory=FakeMemory(), lessons=FakeLessons(), llm=FakeLLM())
    info = demo.seed_demo(ctx)
    outcomes.rebuild_play_stats(ctx.db)
    lessons.collect_lessons(ctx.db)
    ctx.demo_id = int(info["demo_deal"][2:])
    return ctx


def test_reflection_is_missing_then_cached_and_refreshable(ctx):
    assert reflection.get_reflection(ctx, ctx.demo_id)["state"] == "missing"
    first = asyncio.run(reflection.refresh_reflection(ctx, ctx.demo_id))
    assert first["state"] == "ready" and first["text"] and first["created_at"]
    cached = reflection.get_reflection(ctx, ctx.demo_id)
    assert cached["state"] == "ready" and cached["text"] == first["text"]
    assert len(ctx.lessons.reflect_calls) == 1  # reading the cache never asks again
    asyncio.run(reflection.refresh_reflection(ctx, ctx.demo_id))
    assert len(ctx.lessons.reflect_calls) == 2


def test_reflection_needs_the_lessons_bank_and_a_real_deal(ctx):
    with pytest.raises(PipelineError) as missing:
        reflection.get_reflection(ctx, 9999)
    assert missing.value.http_status == 404
    ctx.lessons = None
    with pytest.raises(PipelineError) as off:
        asyncio.run(reflection.refresh_reflection(ctx, ctx.demo_id))
    assert off.value.code == "lessons_disabled"


def test_quality_is_computed_from_the_database_alone(ctx):
    class Boom:
        def __getattr__(self, name):
            raise AssertionError("the self-check must not touch memory")

    ctx.memory, ctx.lessons = Boom(), Boom()
    result = asyncio.run(quality.memory_quality(ctx))
    assert result["n_closed"] == 19 and result["won"]["n"] == 9 and result["lost"]["n"] == 10
    assert 0 <= result["won"]["hit"] <= result["won"]["covered"] <= result["won"]["n"]
    assert 0 <= result["lost"]["warned"] <= result["lost"]["had_unresolved"] <= result["lost"]["n"]
    assert result["won"]["hit"] >= result["won"]["covered"] // 2  # better than a coin flip on the planted patterns
    assert result["lost"]["avoid_hit"] >= 1  # the discount play is flagged for at least one lost pricing deal
    assert "No model" in result["method"] and "closed deals" in result["caveat"]


def test_routes(ctx):
    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as client:
            assert client.get(f"/v1/deals/{ctx.demo_id}/memory-says").json()["state"] == "missing"
            refreshed = client.post(f"/v1/deals/{ctx.demo_id}/memory-says")
            assert refreshed.status_code == 200 and refreshed.json()["state"] == "ready"
            assert client.get(f"/v1/deals/{ctx.demo_id}/memory-says").json()["text"] == refreshed.json()["text"]
            assert client.get("/v1/deals/9999/memory-says").status_code == 404
            q = client.get("/v1/memory/quality").json()
            assert q["n_closed"] == 19 and q["top"] == 3
    finally:
        app.state.v1_factory = None
