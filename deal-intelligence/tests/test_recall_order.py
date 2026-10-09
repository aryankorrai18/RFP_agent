"""The similar-deal pool is cut from Hindsight's candidates; they are ordered by Hindsight's semantic score first
(DEAL_RECALL_ORDER=semantic), so the fused order can't push the closest deals out of the pool (RFP pilot, 2026-10-08)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from deal_intelligence.api.v1.memory import HindsightMemory
from deal_intelligence.config import Settings


def result(code: str, semantic: float | None) -> SimpleNamespace:
    return SimpleNamespace(metadata={"deal_id": code}, document_id=code,
                           scores=SimpleNamespace(semantic=semantic, final=1.0, reranker=None, keyword=None))


class FakeClient:
    def __init__(self, results) -> None:  # noqa: ANN001
        self.results = results

    async def arecall(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return SimpleNamespace(results=self.results)


def recall(order: str, results, limit: int):  # noqa: ANN001, ANN201
    memory = HindsightMemory("http://hindsight.test", "bank", order=order)
    memory._client, memory._bank_ready = FakeClient(results), True
    return asyncio.run(memory.recall("logistics deal with integration and SSO objections", ["deal_summary"], limit=limit))


RESULTS = [result("D-010", 0.40), result("D-011", 0.42), result("D-002", 0.88), result("D-012", 0.39), result("D-002", 0.95)]


def test_the_pool_keeps_the_closest_deals_even_when_hindsight_lists_them_late():
    hits = recall("semantic", RESULTS, limit=2)
    assert [h.code for h in hits] == ["D-002", "D-011"] and hits[0].semantic == 0.95 and [h.rank for h in hits] == [1, 2]


def test_the_old_order_can_be_restored():
    assert [h.code for h in recall("hindsight", RESULTS, limit=2)] == ["D-010", "D-011"]


def test_the_setting_defaults_to_semantic_and_refuses_anything_else(monkeypatch):
    monkeypatch.delenv("DEAL_RECALL_ORDER", raising=False)
    assert Settings.from_env().recall_order == "semantic"
    monkeypatch.setenv("DEAL_RECALL_ORDER", "sideways")
    with pytest.raises(ValueError):
        Settings.from_env()
