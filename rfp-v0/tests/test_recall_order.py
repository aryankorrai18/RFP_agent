"""Pilot 2026-10-08: Hindsight's fused order put the closest approved answer at #9 while its own semantic score put it
first. Candidates are now ordered by that semantic score (RFP_RECALL_ORDER=semantic); "hindsight" keeps the old order."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rfp_assistant.api.v1.memory import HindsightMemory
from rfp_assistant.config import Settings


def result(code: str, semantic: float | None, final: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(metadata={"answer_id": code}, document_id=code,
                           scores=SimpleNamespace(semantic=semantic, final=final, reranker=None, keyword=None))


class FakeClient:
    def __init__(self, results) -> None:  # noqa: ANN001
        self.results = results

    async def arecall(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return SimpleNamespace(results=self.results)


def recall(order: str, results, limit: int = 12):  # noqa: ANN001, ANN201
    memory = HindsightMemory("http://hindsight.test", "bank", order=order)
    memory._client, memory._bank_ready = FakeClient(results), True
    return asyncio.run(memory.recall("How are user passwords protected when stored?", limit=limit))


# ANS-0022 comes back as two chunks, fourth in Hindsight's own order, as in the pilot
PILOT = [result("ANS-0014", 0.47), result("ANS-0005", 0.49), result("ANS-0011", 0.45), result("ANS-0022", 0.83),
         result("ANS-0022", 0.91), result("ANS-0008", 0.46)]


def test_candidates_are_ordered_by_semantic_score_and_a_chunked_answer_keeps_its_best():
    hits = recall("semantic", PILOT)
    assert [h.answer_code for h in hits] == ["ANS-0022", "ANS-0005", "ANS-0014", "ANS-0008", "ANS-0011"]
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5] and hits[0].semantic == 0.91


def test_the_old_order_is_still_available():
    assert [h.answer_code for h in recall("hindsight", PILOT)] == ["ANS-0014", "ANS-0005", "ANS-0011", "ANS-0022", "ANS-0008"]


def test_without_scores_for_every_candidate_hindsights_order_is_kept_and_the_limit_applies():
    hits = recall("semantic", [result("A", 0.2), result("B", None), result("C", 0.9)], limit=2)
    assert [h.answer_code for h in hits] == ["A", "B"]


def test_the_setting_defaults_to_semantic_and_refuses_anything_else(monkeypatch):
    monkeypatch.delenv("RFP_RECALL_ORDER", raising=False)
    assert Settings.from_env().recall_order == "semantic"
    monkeypatch.setenv("RFP_RECALL_ORDER", "random")
    with pytest.raises(ValueError):
        Settings.from_env()
