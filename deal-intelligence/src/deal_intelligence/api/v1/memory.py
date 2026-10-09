"""The interactions bank: the only code that talks to Hindsight about deals.

Bank 1 holds a searchable copy of two kinds of document, retained in CHUNKS mode (verbatim, no
Hindsight LLM call):
- one summary per CLOSED deal (document_id "D-004"), written situation first and outcome last, so
  semantic recall matches the shape of a deal rather than the account's name;
- each interaction of an open deal (document_id "INT-0012") so the deal's own history is searchable.

Hindsight is never the source of truth: every recall result is re-checked against SQLite
(retrieval.py) before it can reach a brief. Behaviour this wrapper relies on: retaining the same
`document_id` again replaces it; recall hits carry `document_id` and typed `scores`; relevance
scores are not calibrated, so no threshold is applied here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING, Protocol

import aiohttp
from hindsight_client import Hindsight
from hindsight_client_api.exceptions import ApiException, NotFoundException

from .db import interaction_code
from .ranking import LOSS_LABELS, deal_facts, situation_text, slug

if TYPE_CHECKING:
    from .db import Deal, DealSignals, Interaction, Stakeholder

DEAL_SUMMARY_TAG = "kind:deal_summary"
INTERACTION_TAG = "kind:interaction"


class MemoryUnavailable(Exception):
    """Hindsight couldn't be reached or refused the call. Callers degrade instead of failing."""


@dataclass(frozen=True)
class RecallHit:
    code: str  # document id: D-xxx or INT-xxxx
    rank: int
    final: float | None = None
    semantic: float | None = None
    reranker: float | None = None
    keyword: float | None = None

    def as_dict(self) -> dict:
        return {"id": self.code, "rank": self.rank, "final": self.final, "semantic": self.semantic,
                "reranker": self.reranker, "keyword": self.keyword}


@dataclass(frozen=True)
class MemoryItem:
    code: str  # becomes the Hindsight document_id
    content: str
    tags: list[str] = field(default_factory=list)
    timestamp: datetime | None = None
    context: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class Memory(Protocol):
    async def ensure_bank(self) -> None: ...

    async def retain(self, item: MemoryItem) -> None: ...

    async def delete(self, code: str) -> None: ...

    async def recall(self, query: str, tags: list[str], limit: int) -> list[RecallHit]: ...

    async def healthy(self) -> bool: ...

    async def extraction_mode(self) -> str | None: ...

    async def close(self) -> None: ...


# ---- documents -------------------------------------------------------------------------------------


def _at(day: date | None, fallback: datetime | None = None) -> datetime | None:
    return datetime.combine(day, time(12, 0), tzinfo=UTC) if day else fallback


def deal_summary_item(
    deal: Deal, signals: DealSignals | None, stakeholders: list[Stakeholder],
    play_names: dict[str, str] | None = None,
) -> MemoryItem:
    """The retained summary of a closed deal: situation first, then plays, then the outcome and why."""
    facts = deal_facts(deal, signals, stakeholders)
    names = play_names or {}
    parts = [situation_text(facts, closed=True)]
    if facts.plays_used:
        parts.append("Plays used: " + ", ".join(f"{p} ({names[p]})" if p in names else p for p in facts.plays_used) + ".")
    else:
        parts.append("No plays were recorded.")
    if deal.result == "won":
        parts.append("Outcome: the deal was won.")
    elif deal.result == "lost":
        reason = LOSS_LABELS.get(deal.loss_reason or "", deal.loss_reason)
        parts.append("Outcome: the deal was lost" + (f", because of {reason}." if reason else "."))
    else:
        parts.append("Outcome: not yet decided.")
    tags = [DEAL_SUMMARY_TAG]
    if facts.industry_key:
        tags.append(f"industry:{facts.industry_key}")
    if facts.segment:
        tags.append(f"segment:{facts.segment}")
    if deal.result in ("won", "lost"):
        tags.append(f"result:{deal.result}")
    if deal.loss_reason:
        tags.append(f"loss:{deal.loss_reason}")
    tags += [f"objection:{t}" for t in sorted(facts.objection_types)]
    tags += [f"competitor:{c}" for c in sorted(facts.competitor_slugs)]
    return MemoryItem(
        code=deal.code, content=" ".join(parts), tags=tags, timestamp=_at(deal.closed_on, deal.updated_at),
        context="summary of a closed sales deal and how it ended", metadata={"deal_id": deal.code},
    )


def interaction_item(deal: Deal, interaction: Interaction) -> MemoryItem:
    header = f"{interaction.kind.replace('_', ' ').capitalize()} on {interaction.occurred_on.isoformat()}"
    if interaction.author:
        header += f" by {interaction.author}"
    if interaction.subject:
        header += f", subject: {interaction.subject}"
    return MemoryItem(
        code=interaction_code(interaction.id), content=f"{header}.\n{interaction.text}",
        tags=[INTERACTION_TAG, f"deal:{deal.code}", f"account:{slug(deal.account)}"],
        timestamp=_at(interaction.occurred_on, interaction.created_at),
        context=f"{interaction.kind.replace('_', ' ')} from the history of deal {deal.code}",
        metadata={"interaction_id": interaction_code(interaction.id), "deal_id": deal.code},
    )


# ---- Hindsight ---------------------------------------------------------------------------------------

_CONNECTION_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, OSError)


class HindsightMemory:
    """Works with a local server (default, http://127.0.0.1:8888) or Hindsight Cloud (set
    DEAL_HINDSIGHT_URL and DEAL_HINDSIGHT_API_KEY). The API surface is the same."""

    def __init__(self, base_url: str, bank: str, api_key: str | None = None, timeout: float = 30.0, order: str = "semantic"):
        self.base_url = base_url
        self.bank = bank
        self.order = order  # "semantic": by Hindsight's semantic score; "hindsight": as returned
        self._api_key = api_key or None
        self.timeout = timeout
        self._client: Hindsight | None = None
        self._bank_ready = False

    @property
    def client(self) -> Hindsight:
        if self._client is None:
            self._client = Hindsight(base_url=self.base_url, api_key=self._api_key, timeout=self.timeout, max_attempts=2)
        return self._client

    async def ensure_bank(self) -> None:
        if self._bank_ready:
            return
        try:
            # Chunks mode keeps summaries and interactions verbatim with no LLM call. Set explicitly so
            # the bank stays verbatim even if the server is later given an LLM.
            await self.client.acreate_bank(
                self.bank, name="Deal interactions", retain_extraction_mode="chunks", enable_observations=False,
            )
        except ApiException as exc:
            if exc.status not in (400, 409):  # already exists
                raise MemoryUnavailable(f"Hindsight refused to create bank {self.bank!r}: {exc.status}") from exc
        except _CONNECTION_ERRORS as exc:
            raise MemoryUnavailable(f"Hindsight is not reachable at {self.base_url}") from exc
        self._bank_ready = True

    async def retain(self, item: MemoryItem) -> None:
        await self.ensure_bank()
        try:
            await self.client.aretain(
                self.bank, content=item.content, document_id=item.code, tags=list(item.tags),
                timestamp=item.timestamp, context=item.context, metadata=dict(item.metadata),
            )
        except ApiException as exc:
            raise MemoryUnavailable(f"Hindsight rejected {item.code}: {exc.status} {exc.reason}") from exc
        except _CONNECTION_ERRORS as exc:
            raise MemoryUnavailable(f"Hindsight is not reachable at {self.base_url}") from exc

    async def delete(self, code: str) -> None:
        try:
            await self.client.documents.delete_document(self.bank, code)
        except NotFoundException:
            return  # already gone: the goal is reached
        except ApiException as exc:
            if exc.status == 404:
                return
            raise MemoryUnavailable(f"Hindsight couldn't delete {code}: {exc.status} {exc.reason}") from exc
        except _CONNECTION_ERRORS as exc:
            raise MemoryUnavailable(f"Hindsight is not reachable at {self.base_url}") from exc

    async def recall(self, query: str, tags: list[str], limit: int) -> list[RecallHit]:
        await self.ensure_bank()
        try:
            response = await self.client.arecall(self.bank, query=query, tags=list(tags), tags_match="all_strict", budget="mid")
        except NotFoundException:
            return []
        except ApiException as exc:
            raise MemoryUnavailable(f"Hindsight recall failed: {exc.status} {exc.reason}") from exc
        except _CONNECTION_ERRORS as exc:
            raise MemoryUnavailable(f"Hindsight is not reachable at {self.base_url}") from exc

        found: dict[str, dict] = {}
        for result in response.results or []:
            code = result.document_id or (result.metadata or {}).get("deal_id") or (result.metadata or {}).get("interaction_id")
            if not code:
                continue
            scores = result.scores
            semantic = getattr(scores, "semantic", None)
            if code in found:  # a long document can come back as several chunks: keep its best semantic score
                if semantic is not None and (found[code]["semantic"] is None or semantic > found[code]["semantic"]):
                    found[code]["semantic"] = semantic
                continue
            found[code] = {"code": code, "order": len(found), "final": getattr(scores, "final", None), "semantic": semantic,
                           "reranker": getattr(scores, "reranker", None), "keyword": getattr(scores, "keyword", None)}
        rows = list(found.values())
        if self.order == "semantic" and rows and all(r["semantic"] is not None for r in rows):
            # Hindsight's fused order can bury the closest match; order by its semantic score before the pool is cut.
            rows.sort(key=lambda r: (-r["semantic"], r["order"]))
        return [RecallHit(code=r["code"], rank=i + 1, final=r["final"], semantic=r["semantic"], reranker=r["reranker"],
                          keyword=r["keyword"]) for i, r in enumerate(rows[:limit])]

    async def healthy(self) -> bool:
        try:
            await self.client.aget_version()
            return True
        except Exception:  # health must never raise
            return False

    async def extraction_mode(self) -> str | None:
        """The bank's actual retain mode. It must be 'chunks'; the status panel shows this so a Cloud
        bank can't silently run with LLM extraction."""
        try:
            await self.ensure_bank()
            config = await self.client.aget_bank_config(self.bank)
        except Exception:  # status must never raise
            return None
        config = config if isinstance(config, dict) else config.to_dict()
        return (config.get("config", config) or {}).get("retain_extraction_mode")

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None
