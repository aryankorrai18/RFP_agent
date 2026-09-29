"""The only code that talks to Hindsight (design §7, §13).

Hindsight is a searchable copy of approved answers, never the source of truth. Behaviour the
spike verified and this wrapper relies on:
- the async API (`client.documents` is async-only), closed with `aclose()`;
- retaining the same `document_id` again replaces it (upsert);
- recall hits carry `document_id`, `metadata`, `tags` and typed `scores`;
- relevance scores aren't calibrated for this content, so no threshold is applied here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import aiohttp
from hindsight_client import Hindsight
from hindsight_client_api.exceptions import ApiException, NotFoundException

ANSWER_TAG = "kind:rfp_answer"


class MemoryUnavailable(Exception):
    """Hindsight couldn't be reached or refused the call. Callers degrade instead of failing."""


@dataclass(frozen=True)
class RecallHit:
    answer_code: str
    rank: int
    final: float | None = None
    semantic: float | None = None
    reranker: float | None = None
    keyword: float | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.answer_code,
            "rank": self.rank,
            "final": self.final,
            "semantic": self.semantic,
            "reranker": self.reranker,
            "keyword": self.keyword,
        }


@dataclass(frozen=True)
class MemoryItem:
    code: str
    question: str
    answer: str
    client: str | None
    industry: str | None
    timestamp: datetime | None


class Memory(Protocol):
    async def ensure_bank(self) -> None: ...

    async def retain(self, item: MemoryItem) -> None: ...

    async def delete(self, code: str) -> None: ...

    async def recall(self, query: str, limit: int) -> list[RecallHit]: ...

    async def healthy(self) -> bool: ...

    async def extraction_mode(self) -> str | None: ...

    async def close(self) -> None: ...


def memory_content(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def memory_tags(item: MemoryItem) -> list[str]:
    tags = [ANSWER_TAG]
    if item.industry:
        tags.append(f"industry:{item.industry.strip().lower()}")
    if item.client:
        tags.append(f"client:{item.client.strip().lower()}")
    return tags


_CONNECTION_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, OSError)


class HindsightMemory:
    """Works with a local server (default, http://127.0.0.1:8888) or Hindsight Cloud (set
    RFP_HINDSIGHT_URL and RFP_HINDSIGHT_API_KEY). The API surface is the same."""

    def __init__(self, base_url: str, bank: str, api_key: str | None = None, timeout: float = 30.0):
        self.base_url = base_url
        self.bank = bank
        self._api_key = api_key or None
        self.timeout = timeout
        self._client: Hindsight | None = None
        self._bank_ready = False

    @property
    def client(self) -> Hindsight:
        if self._client is None:
            self._client = Hindsight(
                base_url=self.base_url, api_key=self._api_key, timeout=self.timeout, max_attempts=2
            )
        return self._client

    async def ensure_bank(self) -> None:
        if self._bank_ready:
            return
        try:
            # Chunks mode stores answers verbatim with no LLM call (V1-D2, V1-D7). Set explicitly
            # so the bank stays verbatim even if the server is later given an LLM.
            await self.client.acreate_bank(
                self.bank,
                name="RFP answer library",
                retain_extraction_mode="chunks",
                enable_observations=False,
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
                self.bank,
                content=memory_content(item.question, item.answer),
                document_id=item.code,
                metadata={"answer_id": item.code},
                tags=memory_tags(item),
                timestamp=item.timestamp,
                context="approved RFP answer",
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

    async def recall(self, query: str, limit: int) -> list[RecallHit]:
        await self.ensure_bank()
        try:
            response = await self.client.arecall(
                self.bank, query=query, tags=[ANSWER_TAG], tags_match="all_strict", budget="mid"
            )
        except NotFoundException:
            return []
        except ApiException as exc:
            raise MemoryUnavailable(f"Hindsight recall failed: {exc.status} {exc.reason}") from exc
        except _CONNECTION_ERRORS as exc:
            raise MemoryUnavailable(f"Hindsight is not reachable at {self.base_url}") from exc

        hits: list[RecallHit] = []
        seen: set[str] = set()
        for result in response.results or []:
            code = (result.metadata or {}).get("answer_id") or result.document_id
            if not code or code in seen:  # a long answer can come back as several chunks
                continue
            seen.add(code)
            scores = result.scores
            hits.append(
                RecallHit(
                    answer_code=code,
                    rank=len(hits) + 1,
                    final=getattr(scores, "final", None),
                    semantic=getattr(scores, "semantic", None),
                    reranker=getattr(scores, "reranker", None),
                    keyword=getattr(scores, "keyword", None),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    async def healthy(self) -> bool:
        try:
            await self.client.aget_version()
            return True
        except Exception:  # health must never raise
            return False

    async def extraction_mode(self) -> str | None:
        """The bank's actual retain mode. V1 requires 'chunks' (no LLM extraction, V1-D7); the
        status panel shows this so a Cloud bank can't silently run with LLM extraction."""
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
