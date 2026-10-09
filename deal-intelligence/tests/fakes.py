"""Offline stand-ins for Hindsight (both banks) and a small seeded deal history used by the memory tests."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date

from deal_intelligence.api.v1.db import Database
from deal_intelligence.api.v1.lessons import LessonBrief, LessonHit
from deal_intelligence.api.v1.memory import MemoryItem, MemoryUnavailable, RecallHit

from .builders import add_deal, add_play

WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are", "was", "were", "with", "as"}


def words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP}


class FakeMemory:
    """In-memory interactions bank. Ranks by word overlap and honours tags with all_strict semantics.
    `available=False` simulates an outage."""

    def __init__(self) -> None:
        self.docs: dict[str, MemoryItem] = {}
        self.available = True
        self.retained: list[str] = []
        self.deleted: list[str] = []
        self.recalls: list[tuple[str, list[str], int]] = []
        self.on_retain: Callable[[MemoryItem], None] | None = None

    def _check(self) -> None:
        if not self.available:
            raise MemoryUnavailable("fake Hindsight is down")

    async def ensure_bank(self) -> None:
        self._check()

    async def retain(self, item: MemoryItem) -> None:
        self._check()
        if self.on_retain:
            self.on_retain(item)
        self.docs[item.code] = item  # same document id replaces, like Hindsight
        self.retained.append(item.code)

    async def delete(self, code: str) -> None:
        self._check()
        self.docs.pop(code, None)
        self.deleted.append(code)

    async def recall(self, query: str, tags: list[str], limit: int) -> list[RecallHit]:
        self._check()
        self.recalls.append((query, list(tags), limit))
        q = words(query)
        scored = sorted(
            ((len(q & words(d.content)) / (len(q) or 1), code) for code, d in self.docs.items()
             if set(tags) <= set(d.tags)),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [RecallHit(code=code, rank=i + 1, final=score, semantic=score) for i, (score, code) in enumerate(scored[:limit])]

    async def healthy(self) -> bool:
        return self.available

    async def extraction_mode(self) -> str | None:
        return "chunks" if self.available else None

    async def close(self) -> None:
        return None


class FakeLessons:
    """In-memory lessons bank: retain replaces by document_id, recall filters by any matching tag and
    ranks by word overlap, reflect and the playbook return canned text built from what was retained."""

    def __init__(self) -> None:
        self.items: list[dict] = []
        self.available = True
        self.batches: list[int] = []
        self.recalls: list[tuple[str, list[str]]] = []
        self.reflect_calls: list[tuple[str, list[str] | None]] = []
        self.playbook_content: str | None = None

    def _check(self) -> None:
        if not self.available:
            raise MemoryUnavailable("fake lessons bank is down")

    async def retain(self, items: list[dict]) -> None:
        self._check()
        self.batches.append(len(items))
        by_id = {item["document_id"]: item for item in self.items}
        for item in items:
            by_id[item["document_id"]] = item
        self.items = list(by_id.values())

    async def recall(self, query: str, tags: list[str], limit: int = 20) -> list[LessonHit]:
        self._check()
        self.recalls.append((query, list(tags)))
        wanted, q = set(tags), words(query)
        matches = [i for i in self.items if wanted & set(i["tags"])]
        matches.sort(key=lambda i: -len(q & words(i["content"])))
        return [LessonHit(text=i["content"], tags=list(i["tags"]), document_id=i["document_id"], type="world", rank=n)
                for n, i in enumerate(matches[:limit], start=1)]

    async def reflect(self, query: str, tags: list[str] | None = None) -> LessonBrief:
        self._check()
        self.reflect_calls.append((query, tags))
        relevant = [i for i in self.items if not tags or set(tags) & set(i["tags"])]
        return LessonBrief(text=f"{len(relevant)} lessons remembered.",
                           based_on=[{"id": str(n), "text": i["content"], "type": "world"} for n, i in enumerate(relevant[:5])])

    async def playbook(self, refresh: bool = False) -> dict | None:
        self._check()
        if refresh:
            self.playbook_content = f"Playbook from {len(self.items)} lessons."
        if self.playbook_content is None:
            return None
        return {"content": self.playbook_content, "last_refreshed_at": None, "is_stale": False}

    async def close(self) -> None:
        return None


# ---- a small seeded history --------------------------------------------------------------------------

PLAYS = {
    "PLAY-01": ("SSO integration workshop", "security", ["sso"]),
    "PLAY-02": ("Security review pack", "security", ["security_review"]),
    "PLAY-03": ("Discount offer", "commercial", ["pricing"]),
    "PLAY-04": ("Executive sponsor call", "stakeholder", []),
    "PLAY-05": ("Reference customer call", "proof", []),
    "PLAY-06": ("Pricing walkthrough", "commercial", ["pricing"]),
    "PLAY-07": ("Mutual action plan", "process", []),
    "PLAY-09": ("Feature roadmap session", "proof", ["feature_gap"]),
}

ENGAGED = ("Dana Ortiz", "VP Engineering", "champion", True, False)
SILENT = ("Dana Ortiz", "VP Engineering", "champion", False, False)
BUYER = ("Chris Vale", "CFO", "neutral", True, True)


def add_catalogue(db: Database, codes: list[str] | None = None) -> None:
    for code in codes or list(PLAYS):
        name, category, addresses = PLAYS[code]
        add_play(db, code, name, category=category, addresses=addresses)


def closed(db: Database, name: str, account: str, result: str, **kw) -> int:
    kw.setdefault("closed_on", date(2026, 3, 1))
    return add_deal(db, name=name, account=account, result=result, **kw)


def seed_fintech_history(db: Database) -> dict[str, int]:
    """Four similar fintech mid-market deals and one off-topic retail deal. SSO is the pattern: it was
    resolved in two wins, left unresolved in a loss, and left unresolved in a win (noise)."""
    add_catalogue(db)
    ids = {
        "won_sso_fixed": closed(db, "Northwind", "Northwind Pay", "won", objections=[("sso", "addressed"), ("security_review", "addressed")],
                                competitors=["Brightline"], plays=["PLAY-01", "PLAY-02"], stakeholders=[ENGAGED, BUYER]),
        "won_sso_fixed_2": closed(db, "Orbital", "Orbital Bank", "won", objections=[("sso", "addressed")],
                                  plays=["PLAY-01"], stakeholders=[ENGAGED]),
        "lost_sso": closed(db, "Quill", "Quill Capital", "lost", loss_reason="security_compliance",
                           objections=[("sso", "unresolved"), ("security_review", "raised")], competitors=["Brightline"],
                           plays=["PLAY-03"], stakeholders=[SILENT]),
        "won_despite": closed(db, "Harbor", "Harbor Lending", "won", objections=[("sso", "unresolved")],
                              plays=["PLAY-02"], stakeholders=[ENGAGED]),
        "offtopic": closed(db, "Pinecone Retail", "Pinecone", "won", industry="retail", segment="smb",
                           objections=[("pricing", "addressed")], plays=["PLAY-06"], stakeholders=[ENGAGED]),
    }
    return ids


def add_open_fintech(db: Database, name: str = "Acme Pay", account: str = "Acme Payments") -> int:
    return add_deal(
        db, name=name, account=account, industry="fintech", segment="mid_market",
        objections=[("sso", "unresolved"), ("security_review", "raised")], competitors=["Brightline"],
        stakeholders=[SILENT], plays=[],
    )
