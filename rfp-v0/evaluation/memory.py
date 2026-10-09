"""Local stand-ins for the two Hindsight banks, so the evaluation needs no server and no model.

The product's real recall is semantic (Hindsight). This is lexical: stemmed word overlap weighted by how rare each word
is. It is a weaker finder of paraphrases than the real thing, and the report says so. What it keeps is what the
evaluation is about: the product's own code decides which of the recalled answers to trust, in what order, and what the
model is shown. Ties go to the lower answer id, like an arbitrary but stable ordering, and which of two near-identical
answers has the lower id is drawn per world (see world.py), so ties favour no system.
"""

from __future__ import annotations

import math
import re

from rfp_assistant.api.v1.lessons import Brief, LessonHit
from rfp_assistant.api.v1.memory import MemoryItem, RecallHit

WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "do", "you", "your", "is", "are", "we", "our", "with", "how",
        "what", "via", "by", "on", "at", "be", "as", "it", "its", "that", "this", "which", "when", "does", "state", "per"}


def words(text: str) -> set[str]:
    out = set()
    for word in WORD.findall(text.lower()):
        if word in STOP:
            continue
        out.add(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return out


class LexicalMemory:
    """The answer bank: question and answer text, ranked by rarity-weighted word overlap with the query."""

    def __init__(self) -> None:
        self.docs: dict[str, MemoryItem] = {}
        self._words: dict[str, set[str]] = {}

    async def ensure_bank(self) -> None:
        return None

    async def retain(self, item: MemoryItem) -> None:
        self.docs[item.code] = item
        self._words[item.code] = words(f"{item.question} {item.answer}")

    async def delete(self, code: str) -> None:
        self.docs.pop(code, None)
        self._words.pop(code, None)

    async def recall(self, query: str, limit: int) -> list[RecallHit]:
        q = words(query)
        n = max(1, len(self.docs))
        idf = {w: math.log(1 + n / (1 + sum(1 for d in self._words.values() if w in d))) for w in q}
        total = sum(idf.values()) or 1.0
        scored = sorted(((sum(idf[w] for w in q & self._words[code]) / total, code) for code in self.docs),
                        key=lambda pair: (-pair[0], pair[1]))
        return [RecallHit(answer_code=code, rank=i + 1, final=score, semantic=score) for i, (score, code) in enumerate(scored[:limit])]

    async def healthy(self) -> bool:
        return True

    async def extraction_mode(self) -> str | None:
        return "lexical"

    async def close(self) -> None:
        return None


class LexicalLessons:
    """The lessons bank: keeps what the product retains and recalls those carrying any wanted tag, best word match first."""

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    async def retain(self, items: list[dict]) -> None:
        for item in items:
            self.items[item["document_id"]] = item

    async def recall(self, query: str, tags: list[str], limit: int = 20) -> list[LessonHit]:
        wanted, q = set(tags), words(query)
        matches = [i for i in self.items.values() if wanted & set(i["tags"])]
        matches.sort(key=lambda i: -len(q & words(i["content"])))
        return [LessonHit(text=i["content"], tags=list(i["tags"]), document_id=i["document_id"], type="world", rank=n)
                for n, i in enumerate(matches[:limit], start=1)]

    async def reflect(self, query: str, tags: list[str] | None = None) -> Brief:
        return Brief(text="", based_on=[])

    async def playbook(self, refresh: bool = False) -> dict | None:
        return None

    async def close(self) -> None:
        return None
