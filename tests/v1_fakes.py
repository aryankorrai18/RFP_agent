"""Offline stand-ins for V1: a fake Hindsight (FakeMemory) and a scripted model (FakeV1LLM)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from backend.config import Settings
from backend.llm import LLMError, LLMResult, TokenUsage
from backend.parser import ParsedDocument
from backend.schemas import (
    DraftClaimOut,
    DraftResult,
    ExtractedPair,
    ExtractedRequirement,
    ExtractionResult,
    Fact,
    JudgeResult,
    JudgeScores,
    PairsResult,
    PastAnswer,
    Requirement,
)
from backend.v1.context import V1Context
from backend.v1.db import Database
from backend.v1.lessons import Brief, LessonHit
from backend.v1.memory import MemoryItem, MemoryUnavailable, RecallHit

WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "do", "you", "your", "is", "are", "we", "our", "with", "how", "what", "via"}


def words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP}


class FakeMemory:
    """In-memory Hindsight. Ranks by word overlap. `available=False` simulates an outage."""

    def __init__(self) -> None:
        self.docs: dict[str, MemoryItem] = {}
        self.available = True
        self.retained: list[str] = []
        self.deleted: list[str] = []
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
        self.docs[item.code] = item
        self.retained.append(item.code)

    async def delete(self, code: str) -> None:
        self._check()
        self.docs.pop(code, None)
        self.deleted.append(code)

    async def recall(self, query: str, limit: int) -> list[RecallHit]:
        self._check()
        q = words(query)
        scored = sorted(
            ((len(q & words(f"{d.question} {d.answer}")) / (len(q) or 1), code) for code, d in self.docs.items()),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [RecallHit(answer_code=code, rank=i + 1, final=score, semantic=score) for i, (score, code) in enumerate(scored[:limit])]

    async def healthy(self) -> bool:
        return self.available

    async def extraction_mode(self) -> str | None:
        return "chunks" if self.available else None

    async def close(self) -> None:
        pass


def cite_first_past_answer(req: Requirement, past: list[PastAnswer] | None, instructions: str | None) -> DraftResult:
    """Default drafter: answer from the top past answer if one was offered, else ask an SME."""
    if past:
        top = past[0]
        text = top.answer + (f" [{instructions}]" if instructions else "")
        return DraftResult(answer=text, claims=[DraftClaimOut(text=top.answer, source_ids=[top.id])],
                           unsupported_claims=[], needs_sme=False, sme_question=None)
    return DraftResult(answer="", claims=[], unsupported_claims=[], needs_sme=True, sme_question=f"Who can answer: {req.question}")


def draft_text(message: str, label: str) -> str:
    match = re.search(rf"<draft_{label}>\n(.*?)\nCited sources:", message, re.S)
    return match.group(1) if match else ""


def verdict(winner: str, reason: str = "fake verdict") -> JudgeResult:
    good, weak = JudgeScores(accurate=5, answers_question=5, specific=5), JudgeScores(accurate=3, answers_question=3, specific=2)
    return JudgeResult(winner=winner, reason=reason, A=good if winner == "A" else weak, B=good if winner == "B" else weak)


def prefers_specific(message: str) -> JudgeResult:
    """Default judge: prefers the draft that names Ironbridge (the specific, accurate answer)."""
    a, b = "Ironbridge" in draft_text(message, "A"), "Ironbridge" in draft_text(message, "B")
    return verdict("A" if a and not b else "B" if b and not a else "tie")


@dataclass
class FakeV1LLM:
    pairs: list[ExtractedPair] = field(default_factory=list)
    requirements: list[ExtractedRequirement] = field(default_factory=list)
    drafter: Callable = cite_first_past_answer
    fail_pairs: LLMError | None = None
    fail_requirements: LLMError | None = None
    model: str = "fake-model"
    drafted: list[tuple[str, list[str] | None, str | None]] = field(default_factory=list)
    fact_sets: list[list[str]] = field(default_factory=list)
    temperatures: list[float | None] = field(default_factory=list)
    judger: Callable | None = None  # judge message -> JudgeResult or LLMError; default prefers_specific
    judged: list[str] = field(default_factory=list)  # every judge message received

    async def judge(self, system: str, message: str) -> LLMResult[JudgeResult]:
        self.judged.append(message)
        result = (self.judger or prefers_specific)(message)
        if isinstance(result, LLMError):
            raise result
        return LLMResult(output=result, model=f"{self.model}-judge", usage=TokenUsage(input_tokens=100, output_tokens=20))

    async def extract_pairs(self, document: ParsedDocument) -> LLMResult[PairsResult]:
        if self.fail_pairs:
            raise self.fail_pairs
        return LLMResult(output=PairsResult(pairs=self.pairs), model=self.model, usage=TokenUsage())

    async def extract_requirements(self, document: ParsedDocument) -> LLMResult[ExtractionResult]:
        if self.fail_requirements:
            raise self.fail_requirements
        return LLMResult(output=ExtractionResult(requirements=self.requirements), model=self.model, usage=TokenUsage())

    async def draft_answer(self, company: str, facts: list[Fact], requirement: Requirement,
                           past_answers: list[PastAnswer] | None = None, instructions: str | None = None,
                           *, temperature: float | None = None) -> LLMResult[DraftResult]:
        self.temperatures.append(temperature)
        self.drafted.append((requirement.id, [p.id for p in past_answers] if past_answers is not None else None, instructions))
        self.fact_sets.append([fact.id for fact in facts])
        result = self.drafter(requirement, past_answers, instructions)
        if isinstance(result, LLMError):
            raise result
        return LLMResult(output=result, model=self.model, usage=TokenUsage())


def pair(question: str, answer: str, **kw) -> ExtractedPair:
    return ExtractedPair(question=question, answer=answer, section=kw.get("section"), reference=kw.get("reference"))


def req(question: str, **kw) -> ExtractedRequirement:
    return ExtractedRequirement(question=question, section=kw.get("section"), mandatory=kw.get("mandatory"),
                                word_limit=kw.get("word_limit"), reference=kw.get("reference"))


class FakeLessons:
    """In-memory Hindsight lessons bank: recall filters by any matching tag and ranks by word overlap;
    reflect and the playbook return canned text built from what was retained."""

    def __init__(self) -> None:
        self.items: list[dict] = []
        self.available = True
        self.reflect_calls: list[tuple[str, list[str] | None]] = []
        self.playbook_content: str | None = None

    def _check(self) -> None:
        if not self.available:
            raise MemoryUnavailable("fake lessons bank is down")

    async def retain(self, items: list[dict]) -> None:
        self._check()
        by_id = {item["document_id"]: item for item in self.items}
        for item in items:
            by_id[item["document_id"]] = item  # same document_id replaces, like Hindsight
        self.items = list(by_id.values())

    async def recall(self, query: str, tags: list[str], limit: int = 20) -> list[LessonHit]:
        self._check()
        wanted, q = set(tags), words(query)
        matches = [i for i in self.items if wanted & set(i["tags"])]
        matches.sort(key=lambda i: -len(q & words(i["content"])))
        return [LessonHit(text=i["content"], tags=list(i["tags"]), document_id=i["document_id"], type="world", rank=n)
                for n, i in enumerate(matches[:limit], start=1)]

    async def reflect(self, query: str, tags: list[str] | None = None) -> Brief:
        self._check()
        self.reflect_calls.append((query, tags))
        relevant = [i for i in self.items if not tags or set(tags) & set(i["tags"])]
        return Brief(text=f"{len(relevant)} lessons remembered.",
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


def make_context(tmp_path, llm: FakeV1LLM, memory: FakeMemory | None = None, lessons: FakeLessons | None = None,
                 **settings_overrides) -> V1Context:
    defaults = {
        "db_path": tmp_path / "rfp.db",
        "uploads_dir": tmp_path / "uploads",
        "draft_concurrency": 2,
    }
    settings = replace(Settings(), **{**defaults, **settings_overrides})
    return V1Context(
        db=Database(settings.db_path),
        memory=memory or FakeMemory(),
        lessons=lessons,
        settings_provider=lambda: settings,
        llm_provider=lambda _settings: llm,
    )
