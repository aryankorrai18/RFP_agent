"""Loads a synthetic world into a real (temporary) RFP Memory Assistant database, runs the product's own retrieval and
drafting on it, and marks the results against the planted rules.

Everything the product does is the product's own code: importing past proposals, outcome credit, review statistics,
lessons, the sync to memory, retrieval and ranking, drafting, and the grounding checks. Only the two memory banks are
local lexical stand-ins (memory.py), and the one place a model is used is `draft_answer`."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from rfp_assistant.api.v1 import library
from rfp_assistant.api.v1.context import V1Context
from rfp_assistant.api.v1.db import (
    Answer, AnswerStats, Database, MemoryEvent, PastProposal, Project, RequirementRow,
)
from rfp_assistant.api.v1.retrieval import Retrieval, retrieve
from rfp_assistant.api.v1.storage import store_document
from rfp_assistant.config import Settings
from rfp_assistant.schemas import Fact
from sqlalchemy import select

from .memory import LexicalLessons, LexicalMemory
from .world import LIBRARY_KINDS, PROJECTS, PROPOSALS, WORD_LIMIT, Question, World

TOP = 3


@dataclass
class Loaded:
    world: World
    ctx: V1Context
    refs: dict[tuple[str, str], str]  # (proposal, topic) -> answer code
    projects: dict[str, Project]
    requirements: dict[str, RequirementRow]  # question id -> row
    facts: list[Fact]
    holder: dict

    @property
    def settings(self) -> Settings:
        return self.holder["settings"]

    def use(self, **changes) -> None:  # noqa: ANN003
        self.holder["settings"] = replace(self.holder["settings"], **changes)

    async def close(self) -> None:
        """Release the database files so the temporary folder can be removed (Windows keeps them locked)."""
        await self.ctx.shutdown()
        self.ctx.db.engine.dispose()


def temp_folder() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="rfp-eval-", ignore_cleanup_errors=True)


def plant_history(ctx: V1Context, world: World, refs: dict[tuple[str, str], str], rounds: int | None) -> int:
    """Write the first `rounds` rounds of the world's review history the way the product's review path writes them:
    the same counters on the answer's statistics and the same event text, which later becomes a lesson."""
    planted = 0
    with ctx.db.session() as session:
        for ref, action, tags in world.history(rounds):
            answer = session.scalars(select(Answer).where(Answer.id == int(refs[ref].split("-")[1]))).one()
            stats = session.get(AnswerStats, answer.id)
            stats.times_used = (stats.times_used or 0) + 1
            if action == "accepted":
                stats.accepted = (stats.accepted or 0) + 1
            elif action == "rewritten":
                stats.rewritten = (stats.rewritten or 0) + 1
            else:
                stats.rejected = (stats.rejected or 0) + 1
            if "outdated" in tags:
                stats.outdated_signals = (stats.outdated_signals or 0) + 1
                stats.suggested_supersede = True
            session.add(MemoryEvent(kind="review_signal", answer_id=answer.id,
                                    detail=f"{answer.code} was {action}" + (f" ({', '.join(tags)})" if tags else "") + "."))
            planted += 1
        session.commit()
    return planted


async def load_world(world: World, folder: Path, *, base: Settings | None = None, llm_provider=None,  # noqa: ANN001
                     history: int | None = None) -> Loaded:
    """A fresh database holding the world: past proposals imported through the product's own importer, review history
    planted, lessons and answers synced to the local memory banks, and one project per client with its questions."""
    settings = replace(base or Settings(), db_path=folder / "eval.db", uploads_dir=folder / "uploads", lessons_enabled=True,
                       retrieval_mode="hindsight", retrieval_top_k=TOP, draft_concurrency=2)
    holder = {"settings": settings}
    ctx = V1Context(db=Database(settings.db_path), memory=LexicalMemory(), lessons=LexicalLessons(),
                    settings_provider=lambda: holder["settings"], llm_provider=llm_provider or (lambda _s: None))
    refs: dict[tuple[str, str], str] = {}
    for key in world.proposals:
        client, industry, when, result, reason = PROPOSALS[key]
        rows = world.pairs[key]
        created = library.import_prepared_proposal(
            ctx, filename=f"{key}.docx", data=f"{world.seed}-{key}".encode(), client=client, industry=industry,
            submitted_on=when, result=result, loss_reason=reason,
            pairs=[{"question": question, "answer": answer} for _topic, question, answer in rows])
        for (topic, _q, _a), answer in zip(rows, created, strict=True):
            refs[(key, topic)] = answer.code
    plant_history(ctx, world, refs, history)
    await ctx.sync()

    projects: dict[str, Project] = {}
    requirements: dict[str, RequirementRow] = {}
    for name, (client, industry) in PROJECTS.items():
        document = store_document(ctx, "rfp", f"{name}.docx", f"{world.seed}-project-{name}".encode())
        with ctx.db.session() as session:
            project = Project(document_id=document.id, name=f"{client} RFP", client=client, industry=industry, state="requirements_extracted")
            session.add(project)
            session.flush()
            for order, question in enumerate([q for q in world.questions if q.project == name], start=1):
                row = RequirementRow(project_id=project.id, order=order, question=question.text, mandatory=True, word_limit=WORD_LIMIT)
                session.add(row)
                session.flush()
                requirements[question.id] = row
            session.commit()
        projects[name] = project
    facts = [Fact(id=f.id, topic=f.topic, statement=f.statement, valid_from=date.fromisoformat(f.valid_from) if f.valid_from else None)
             for f in world.facts]
    return Loaded(world, ctx, refs, projects, requirements, facts, holder)


# ---- retrieval: the product's own and the baselines ---------------------------------------------------------

async def product_retrieval(loaded: Loaded, question: Question, mode: str, *, exclude_lost: bool = True) -> Retrieval:
    """What the product would put in front of the model for this question (no model call)."""
    s = loaded.settings
    client, industry = PROJECTS[question.project]
    return await retrieve(
        question.text, memory=loaded.ctx.memory, db=loaded.ctx.db, k=TOP, mode=mode, client=client, industry=industry,
        freshness_half_life_days=s.retrieval_freshness_half_life_days, lessons=loaded.ctx.lessons, relevance=s.retrieval_relevance,
        min_share=s.retrieval_relevance_min_share, exclude_lost_proposals=exclude_lost)


async def raw_search(loaded: Loaded, question: Question) -> list[str]:
    """Plain search: the memory's own order, no checks and no memory of outcomes."""
    hits = await loaded.ctx.memory.recall(question.text, limit=TOP)
    return [h.answer_code for h in hits]


async def newest_first(loaded: Loaded, question: Question) -> list[str]:
    """A simple heuristic: among the answers the search scores about as high as its best (within 80%), drop those from lost
    proposals and put the newest proposal first."""
    hits = await loaded.ctx.memory.recall(question.text, limit=12)
    with loaded.ctx.db.session() as session:
        dated = []
        for hit in hits:
            row = session.get(Answer, int(hit.answer_code.split("-")[1]))
            proposal = session.get(PastProposal, row.past_proposal_id)
            if proposal.result != "lost":
                dated.append((proposal.submitted_on, hit.rank, hit.answer_code))
    top_score = hits[0].final if hits else 0.0
    keep = {h.answer_code for h in hits if (h.final or 0) >= 0.8 * (top_score or 1)}
    return [code for _d, _r, code in sorted(dated, key=lambda t: (-t[0].toordinal(), t[1])) if code in keep][:TOP]


def mark_retrieval(question: Question, ranked: list[str], refs: dict[tuple[str, str], str]) -> dict:
    """Mark a ranked list of answer codes against the question's right and wrong library answers."""
    gold = refs.get(question.gold_ref) if question.gold_ref else None
    bad = refs.get(question.bad_ref) if question.bad_ref else None
    top = ranked[:TOP]
    rank = top.index(gold) + 1 if gold in top else None
    return {"hit1": bool(top) and top[0] == gold, "hit3": gold in top, "rr": 1 / rank if rank else 0.0,
            "bad1": bool(top) and top[0] == bad, "bad_shown": bad in top, "empty": not top, "ranked": top}


# ---- drafting: marking a drafted answer ----------------------------------------------------------------------

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def value_in(text: str, value: str) -> bool:
    """The value appears as a whole phrase (so '99.9%' is not found inside '99.99%' and 'Type I' not inside 'Type II')."""
    return re.search(r"(?<![\w.,-])" + re.escape(value) + r"(?!\w|[.,]\d)", text, re.IGNORECASE) is not None


def mark_draft(question: Question, draft: dict, evidence: str, gold_code: str | None) -> dict:
    """Mark one drafted answer. `draft` holds status, answer, sources, flags and retrieved ids; `evidence` is the text the
    model was shown (facts, past answers, the question). Returns the outcome and the deterministic checks behind it."""
    answer = draft.get("answer") or ""
    status = draft["status"]
    answered = status == "drafted" and bool(answer.strip())
    flags = set(draft.get("flags") or [])
    cited = set(draft.get("sources") or [])
    invented = [n.rstrip(",") for n in NUMBER.findall(answer) if n.rstrip(",") not in evidence]
    marks = {
        "status": status, "answered": answered, "abstained": status == "needs_sme",
        "has_gold": bool(question.gold_value) and value_in(answer, question.gold_value),
        "stale_value": any(value_in(answer, v) for v in question.forbidden),
        "cites_gold": (gold_code or question.gold_fact) in cited if (gold_code or question.gold_fact) else False,
        "bad_citation": "invalid_citation" in flags, "over_limit": "over_word_limit" in flags,
        "unsupported": "unsupported_claims" in flags or "evidence_unsupported" in flags,
        "invented_number": bool(invented), "words": draft.get("word_count") or 0,
        "shown": list(draft.get("shown") or []), "failed": status == "failed",
    }
    if question.kind == "unanswerable":
        marks["correct"] = marks["abstained"] and not marks["invented_number"]  # an SME handoff template is not an answer
    else:
        marks["correct"] = (answered and marks["has_gold"] and not marks["stale_value"] and marks["cites_gold"]
                            and not marks["bad_citation"])
    marks["wrong_value"] = answered and not marks["correct"] and question.kind != "unanswerable" and (marks["stale_value"] or not marks["has_gold"])
    marks["fabricated"] = question.kind == "unanswerable" and answered
    return marks


def library_question(question: Question) -> bool:
    return question.kind in LIBRARY_KINDS
