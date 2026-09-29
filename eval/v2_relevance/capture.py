"""Capture the frozen input for the V2 relevance evaluation (Hindsight lookups only, no model calls).

For every question in the two corpus RFPs (samples/corpus_v2/answer_key.json), record what Hindsight
returns in the open workspace: the library candidates with their similarity scores, and the lessons
it recalls about those candidates. evaluate.py then scores any ranking formula on this same input,
so formulas are compared on identical candidates and no further calls are needed.

    .\\.venv\\Scripts\\python -m eval.v2_relevance.capture      (open a clean demo workspace first)
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from backend import workspaces
from backend.config import ROOT
from backend.main import base_settings
from backend.v1.db import Answer, Database
from backend.v1.lessons import HindsightLessons, answer_tag
from backend.v1.memory import HindsightMemory

HERE = Path(__file__).parent
CORPUS = ROOT / "samples" / "corpus_v2"
OUT = HERE / "recall_cache.json"
CANDIDATES = 12  # what retrieval asks Hindsight for (k=3 x overfetch 4)


def corpus_keys() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("make_corpus", CORPUS / "make_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {answer: key for key, (_question, answer, _kind) in module.A.items()}


async def main() -> None:
    settings = workspaces.apply_workspace(base_settings())
    space = workspaces.active()
    db = Database(settings.db_path)
    memory = HindsightMemory(settings.hindsight_url, settings.hindsight_bank, api_key=settings.hindsight_api_key)
    lessons = HindsightLessons(settings.hindsight_url, settings.hindsight_lessons_bank, api_key=settings.hindsight_api_key)
    key_of = corpus_keys()
    with db.session() as session:
        answers = {a.code: a for a in session.query(Answer).all()}
    answer_key = json.loads((CORPUS / "answer_key.json").read_text(encoding="utf-8"))
    try:
        # Hindsight turns lessons into facts in the background; wait until they are recallable.
        for _ in range(30):
            probe = await lessons.recall("penetration testing", tags=[answer_tag(c) for c in answers])
            if sum(any(t.startswith("signal:") for t in h.tags) for h in probe) >= 3:
                break
            time.sleep(10)
        questions = []
        for rfp_name, rfp in answer_key["rfps"].items():
            for item in rfp["questions"]:
                hits = await memory.recall(item["question"], limit=CANDIDATES)
                codes = sorted({h.answer_code for h in hits if h.answer_code in answers})
                lesson_hits = await lessons.recall(item["question"], tags=[answer_tag(c) for c in codes]) if codes else []
                questions.append({
                    "rfp": rfp_name, "client": rfp["client"], "industry": rfp["industry"], "ref": item["ref"],
                    "question": item["question"], "gap": item["gap"],
                    "preferred": item["preferred"]["key"] if item["preferred"] else None,
                    "traps": [t["key"] for t in item["traps"]],
                    "candidates": [h.as_dict() | {"key": key_of.get(answers[h.answer_code].answer) if h.answer_code in answers else None}
                                   for h in hits],
                    "lesson_hits": [{"text": h.text, "tags": h.tags, "document_id": h.document_id, "type": h.type, "rank": h.rank}
                                    for h in lesson_hits],
                })
                print(f"{item['ref']:>6} {len(hits)} candidates, {len(lesson_hits)} lessons | {item['question'][:60]}", flush=True)
    finally:
        await memory.close()
        await lessons.close()
    OUT.write_text(json.dumps({
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds"), "workspace": space.id if space else None,
        "library_bank": settings.hindsight_bank, "lessons_bank": settings.hindsight_lessons_bank,
        "questions": questions,
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(questions)} questions")


if __name__ == "__main__":
    asyncio.run(main())
