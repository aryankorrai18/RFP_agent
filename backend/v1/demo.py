"""Seed a demo workspace with a fictional company's history, with no model calls.

Larkspur Data (fictional) has 8 past proposals from 2023 to 2026: 4 won and 4 lost (technical fit,
price, response quality, incumbent), with competing answers to the same questions, some specific and
current, some vague or out of date. The questions and answers are taken exactly as they appear in each
document (samples/corpus_v2/demo_seed.json), so nothing is extracted by a
model. The workspace then syncs as usual: Hindsight stores the 45 answers, and the won/lost results
become lessons in its lessons bank. That uses Hindsight, not Gemini.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from typing import TYPE_CHECKING, Any

from ..config import ROOT
from ..core import PipelineError
from . import library

if TYPE_CHECKING:
    from .context import V1Context

SEED = ROOT / "samples" / "corpus_v2" / "demo_seed.json"
VIRTUSA_CYBER_SEED = ROOT / "samples" / "virtusa_demo" / "cybersecurity_seed.json"
SAMPLE_FACT_SHEET = ROOT / "data" / "fact_sheet.json"


def seed_demo(ctx: V1Context) -> dict[str, Any]:
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    facts = ctx.settings.fact_sheet_path
    if not facts.exists():  # the demo company's official facts, as a copy the workspace can edit
        facts.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SAMPLE_FACT_SHEET, facts)
    imported = already = answers = 0
    for item in seed["proposals"]:
        try:
            created = library.import_prepared_proposal(
                ctx, filename=item["file"], data=(SEED.parent / item["file"]).read_bytes(), client=item["client"],
                industry=item["industry"], submitted_on=date.fromisoformat(item["submitted_on"]), result=item["result"],
                loss_reason=item["loss_reason"], pairs=item["pairs"],
            )
        except PipelineError as exc:
            if exc.code != "already_imported":
                raise
            already += 1
            continue
        imported += 1
        answers += len(created)
    ctx.schedule_sync()
    return {"proposals": imported, "already_there": already, "answers": answers, "sample_rfps": seed["sample_rfps"],
            "company": seed["vendor"]}


def seed_virtusa_cyber(ctx: V1Context) -> dict[str, Any]:
    """Add focused, explicitly synthetic cybersecurity evidence without a model extraction call."""
    seed = json.loads(VIRTUSA_CYBER_SEED.read_text(encoding="utf-8"))
    imported = already = answers = 0
    for item in seed["proposals"]:
        document = (VIRTUSA_CYBER_SEED.parent / item["file"]).read_bytes()
        try:
            created = library.import_prepared_proposal(
                ctx,
                filename=item["file"],
                data=document,
                client=item["client"],
                industry=item["industry"],
                submitted_on=date.fromisoformat(item["submitted_on"]),
                result=item["result"],
                loss_reason=item["loss_reason"],
                pairs=_prepared_markdown_pairs(document.decode("utf-8")),
            )
        except PipelineError as exc:
            if exc.code != "already_imported":
                raise
            already += 1
            continue
        imported += 1
        answers += len(created)
    ctx.schedule_sync()
    return {
        "pack": seed["pack"],
        "disclosure": seed["disclosure"],
        "proposals": imported,
        "already_there": already,
        "answers": answers,
        "uses_model_calls": False,
    }


def _prepared_markdown_pairs(text: str) -> list[dict[str, str]]:
    """Read the deliberately simple Question/Response sections in a bundled demo document."""
    pairs: list[dict[str, str]] = []
    for block in text.split("\n## ")[1:]:
        heading, _, body = block.partition("\n")
        question_marker, response_marker = "Question: ", "\n\nResponse: "
        if question_marker not in body or response_marker not in body:
            continue
        question, answer = body.split(response_marker, 1)
        reference, _, section = heading.partition(". ")
        pairs.append({
            "section": section.strip(),
            "reference": reference.strip(),
            "question": question.removeprefix(question_marker).strip(),
            "answer": answer.split("\n\nEnd of synthetic demonstration proposal.", 1)[0].strip(),
        })
    return pairs
