"""One-case, low-token live smoke test for V4.

This intentionally makes one Hindsight recall, two production drafting calls, and one compact
Gemini judge call. It is a wiring check, not the frozen V4 acceptance gate.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from datetime import date
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel
from sqlalchemy import select

from backend.llm_gemini import GeminiLLM, json_schema_for
from backend.main import get_settings
from backend.schemas import PastAnswer, Requirement
from backend.service import load_default_fact_sheet
from backend.v1.db import Answer, AnswerStats, Database, parse_answer_code
from backend.v1.library import backfill_historical_outcomes
from backend.v1.memory import HindsightMemory
from backend.v1.ranking import rank_factors


class JudgeVerdict(BaseModel):
    preferred: Literal["A", "B", "tie"]
    reason: str
    a_grounded: bool
    b_grounded: bool


def _tokens(usage) -> dict:  # noqa: ANN001
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": usage.cache_read_input_tokens,
    }


async def judge_only() -> None:
    """One request, no SDK retries: cheapest possible validation of the live V4 judge."""
    settings = get_settings()
    model = os.environ.get("RFP_LIVE_SMOKE_MODEL", "").strip() or settings.model
    client = genai.Client(http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=1)
    ))
    prompt = (
        "Requirement: Is automatic user provisioning via SCIM supported?\n"
        "Reference source: We support SCIM 2.0 provisioning with Okta.\n\n"
        "Answer A: Yes. We provide automatic lifecycle management for every identity provider.\n"
        "Answer B: Yes. We support SCIM 2.0 provisioning with Okta. [ANS-0002]\n\n"
        "Blindly choose A, B, or tie. Prefer direct, source-supported language. "
        "Keep the reason under 20 words."
    )
    response = await client.aio.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction="You are a strict, neutral RFP quality evaluator.",
            response_mime_type="application/json",
            response_json_schema=json_schema_for(JudgeVerdict),
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            max_output_tokens=128,
        ),
    )
    verdict = JudgeVerdict.model_validate_json(response.text)
    usage = getattr(response, "usage_metadata", None)
    print(json.dumps({
        "status": "live_judge_smoke",
        "model": getattr(response, "model_version", None) or model,
        "external_calls": {"gemini_judge": 1},
        "expected": "B",
        "observed": verdict.preferred,
        "passed": verdict.preferred == "B" and verdict.b_grounded and not verdict.a_grounded,
        "reason": verdict.reason,
        "tokens": {
            "input": getattr(usage, "prompt_token_count", 0) or 0,
            "output": (getattr(usage, "candidates_token_count", 0) or 0)
            + (getattr(usage, "thoughts_token_count", 0) or 0),
        },
        "v4_complete": False,
    }, indent=2))
    await client.aio.aclose()


async def main() -> None:
    settings = get_settings()
    if settings.provider != "gemini":
        raise SystemExit("The low-token smoke runner currently requires the configured Gemini provider.")

    db = Database(settings.db_path)
    seeded = backfill_historical_outcomes(db)
    memory = HindsightMemory(
        settings.hindsight_url, settings.hindsight_bank, api_key=settings.hindsight_api_key
    )
    query = "Is automatic user provisioning via SCIM supported?"
    hits = await memory.recall(query, limit=12)  # exactly one live Hindsight search
    ids = [i for i in (parse_answer_code(hit.answer_code) for hit in hits) if i is not None]
    with db.session() as session:
        answers = {
            answer.id: answer
            for answer in session.scalars(select(Answer).where(Answer.id.in_(ids)))
            if answer.live
        }
        stats = {
            row.answer_id: row
            for row in session.scalars(select(AnswerStats).where(AnswerStats.answer_id.in_(ids)))
        }
    candidates = [
        (hit, answers[answer_id])
        for hit in hits
        if (answer_id := parse_answer_code(hit.answer_code)) in answers
    ]
    if not candidates:
        raise SystemExit("Hindsight returned no live SQLite-backed candidates for the smoke case.")

    plain = sorted(candidates, key=lambda pair: pair[0].rank)[: settings.retrieval_top_k]
    scored = [
        (
            rank_factors(
                answer, stats.get(answer.id), recall_rank=hit.rank,
                client="Harborview Credit Union", industry="finance",
                half_life_days=settings.retrieval_freshness_half_life_days,
            ),
            hit,
            answer,
        )
        for hit, answer in candidates
    ]
    outcome = sorted(scored, key=lambda item: (-item[0].score, item[1].rank))[: settings.retrieval_top_k]

    def past(rows) -> list[PastAnswer]:  # noqa: ANN001
        normalized = [(row[-2], row[-1]) for row in rows]
        return [
            PastAnswer(
                id=answer.code, question=answer.question, answer=answer.answer,
                client=answer.client, industry=answer.industry,
                approved_on=answer.updated_at.date(),
            )
            for _hit, answer in normalized
        ]

    sheet = load_default_fact_sheet(settings)
    facts = [fact for fact in sheet.facts if fact.is_live(date.today())]
    requirement = Requirement(
        id="LIVE-V4-001", question=query, section="Identity", reference="smoke",
        mandatory=True, word_limit=80,
    )
    llm = GeminiLLM(settings)
    plain_draft = await llm.draft_answer(sheet.company, facts, requirement, past(plain))
    outcome_draft = await llm.draft_answer(sheet.company, facts, requirement, past(outcome))

    variants = [("plain", plain_draft.output.answer), ("outcome", outcome_draft.output.answer)]
    random.Random(20260928).shuffle(variants)
    judge_prompt = (
        "Blindly compare two RFP draft answers. Prefer the answer that is complete, directly "
        "answers the requirement, and avoids claims not supported by its citations. A tie is "
        "allowed. Keep the reason under 30 words.\n\n"
        f"Requirement: {query}\n\n"
        f"Answer A:\n{variants[0][1]}\n\nAnswer B:\n{variants[1][1]}"
    )
    response = await llm.client.aio.models.generate_content(
        model=settings.model,
        contents=[judge_prompt],
        config=types.GenerateContentConfig(
            system_instruction="You are a strict, neutral RFP quality evaluator.",
            response_mime_type="application/json",
            response_json_schema=json_schema_for(JudgeVerdict),
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            max_output_tokens=256,
        ),
    )
    verdict = JudgeVerdict.model_validate_json(response.text)
    usage = getattr(response, "usage_metadata", None)
    judge_tokens = {
        "input": getattr(usage, "prompt_token_count", 0) or 0,
        "output": (getattr(usage, "candidates_token_count", 0) or 0)
        + (getattr(usage, "thoughts_token_count", 0) or 0),
        "cache_read": getattr(usage, "cached_content_token_count", 0) or 0,
    }
    winner = "tie" if verdict.preferred == "tie" else variants[0 if verdict.preferred == "A" else 1][0]
    result = {
        "status": "live_smoke_only",
        "case": requirement.id,
        "model": getattr(response, "model_version", None) or settings.model,
        "external_calls": {"hindsight_recall": 1, "gemini_draft": 2, "gemini_judge": 1},
        "historical_track_records_seeded": seeded,
        "plain_sources": [answer.code for _hit, answer in plain],
        "outcome_sources": [answer.code for _factors, _hit, answer in outcome],
        "blind_mapping": {"A": variants[0][0], "B": variants[1][0]},
        "judge": {
            "preferred_label": verdict.preferred, "winner": winner, "reason": verdict.reason,
            "a_grounded": verdict.a_grounded, "b_grounded": verdict.b_grounded,
        },
        "tokens": {
            "plain_draft": _tokens(plain_draft.usage),
            "outcome_draft": _tokens(outcome_draft.usage),
            "judge": judge_tokens,
        },
        "v4_complete": False,
        "note": "A one-case live smoke does not replace the frozen representative V4 gate.",
    }
    print(json.dumps(result, indent=2))
    await memory.close()


if __name__ == "__main__":
    asyncio.run(judge_only() if "--judge-only" in sys.argv else main())
