"""Frozen, token-free retrieval evaluation for the V4 gate.

This module proves the corpus, ranking, ablations and deterministic metrics without spending Gemini
or Hindsight tokens. The live parts of V4 (the blind pairwise judge and the human spot check) run in
the app and are recorded by eval/v4/record_live_judge.py; this report reads that record and only
calls V4 complete when the deterministic gate passed and both live parts were run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from backend.config import ROOT

from backend.v1.db import Answer, AnswerStats
from backend.v1.ranking import rank_factors

TOPICS = (
    "sso", "scim", "encryption", "penetration_testing", "data_residency",
    "disaster_recovery", "support", "implementation", "api", "audit_logs",
)
INDUSTRIES = ("finance", "healthcare", "insurance", "public_sector")
LIVE_RESULTS = ROOT / "eval" / "v4" / "live_judge_results.json"


@dataclass(frozen=True)
class Candidate:
    id: str
    semantic_rank: int
    client: str
    industry: str
    age_days: int
    times_used: int
    accepted: int = 0
    light_edits: int = 0
    heavy_edits: int = 0
    rewritten: int = 0
    rejected: int = 0
    outcome_credit: float = 0.0
    outdated: bool = False


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    client: str
    industry: str
    best_answer_id: str
    candidates: tuple[Candidate, ...]


def generate_corpus() -> list[EvalCase]:
    """Five held-out RFPs x 40 questions, backed by 20 synthetic proposal identities."""
    cases: list[EvalCase] = []
    for rfp in range(5):
        client = f"Client-{rfp + 1}"
        industry = INDUSTRIES[rfp % len(INDUSTRIES)]
        for question_no in range(40):
            topic = TOPICS[question_no % len(TOPICS)]
            candidates = []
            # The semantically closest three are deliberately poor; the known-best answer is
            # fourth. Outcome ranking must discover it from review history.
            profiles = {
                1: dict(times_used=8, rejected=8, outdated=True, age_days=1500),
                2: dict(times_used=8, heavy_edits=8, age_days=500),
                3: dict(times_used=8, rewritten=8, age_days=350),
                4: dict(times_used=8, accepted=8, outcome_credit=0.1, age_days=60),
            }
            for rank in range(1, 13):
                proposal = (question_no + rank + rfp) % 20
                profile = profiles.get(rank, dict(times_used=0, age_days=180 + rank * 20))
                candidates.append(Candidate(
                    id=f"ANS-E{rfp + 1:02d}-{question_no + 1:03d}-{proposal + 1:02d}",
                    semantic_rank=rank,
                    client=client if rank == 4 else f"History-{proposal + 1}",
                    industry=industry if rank in (4, 7) else INDUSTRIES[proposal % len(INDUSTRIES)],
                    **profile,
                ))
            cases.append(EvalCase(
                id=f"RFP-{rfp + 1}-Q{question_no + 1:03d}",
                question=f"{topic.replace('_', ' ').title()} requirement {question_no + 1}",
                client=client,
                industry=industry,
                best_answer_id=candidates[3].id,
                candidates=tuple(candidates),
            ))
    return cases


def _objects(candidate: Candidate, now: datetime) -> tuple[Answer, AnswerStats]:
    numeric_id = abs(hash(candidate.id)) % 2_000_000_000
    answer = Answer(
        id=numeric_id, question="q", answer="a", source="library", client=candidate.client,
        industry=candidate.industry, status="approved", superseded_by=None,
        updated_at=now - timedelta(days=candidate.age_days), hindsight_status="retained",
    )
    stats = AnswerStats(
        answer_id=numeric_id, times_used=candidate.times_used, accepted=candidate.accepted,
        light_edits=candidate.light_edits, heavy_edits=candidate.heavy_edits,
        rewritten=candidate.rewritten, rejected=candidate.rejected,
        outcome_credit=candidate.outcome_credit, suggested_supersede=candidate.outdated,
    )
    return answer, stats


def rank_case(case: EvalCase, variant: str) -> list[str]:
    if variant == "none":
        return []
    if variant == "plain":
        return [c.id for c in sorted(case.candidates, key=lambda c: c.semantic_rank)]
    if variant not in {"outcome", "no_review_signals", "no_context", "no_freshness"}:
        raise ValueError(f"Unknown evaluation variant {variant!r}")
    now = datetime(2026, 9, 28, tzinfo=UTC)
    scored = []
    for candidate in case.candidates:
        answer, stats = _objects(candidate, now)
        if variant == "no_review_signals":
            stats = AnswerStats(answer_id=stats.answer_id)
        factors = rank_factors(
            answer, stats, recall_rank=candidate.semantic_rank,
            client=None if variant == "no_context" else case.client,
            industry=None if variant == "no_context" else case.industry,
            now=now, half_life_days=10_000_000 if variant == "no_freshness" else 730,
        )
        scored.append((factors.score, candidate.semantic_rank, candidate.id))
    return [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))]


def live_status(path: Path = LIVE_RESULTS) -> dict:
    """What eval/v4/record_live_judge.py recorded about the live judge and the spot check."""
    try:
        totals = json.loads(path.read_text(encoding="utf-8")).get("totals", {})
    except (OSError, ValueError):
        return {"live_pairwise_judge": "not_run", "human_spot_check": "not_run"}
    return {
        "live_pairwise_judge": "run" if totals.get("verdicts") else "not_run",
        "human_spot_check": "run" if totals.get("spot_checked") else "not_run",
        "live_totals": totals,
    }


def evaluate(cases: list[EvalCase] | None = None, live_results: Path = LIVE_RESULTS) -> dict:
    cases = cases or generate_corpus()
    result = {"cases": len(cases), "variants": {}}
    for variant in ("none", "plain", "outcome", "no_review_signals", "no_context", "no_freshness"):
        rankings = [rank_case(case, variant) for case in cases]
        top1 = sum(bool(ranking) and ranking[0] == case.best_answer_id for ranking, case in zip(rankings, cases))
        top3 = sum(case.best_answer_id in ranking[:3] for ranking, case in zip(rankings, cases))
        result["variants"][variant] = {
            "right_source_top1": top1 / len(cases),
            "right_source_top3": top3 / len(cases),
        }
    margin = (
        result["variants"]["outcome"]["right_source_top3"]
        - result["variants"]["plain"]["right_source_top3"]
    )
    result["deterministic_gate"] = {
        "required_margin": 0.20,
        "observed_margin": round(margin, 4),
        "passed": margin >= 0.20,
    }
    result.update(live_status(live_results))
    result["v4_complete"] = (
        result["deterministic_gate"]["passed"]
        and result["live_pairwise_judge"] == "run"
        and result["human_spot_check"] == "run"
    )
    return result


def corpus_manifest(cases: list[EvalCase] | None = None) -> dict:
    cases = cases or generate_corpus()
    return {"cases": len(cases), "first": asdict(cases[0]), "last": asdict(cases[-1])}
