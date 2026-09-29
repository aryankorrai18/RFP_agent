"""Score ranking rules on the frozen V2 relevance input (recall_cache.json). No calls of any kind.

For each of the 22 corpus questions that have a right answer, the answer key gives the preferred past
answer and its topic. A rule ranks the same Hindsight candidates with the same lessons; we count:

    top1_best       the preferred answer is ranked first
    top3_best       the preferred answer is in the top 3 (what the drafter is shown)
    top1_trap       a trap (outdated, vague, lost) is ranked first
    top1_off_topic  the first answer is about a different topic
    top3_off_topic  off-topic answers in the top 3, summed over questions
    stress_off_topic  after one simulated rejection of whatever ranks first, an off-topic answer takes
                      first place although an on-topic one is available (the invariant: memory may
                      reorder relevant answers, never lift an irrelevant one to the top)

    .\\.venv\\Scripts\\python -m eval.v2_relevance.evaluate
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from sqlalchemy import select

from backend import workspaces
from backend.main import base_settings
from backend.v1.db import Answer, Database, PastProposal
from backend.v1.lessons import LessonHit, answer_signals
from backend.v1.ranking import Scored, order_candidates, rank_factors

HERE = Path(__file__).parent
CACHE = HERE / "recall_cache.json"


def topic(key: str | None) -> str:
    return (key or "?").split("_")[0]


@dataclass(frozen=True)
class Rule:
    name: str
    relevance: str  # "plain" or a ranking.RELEVANCE_POLICIES value
    margin: float = 0.0
    within: str = "rank"


RULES = [
    Rule("plain (V1, Hindsight order)", "plain"),
    Rule("V2 pre-fix: 1/rank x lessons", "rank"),
    *[Rule(f"V2 gated {m}, Hindsight order inside groups", "gated", m, "rank") for m in (0.005, 0.01, 0.02, 0.05)],
    *[Rule(f"V2 gated {m}, memory decides inside groups", "gated", m, "flat") for m in (0.005, 0.01, 0.02, 0.05)],
]


def load_workspace():
    settings = workspaces.apply_workspace(base_settings())
    db = Database(settings.db_path)
    with db.session() as session:
        answers = {a.code: a for a in session.scalars(select(Answer))}
        written = dict(session.execute(
            select(Answer.id, PastProposal.submitted_on).join(PastProposal, Answer.past_proposal_id == PastProposal.id)
        ).all())
    return settings, answers, written


def rank(question: dict, rule: Rule, answers, written, extra_lessons=(), now=None) -> list[dict]:
    candidates = [c for c in question["candidates"] if c["id"] in answers]
    if rule.relevance == "plain":
        return candidates
    codes = {c["id"] for c in candidates}
    hits = [LessonHit(h["text"], h["tags"], h["document_id"], h["type"], h["rank"]) for h in question["lesson_hits"]]
    signals = answer_signals([*hits, *extra_lessons], codes)
    scored = []
    for c in candidates:
        answer = answers[c["id"]]
        factors = rank_factors(
            answer, None, recall_rank=c["rank"], client=question["client"], industry=question["industry"],
            written_on=written.get(answer.id), lessons=signals.get(c["id"]), now=now,
        )
        scored.append(Scored(c, factors.score, factors.relevance, c["final"], c["rank"]))
    return order_candidates(scored, policy=rule.relevance, min_share=rule.margin, within=rule.within)


def evaluate(rule: Rule, questions, answers, written, now=None) -> dict:
    m = dict(top1_best=0, top3_best=0, top1_trap=0, top1_off_topic=0, top3_off_topic=0, stress_off_topic=0, stress_cases=0)
    for q in questions:
        order = rank(q, rule, answers, written, now=now)
        t = topic(q["preferred"])
        top = [c["key"] for c in order[:3]]
        m["top1_best"] += top[0] == q["preferred"]
        m["top3_best"] += q["preferred"] in top
        m["top1_trap"] += top[0] in q["traps"]
        m["top1_off_topic"] += topic(top[0]) != t
        m["top3_off_topic"] += sum(topic(k) != t for k in top)
        # Stress: one reviewer rejects whatever is first. Plain retrieval ignores lessons, so it can't move.
        first = order[0]
        rejection = LessonHit(f"Reviewer rejected a draft that cited {first['id']}.",
                              ["kind:review", "signal:negative", f"answer:{first['id']}", "action:rejected"],
                              f"sim-{first['id']}", "world", 1)
        on_topic_left = any(topic(c["key"]) == t for c in order[1:])
        if on_topic_left and rule.relevance != "plain":
            m["stress_cases"] += 1
            after = rank(q, rule, answers, written, extra_lessons=[rejection], now=now)
            m["stress_off_topic"] += topic(after[0]["key"]) != t
    return m


def main() -> None:
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    questions = [q for q in data["questions"] if not q["gap"]]
    _settings, answers, written = load_workspace()
    # Freshness is measured from the capture date, so reruns on another day give the same numbers.
    from datetime import UTC, datetime

    now = datetime.fromisoformat(data["captured_at"]).astimezone(UTC)
    results = {}
    print(f"{len(questions)} questions with a right answer (captured {data['captured_at']}, workspace {data['workspace']})\n")
    header = f"{'rule':52} top1  top3  trap1  off1  off3  stress(off1 after 1 rejection)"
    print(header)
    for rule in RULES:
        m = evaluate(rule, questions, answers, written, now=now)
        results[rule.name] = m
        stress = f"{m['stress_off_topic']}/{m['stress_cases']}" if m["stress_cases"] else "n/a"
        print(f"{rule.name:52} {m['top1_best']:>4}  {m['top3_best']:>4}  {m['top1_trap']:>5}  {m['top1_off_topic']:>4}  {m['top3_off_topic']:>4}  {stress}")
    (HERE / "results.json").write_text(json.dumps({"questions": len(questions), "captured_at": data["captured_at"],
                                                   "rules": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
