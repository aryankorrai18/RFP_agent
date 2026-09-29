"""Score a before/after comparison against the corpus answer key.

The answer key says, for each question, which past answer is the right one to reuse (from a won
proposal, current, specific) and which are traps (outdated, from a lost proposal, vague, another
client's details). This counts, per memory state, how often the draft cited each kind.

  python samples/corpus_v2/score_comparison.py <project_id> [--server http://127.0.0.1:8001]

Run the comparison first (the "Does memory change the draft?" card on the project page). The RFP
must be one of the corpus RFPs in answer_key.json. Only reads from the server; uses no model calls.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

KEY = Path(__file__).with_name("answer_key.json")


def get(server: str, path: str) -> dict:
    with urllib.request.urlopen(server + path, timeout=60) as response:
        return json.load(response)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def find_rfp(key: dict, questions: list[str]) -> dict:
    """The corpus RFP whose questions best match the project's."""
    def overlap(rfp: dict) -> float:
        wanted = [norm(q["question"]) for q in rfp["questions"]]
        return sum(max((SequenceMatcher(None, w, norm(q)).ratio() for q in questions), default=0) for w in wanted) / len(wanted)

    name, rfp = max(key["rfps"].items(), key=lambda item: overlap(item[1]))
    if overlap(rfp) < 0.6:
        raise SystemExit("This project doesn't look like one of the corpus RFPs in answer_key.json.")
    return rfp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id", type=int)
    parser.add_argument("--server", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    key = json.loads(KEY.read_text(encoding="utf-8"))
    comparison = get(args.server, f"/v1/projects/{args.project_id}/comparison")["comparison"]
    if not comparison or comparison["status"] != "completed":
        raise SystemExit("Run the comparison on the project page first, and wait for it to finish.")
    text_of = {a["id"]: norm(a["answer"]) for a in get(args.server, "/v1/library/answers?limit=200")["answers"]}
    rfp = find_rfp(key, [q["question"] for q in comparison["questions"]])

    tally = {arm: Counter() for arm in comparison["arms"]}
    for question in comparison["questions"]:
        entry = max(rfp["questions"], key=lambda k: SequenceMatcher(None, norm(k["question"]), norm(question["question"])).ratio())
        preferred = norm(entry["preferred"]["answer"]) if entry["preferred"] else None
        traps = {norm(t["answer"]): t["kind"] for t in entry["traps"]}
        for arm, draft in question["arms"].items():
            counts = tally[arm]
            counts["questions"] += 1
            cited = [text_of.get(code, "") for code in draft["cited_answers"]]
            if entry.get("gap"):
                counts["gap questions"] += 1
                counts["gap: no answer invented" if draft["status"] == "needs_sme" or not draft["answer"] else "gap: answered anyway"] += 1
                continue
            if preferred and preferred in cited:
                counts["cited the preferred answer"] += 1
            for text in cited:
                if text in traps:
                    counts[f"cited a trap ({traps[text]})"] += 1

    labels = {"plain": "Before (no lessons)", "hindsight": "After (lessons)", "none": "No memory", "outcome": "Local signals"}
    rows = sorted({k for c in tally.values() for k in c if k not in ("questions", "gap questions")})
    print(f"Project {args.project_id}: {tally[comparison['arms'][0]]['questions']} questions, model {comparison['model']}")
    print(f"{'':44}" + "".join(f"{labels.get(a, a):>22}" for a in comparison["arms"]))
    for row in rows:
        print(f"{row:44}" + "".join(f"{tally[a][row]:>22}" for a in comparison["arms"]))


if __name__ == "__main__":
    main()
