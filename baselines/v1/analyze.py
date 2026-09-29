"""Score V1 baseline runs against the acceptance criteria (see baselines/v1/BASELINE.md).

    python baselines/v1/analyze.py [run-*.json ...]      # default: every run in this folder

Expected matches are keyed by the library answer's question text, not by ANS code, so the check
survives a library rebuilt in a different order. The app must be running (it looks the codes up).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent

# RFP reference -> words that identify the past answer that should be retrieved for it.
EXPECTED = {
    "2.1": "about your organisation",
    "3.1": "single sign-on standards",
    "3.2": "created and removed automatically",
    "3.3": "protect customer data cryptographically",
    "3.5": "penetration testing",
    "4.1": "within the European Union",
    "4.2": "backup and recovery",
    "4.3": "at contract termination",
    "5.1": "availability do you guarantee",
    "5.2": "reach support",
    "5.3": "onboarding methodology",
    "6.1": "integration options",
    "8.2": "audit logs kept",
    "8.3": "multi-factor authentication required for administrators",
}
TRUE_GAPS = ("2.2", "6.2", "7.1")  # no fact and no past answer: must stay Needs SME
V0_NEEDS_SME = ("2.2", "3.2", "5.1", "5.3", "6.2", "7.1", "8.3")  # baselines/v0, prompt v0.3
# Outdated past answers that conflict with the fact sheet: (reference, must contain, must not contain)
CONFLICTS = (("2.1", "240", "200"), ("3.3", "TLS 1.2", "TLS 1.1"))


def library_codes() -> dict[str, str]:
    answers = httpx.get("http://127.0.0.1:8001/v1/library/answers", params={"limit": 200}).json()["answers"]
    codes = {}
    for ref, words in EXPECTED.items():
        matches = [a["id"] for a in answers if words.lower() in a["question"].lower() and a["source"] == "library"]
        codes[ref] = matches[0] if matches else None
    return codes


def score(run: dict, codes: dict[str, str]) -> dict:
    rows = {r["reference"]: r for r in run["requirements"]}
    retrieved_ok = [ref for ref, code in codes.items() if ref in rows and code in [x["id"] for x in rows[ref]["retrieved"]]]
    first_ok = [ref for ref, code in codes.items() if ref in rows and rows[ref]["retrieved"][:1] and rows[ref]["retrieved"][0]["id"] == code]
    cites = [ref for ref, row in rows.items() if any(s.startswith("ANS-") for s in row["sources"])]
    cites_expected = [ref for ref, code in codes.items() if ref in rows and code in rows[ref]["sources"]]
    gaps_ok = [ref for ref in TRUE_GAPS if ref in rows and rows[ref]["category"] == "needs_sme"]
    recovered = [ref for ref in V0_NEEDS_SME if ref not in TRUE_GAPS and ref in rows and rows[ref]["category"] in ("grounded", "check")]
    conflicts = []
    for ref, good, bad in CONFLICTS:
        answer = (rows.get(ref) or {}).get("answer") or ""
        conflicts.append((ref, good in answer and bad not in answer, answer))
    return {
        "missing_refs": sorted((set(EXPECTED) | set(TRUE_GAPS)) - set(rows)),
        "retrieved_in_top3": f"{len(retrieved_ok)}/{len(codes)}",
        "retrieved_first": f"{len(first_ok)}/{len(codes)}",
        "not_retrieved": sorted(set(codes) - set(retrieved_ok)),
        "drafts_citing_ANS": len(cites),
        "cite_the_expected_ANS": f"{len(cites_expected)}/{len(codes)}",
        "invalid_citations": [ref for ref, row in rows.items() if row["invalid_citations"]],
        "true_gaps_still_sme": f"{len(gaps_ok)}/{len(TRUE_GAPS)}",
        "gap_violations": [ref for ref in TRUE_GAPS if ref not in gaps_ok],
        "v0_sme_now_answered": recovered,
        "conflicts_fact_sheet_wins": [(ref, ok) for ref, ok, _ in conflicts],
        "conflict_answers": {ref: answer for ref, ok, answer in conflicts if not ok},
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(HERE.glob("run-*.json"))
    codes = library_codes()
    print("expected past answers:", json.dumps(codes))
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n== {path.name}  {json.dumps(run['summary'])}")
        for key, value in score(run, codes).items():
            print(f"   {key}: {value}")
        print("   per question: " + "  ".join(f"{r['reference']}={r['category'][:5]}{'+A' if any(s.startswith('ANS-') for s in r['sources']) else ''}"
                                            for r in run["requirements"]))


if __name__ == "__main__":
    main()
