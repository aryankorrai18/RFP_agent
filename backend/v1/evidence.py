"""Token-free evidence-support check for V2.

This conservative lexical checker is the default while provider tokens are scarce. The interface
and stored verdicts are provider-neutral so an entailment model can replace it for the final V4
protocol without changing drafts or the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "and", "that", "with", "from", "this", "have", "has", "our", "are", "for", "into", "your"}


@dataclass(frozen=True)
class EvidenceVerdict:
    status: str
    coverage: float
    reason: str


def _terms(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if len(w) >= 3 and w not in STOP}


def check_claim(claim: str, source_ids: list[str], sources: dict[str, str]) -> EvidenceVerdict:
    available = [sources[i] for i in source_ids if i in sources]
    if not available:
        return EvidenceVerdict("unverifiable", 0.0, "No live cited source text was available.")
    wanted = _terms(claim)
    if not wanted:
        return EvidenceVerdict("unverifiable", 0.0, "The claim contained no checkable terms.")
    evidence = _terms(" ".join(available))
    coverage = len(wanted & evidence) / len(wanted)
    if coverage >= 0.65:
        status = "supported"
    elif coverage >= 0.35:
        status = "partial"
    else:
        status = "unsupported"
    return EvidenceVerdict(status, round(coverage, 3), f"Lexical evidence coverage: {coverage:.0%}.")


def check_claims(claims: list[dict], sources: dict[str, str]) -> tuple[list[dict], list[str]]:
    checked: list[dict] = []
    statuses: list[str] = []
    for claim in claims:
        verdict = check_claim(claim.get("text", ""), claim.get("source_ids", []), sources)
        checked.append(claim | {
            "support_status": verdict.status,
            "support_coverage": verdict.coverage,
            "support_reason": verdict.reason,
        })
        statuses.append(verdict.status)
    return checked, statuses
