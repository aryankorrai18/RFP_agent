"""Diagnostic (free, no model call, product code unchanged): how big would the industry/client bonus have to be to change a
ranking, and what breaks when it is that big?

Part 1 is arithmetic on the product's own scoring rule. Part 2 re-runs the retrieval with the bonus raised to a power
(1 = as shipped) by wrapping `rank_factors` for the duration of the run; the product's source is not modified.

    python -m evaluation.context_diagnostic --seeds 10 --first-seed 1

Limit: in the generated worlds only the context questions have an answer whose industry matches the new client, so this shows
how large the bonus must be to fix them but cannot show what a large bonus would break (an off-topic answer from the client's
own industry outranking the right one). That needs a world with such a case, on new seeds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from rfp_assistant.api.v1 import ranking, retrieval

from . import retrieval_eval, stats

RESULTS = Path(__file__).resolve().parent / "results"
POWERS = (1.0, 2.0, 4.0, 8.0, 16.0)
MAX_SHIPPED_BONUS = 1.10 * 1.05  # same client and same industry


def arithmetic() -> list[dict]:
    """Two answers equal in quality and freshness: the one at search position r+1 has relevance 1/(r+1) against 1/r, so
    a bonus must exceed (r+1)/r to move it up one place."""
    return [{"position": r, "bonus_needed": (r + 1) / r, "shipped_max_bonus": MAX_SHIPPED_BONUS,
             "enough": MAX_SHIPPED_BONUS > (r + 1) / r} for r in range(1, 9)]


def set_power(power: float) -> None:
    """Raise the context factor to `power` inside every score (1 = the product as shipped)."""
    def wrapped(*args, **kwargs):  # noqa: ANN002, ANN003
        f = ranking.rank_factors(*args, **kwargs)
        if power == 1.0 or not f.context:
            return f
        context = f.context ** power
        return replace(f, context=round(context, 6), score=round(f.score / f.context * context, 8))

    retrieval.rank_factors = wrapped


async def run(seeds: list[int]) -> dict:
    rows: list[dict] = []
    try:
        for power in POWERS:
            set_power(power)
            for seed in seeds:
                rows += [r | {"power": power} for r in await retrieval_eval.evaluate_world(seed, systems=("outcome",))]
    finally:
        retrieval.rank_factors = ranking.rank_factors
    return {"arithmetic": arithmetic(), "powers": POWERS, "seeds": seeds, "rows": rows}


def summary(data: dict) -> list[str]:
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in data["rows"]:
        by[(r["power"], r["kind"])].append(r)
    lines = ["| Industry bonus (1.05 as shipped, raised to a power) | " + " | ".join(k for k in ("context", "exact", "stale", "reviewed", "lost_trap")) + " |", "|---|---|---|---|---|---|"]
    for p in data["powers"]:
        cells = [stats.interval(*stats.wilson(sum(r["hit1"] for r in by[(p, k)]), len(by[(p, k)]))) for k in ("context", "exact", "stale", "reviewed", "lost_trap")]
        lines.append(f"| 1.05^{p:g} = x{1.05 ** p:.2f} | " + " | ".join(cells) + " |")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evaluation.context_diagnostic")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=1, help="development worlds by default: this is a diagnostic, not a headline")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    data = asyncio.run(run(list(range(args.first_seed, args.first_seed + args.seeds))))
    out = RESULTS / "context_diagnostic.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    for a in data["arithmetic"]:
        print(f"position {a['position']} -> {a['position'] + 1}: bonus needed x{a['bonus_needed']:.2f}, shipped max x{a['shipped_max_bonus']:.3f}, enough: {a['enough']}")
    print("\n".join(summary(data)))


if __name__ == "__main__":
    main()
