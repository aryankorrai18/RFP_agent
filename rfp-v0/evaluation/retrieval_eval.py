"""Layer 1 and 3, free (no model call): does memory put the right approved answer in front of the model?

For every generated world the product's own retrieval runs on every question that has a right library answer, under each
mode, next to simple baselines. The right answer is known from the planted rules, so each result is marked by code.

  lexical    plain search: the memory's own order, no checks, no memory of outcomes (what a basic RAG does)
  newest     search, drop lost proposals, newest proposal first (a one-line heuristic)
  plain      the product with memory off for ranking (it still drops answers from lost proposals when drafting)
  outcome    the product ranking by review statistics, outcomes and freshness kept in its own database
  hindsight  the same plus the lessons bank (here a local stand-in)

    python -m evaluation.retrieval_eval --seeds 10 --first-seed 101 --out heldout.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import harness
from .world import LIBRARY_KINDS, REVIEW_PLAN, make_world

RESULTS = Path(__file__).resolve().parent / "results"
SYSTEMS = ("lexical", "newest", "plain", "outcome", "hindsight")
CURVE_STEPS = (0, 1, 2, 3, 4, 5)  # rounds of review history per answer pair (see world.REVIEW_PLAN)


def set_variant(name: str) -> None:
    """"default" is the product as shipped. "flat" is a diagnostic: inside a group of about equally relevant answers it
    drops the search position from the score, so memory, freshness and context alone decide (ranking.order_candidates
    already offers this, unused). Reported as an experiment, never as the product."""
    import functools

    from rfp_assistant.api.v1 import ranking, retrieval

    retrieval.order_candidates = functools.partial(ranking.order_candidates, within="flat") if name == "flat" else ranking.order_candidates


async def rank_all(loaded: harness.Loaded, question, systems=SYSTEMS) -> dict[str, list[str]]:  # noqa: ANN001
    out: dict[str, list[str]] = {}
    for system in systems:
        if system == "lexical":
            out[system] = await harness.raw_search(loaded, question)
        elif system == "newest":
            out[system] = await harness.newest_first(loaded, question)
        else:
            found = await harness.product_retrieval(loaded, question, system)
            out[system] = [p.id for p in found.past_answers]
    return out


async def evaluate_world(seed: int, history: int | None = None, systems=SYSTEMS) -> list[dict]:  # noqa: ANN001
    world = make_world(seed)
    rows: list[dict] = []
    with harness.temp_folder() as tmp:
        loaded = await harness.load_world(world, Path(tmp), history=history)
        try:
            for question in world.questions:
                if question.kind not in LIBRARY_KINDS:
                    continue
                for system, ranked in (await rank_all(loaded, question, systems)).items():
                    rows.append({"seed": seed, "question": question.id, "kind": question.kind, "system": system,
                                 "history": history} | harness.mark_retrieval(question, ranked, loaded.refs))
        finally:
            await loaded.close()
    return rows


async def run(seeds: list[int], curve: bool = True, say=print) -> dict:  # noqa: ANN001
    rows: list[dict] = []
    for seed in seeds:
        rows += await evaluate_world(seed)
        say(f"  world {seed}: {len(rows)} results so far")
    curve_rows: list[dict] = []
    if curve:
        for seed in seeds:
            for n in CURVE_STEPS:
                curve_rows += await evaluate_world(seed, history=n, systems=("plain", "outcome", "hindsight"))
            say(f"  curve world {seed} done")
    return {"design": {"seeds": seeds, "systems": SYSTEMS, "kinds": LIBRARY_KINDS, "curve_steps": CURVE_STEPS,
                       "review_plan": [list(step) for step in REVIEW_PLAN]}, "rows": rows, "curve_rows": curve_rows}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evaluation.retrieval_eval")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=101, help="101 and up are the held-out worlds, not used to design anything")
    parser.add_argument("--no-curve", action="store_true")
    parser.add_argument("--variant", choices=("default", "flat"), default="default")
    parser.add_argument("--out", default="heldout.json")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    set_variant(args.variant)
    data = asyncio.run(run(seeds, curve=not args.no_curve))
    out = RESULTS / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data), encoding="utf-8")
    print(f"wrote {len(data['rows'])} rows to {out}")


if __name__ == "__main__":
    main()
