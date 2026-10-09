"""Layer 1, free and fast: does the retrieval and ranking recommend the right play and warn about the trap?

No model call is made anywhere in this file. For each of many generated worlds, every open test deal is advised by the
product's own retrieval and by three simple baselines, and each answer is marked against the planted rules."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import harness
from .world import SITUATIONS, make_world

RESULTS = Path(__file__).resolve().parent / "results"
SYSTEMS = ("popularity", "naive_rag", "neighbour_wins", "ours_similar", "ours_hindsight")
CURVE_SIZES = (10, 20, 40, 60)


async def evaluate_world(seed: int, n_train: int, n_test: int, size: int | None = None) -> list[dict]:
    world = make_world(seed, n_train, n_test)
    if size is not None:
        world = harness.truncate(world, size)
    rows: list[dict] = []
    with harness.temp_folder() as tmp:
        loaded = await harness.load_world(world, Path(tmp))
        try:
            rows += await _mark(loaded, world, seed)
        finally:
            await loaded.close()
    return rows


async def _mark(loaded: harness.Loaded, world, seed: int) -> list[dict]:  # noqa: ANN001
    rows: list[dict] = []
    pop = harness.popularity(world.train)
    for truth in world.test:
        kind = SITUATIONS[truth.situation].kind
        answers = {
            "popularity": (pop, []),
            "naive_rag": (harness.naive_rag(world.train, truth.situation), []),
            "neighbour_wins": (harness.neighbour_wins(world.train, truth.situation), []),
            "ours_similar": await harness.product_plays(loaded, truth, "similar"),
            "ours_hindsight": await harness.product_plays(loaded, truth, "hindsight"),
        }
        for system, (ranked, avoid) in answers.items():
            rows.append({"seed": seed, "n_train": len(world.train), "deal": truth.account, "situation": truth.situation,
                         "kind": kind, "system": system, **harness.score(truth.situation, ranked, avoid)})
    return rows


async def run(seeds: list[int], n_train: int = 60, n_test: int = 25, curve: bool = True, say=print) -> dict:  # noqa: ANN001
    main_rows: list[dict] = []
    for seed in seeds:
        main_rows += await evaluate_world(seed, n_train, n_test)
        say(f"seed {seed}: {len(main_rows)} marked answers so far")
    curve_rows: list[dict] = []
    if curve:
        for size in CURVE_SIZES:
            for seed in seeds:
                curve_rows += await evaluate_world(seed, n_train, n_test, size)
            say(f"learning curve: {size} closed deals done")
    return {"design": {"seeds": seeds, "n_train": n_train, "n_test": n_test, "systems": list(SYSTEMS), "curve_sizes": list(CURVE_SIZES),
                       "top": harness.TOP, "model_calls": 0},
            "rows": main_rows, "curve_rows": curve_rows}


def set_variant(variant: str) -> None:
    """"before" switches off the one ranking rule this evaluation led to (so the old behaviour can be re-measured on new worlds)."""
    from deal_intelligence.api.v1 import ranking

    ranking.LABEL_OVERRULED_MIN_USED = 10**9 if variant == "before" else 3


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evaluation.retrieval_eval")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--variant", choices=("before", "after"), default="after")
    parser.add_argument("--no-curve", action="store_true")
    parser.add_argument("--out", default="retrieval.json")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    set_variant(args.variant)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    result = asyncio.run(run(seeds, curve=not args.no_curve, say=lambda m: print(m, flush=True)))
    result["design"]["variant"] = args.variant
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {RESULTS / args.out}: {len(result['rows'])} marked answers, {len(result['curve_rows'])} for the learning curve", flush=True)


if __name__ == "__main__":
    main()
