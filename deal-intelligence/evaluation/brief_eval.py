"""Layer 2, costs model calls: do the written briefs recommend the right play, with a plain model and with memory?

For each generated world and each open test deal, the product writes a brief under four conditions with the SAME model,
prompt and temperature (0). They differ only in what the model is shown about past deals:

  none       nothing about the past (a plain model with the deal's own notes and the play catalogue)
  longctx    a summary of every closed deal pasted into the prompt (no retrieval, no ranking, no warnings)
  similar    the product's retrieval over the local database: similar deals, ranked plays, counted warnings, plays to avoid
  hindsight  the same plus lessons learned from outcomes (kept in the local free memory, so no Hindsight Cloud is used)

Each brief is marked by code against the planted rules (see world.py): does it recommend a good play, and does it recommend
the trap? No model judges another model. One model call per brief; signals are seeded, so nothing else is called.

    python -m evaluation.brief_eval --estimate          # no network: how many calls and roughly what they cost
    python -m evaluation.brief_eval --seeds 2           # the real run; resumes where it stopped
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from deal_intelligence.api.v1.briefs import generate_brief
from deal_intelligence.providers.base import LLMError, LLMResult, TokenUsage
from deal_intelligence.schemas import BriefStep, DealBriefResult

from . import harness
from .world import SITUATIONS, make_world

RESULTS = Path(__file__).resolve().parent / "results"
ARMS = ("none", "longctx", "similar", "hindsight")
# Worst-case list prices per million tokens, used only for the estimate; the real run reports the tokens actually used.
ASSUMED_USD_PER_M_IN, ASSUMED_USD_PER_M_OUT = 0.30, 2.50


class ScriptedLLM:
    """Stands in for the model in --estimate and in tests: it only records how big each prompt was and answers with the
    play the catalogue says addresses the objection (what a plain model would obviously pick). NEVER used for reported results."""

    model = "scripted"

    def __init__(self) -> None:
        self.sizes: list[int] = []

    async def structured(self, *, purpose, output_format, system, user, effort="medium", max_tokens=4000, temperature=None):  # noqa: ANN001, ANN201
        self.sizes.append(len(system) + len(user))
        import re

        plays = re.findall(r'<play code="(PLAY-\d+)"', user)
        ids = re.findall(r'<interaction id="(INT-\d+)"', user)
        step = [BriefStep(play_code=plays[0], rationale="Offered for this deal.", source_ids=[])] if plays else []
        return LLMResult(output=DealBriefResult(summary="Scripted.", summary_sources=ids[:1], next_steps=step),
                         model=self.model, usage=TokenUsage(input_tokens=len(system + user) // 4, output_tokens=450))


def mark_brief(situation: str, content: dict) -> dict:
    plays = [s["play_code"] for s in content.get("next_steps", [])]
    marks = harness.score(situation, plays, [a["play_code"] for a in content.get("avoid", [])])
    return {k: marks[k] for k in ("hit1", "hit3", "trap1", "trap3", "flagged", "false_alarm")} | {"plays": plays}


async def run_seed(seed: int, n_train: int, n_test: int, arms: tuple[str, ...], llm_provider, settings, out: Path,  # noqa: ANN001
                   done: set[tuple], concurrency: int, say=print) -> int:  # noqa: ANN001
    world = make_world(seed, n_train, n_test)
    spent = 0
    with harness.temp_folder() as tmp:
        loaded = await harness.load_world(world, Path(tmp), settings=settings, llm_provider=llm_provider)
        gate = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()

        async def one(truth, arm: str) -> None:  # noqa: ANN001
            nonlocal spent
            key = (seed, truth.account, arm)
            if key in done:
                return
            async with gate:
                started = time.perf_counter()
                brief = await generate_brief(loaded.ctx, loaded.ids[truth.account], arm, today=harness.TODAY)
                seconds = time.perf_counter() - started
            row = {"seed": seed, "deal": truth.account, "situation": truth.situation, "kind": SITUATIONS[truth.situation].kind, "arm": arm,
                   "status": brief.status, "seconds": round(seconds, 2), "input_tokens": brief.input_tokens or 0,
                   "output_tokens": brief.output_tokens or 0, "model": brief.model, "flags": [f["code"] for f in (brief.flags or [])]}
            if brief.status == "ready":
                row.update(mark_brief(truth.situation, brief.content))
            else:
                row["error"] = (brief.error or "")[:300]
            async with lock:
                with out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                spent += 1
                if spent % 10 == 0:
                    say(f"  seed {seed}: {spent} briefs written")

        try:
            await asyncio.gather(*(one(t, arm) for t in world.test for arm in arms))
        finally:
            await loaded.close()
    return spent


def load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(r["seed"], r["deal"], r["arm"]) for r in rows if r.get("status") == "ready"}


async def estimate(seeds: int, n_train: int, n_test: int, arms: tuple[str, ...]) -> dict:
    llm = ScriptedLLM()
    for seed in range(1, seeds + 1):
        world = make_world(seed, n_train, n_test)
        with harness.temp_folder() as tmp:
            loaded = await harness.load_world(world, Path(tmp), llm_provider=lambda _s: llm)
            try:
                for truth in world.test:
                    for arm in arms:
                        await generate_brief(loaded.ctx, loaded.ids[truth.account], arm, today=harness.TODAY)
            finally:
                await loaded.close()
    calls = len(llm.sizes)
    tokens_in = sum(llm.sizes) // 4
    tokens_out = calls * 450
    return {"calls": calls, "input_tokens": tokens_in, "output_tokens": tokens_out,
            "usd_upper_bound": round(tokens_in / 1e6 * ASSUMED_USD_PER_M_IN + tokens_out / 1e6 * ASSUMED_USD_PER_M_OUT, 3)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evaluation.brief_eval")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--first-seed", type=int, default=101, help="101 and up are the held-out worlds, not used to design anything")
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--model", help="pin the model (default: the one the app is set to use)")
    parser.add_argument("--effort", choices=("low", "medium", "high"), help="thinking effort for the briefs (default: the app's setting)")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--estimate", action="store_true", help="count the calls and size the prompts without calling a model")
    parser.add_argument("--out", default=str(RESULTS / "briefs.jsonl"))
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    if args.estimate:
        print(json.dumps(asyncio.run(estimate(args.seeds, args.n_train, args.n_test, arms)), indent=1))
        return
    from deal_intelligence.main import _llm_for, base_settings  # reads the app's own settings and keys; nothing is printed

    settings = base_settings()
    if args.model:
        settings = replace(settings, model=args.model)
    if args.effort:
        settings = replace(settings, brief_effort=args.effort)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    print(f"model {settings.provider}/{settings.model} (effort {settings.brief_effort}); {len(done)} briefs already done; {args.seeds} worlds x {args.n_test} deals x {len(arms)} arms")
    total = 0
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        try:
            total += asyncio.run(run_seed(seed, args.n_train, args.n_test, arms, _llm_for, settings, out, done, args.concurrency))
        except LLMError as exc:
            print("stopped:", exc.message)
            break
    print(f"wrote {total} briefs to {out}")


if __name__ == "__main__":
    main()
