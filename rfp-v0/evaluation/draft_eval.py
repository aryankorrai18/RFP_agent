"""Layer 2, costs model calls: do the drafted answers come out right, grounded and honest with memory?

For each generated world and each question the product drafts an answer under four conditions with the SAME model and
prompts. They differ only in what the model is shown from the past:

  none       no past answers (only the fact sheet), the plain-model baseline
  plain      the top past answers by search order
  outcome    the top past answers ranked by review statistics, outcomes and freshness
  hindsight  the same plus the lessons bank (a local stand-in here, so no Hindsight Cloud is used)

The product's own drafting runs (`draft_requirement`: retrieval, the drafting prompt, the grounding checks, the evidence
check, the SME handoff). Each drafted answer is then marked by code against the planted rules (harness.mark_draft): does
it state the right value and not the wrong one, does it cite the right evidence, does it invent a number, does it hand an
unanswerable question to an expert. One model call per draft. No model judges another model.

    python -m evaluation.draft_eval --estimate               # no network: calls and rough cost
    python -m evaluation.draft_eval --seeds 3 --effort low   # the real run; resumes where it stopped
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from rfp_assistant.api.v1.db import Answer, DraftRow
from rfp_assistant.api.v1.projects import draft_requirement
from rfp_assistant.providers.base import LLMError, LLMResult, TokenUsage
from rfp_assistant.schemas import DraftClaimOut, DraftResult

from . import harness
from .world import make_world

RESULTS = Path(__file__).resolve().parent / "results"
ARMS = ("none", "plain", "outcome", "hindsight")
# Worst-case list prices per million tokens, used only for the estimate; the real run reports the tokens actually used.
ASSUMED_USD_PER_M_IN, ASSUMED_USD_PER_M_OUT = 0.30, 2.50


class RecordingLLM:
    """Wraps the real model for `draft_answer` and records the size of each call. Nothing else is ever called."""

    def __init__(self, inner) -> None:  # noqa: ANN001
        self.inner = inner
        self.model = getattr(inner, "model", "scripted")
        self.usage: dict[str, tuple[int, int, float]] = {}  # question text -> (tokens in, tokens out, seconds)

    async def draft_answer(self, company, facts, requirement, past_answers=None, instructions=None, *, temperature=None):  # noqa: ANN001, ANN201
        started = time.perf_counter()
        result = await self.inner.draft_answer(company, facts, requirement, past_answers, instructions, temperature=temperature)
        usage = result.usage
        self.usage[requirement.question] = (usage.input_tokens or 0, usage.output_tokens or 0, time.perf_counter() - started)
        return result


class ScriptedLLM:
    """Stands in for the model in --estimate and in tests: it answers from the first past answer it is shown, or hands over to
    an expert. NEVER used for reported results."""

    model = "scripted"

    def __init__(self) -> None:
        self.sizes: list[int] = []

    async def draft_answer(self, company, facts, requirement, past_answers=None, instructions=None, *, temperature=None):  # noqa: ANN001, ANN201
        from rfp_assistant.providers import prompts

        self.sizes.append(len(str(prompts.drafting_system(company, facts))) + len(str(prompts.drafting_user_message(requirement, past_answers, instructions))))
        if past_answers:
            top = past_answers[0]
            out = DraftResult(answer=top.answer, claims=[DraftClaimOut(text=top.answer, source_ids=[top.id])],
                              unsupported_claims=[], needs_sme=False, sme_question=None)
        else:
            out = DraftResult(answer="", claims=[], unsupported_claims=[], needs_sme=True, sme_question="Who can answer this?")
        return LLMResult(output=out, model=self.model, usage=TokenUsage(input_tokens=self.sizes[-1] // 4, output_tokens=300))


def _evidence(loaded: harness.Loaded, question, row: DraftRow) -> str:  # noqa: ANN001
    """The text the model was shown: the live facts, the past answers retrieved for this draft, and the question."""
    shown = {r["id"] for r in (row.retrieved or [])}
    with loaded.ctx.db.session() as session:
        past = [session.get(Answer, int(code.split("-")[1])).answer for code in shown if code.startswith("ANS-")]
    return " ".join([f.statement for f in loaded.facts] + past + [question.text])


def quota_gone(error: str | None) -> bool:
    text = (error or "").lower()
    return "quota used up" in text or "credit balance" in text or "authentication" in text


async def run_seed(seed: int, arms: tuple[str, ...], provider, base, out: Path, done: set[tuple], concurrency: int,  # noqa: ANN001
                   say=print, only: set[str] | None = None) -> int:  # noqa: ANN001
    """Draft every question (or only the ids in `only`) of one world under each arm. Stops, writing nothing for the failed
    call, the moment the provider says its quota is used up, so a run cannot burn through a batch of failures."""
    world = make_world(seed)
    spent = 0
    stopped: list[str] = []
    with harness.temp_folder() as tmp:
        recorder = RecordingLLM(provider)
        loaded = await harness.load_world(world, Path(tmp), base=base, llm_provider=lambda _s: recorder)
        gate = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        company = world.company

        async def one(question, arm: str) -> None:  # noqa: ANN001
            nonlocal spent
            if (seed, question.id, arm) in done or stopped or (only is not None and question.id not in only):
                return
            requirement = loaded.requirements[question.id]
            project = loaded.projects[question.project]
            async with gate:
                row = await draft_requirement(loaded.ctx, project, requirement, company, loaded.facts)
            if row.status == "failed" and quota_gone(row.error):
                stopped.append(row.error)
                return
            usage = recorder.usage.get(question.text, (0, 0, 0.0))
            line = {"seed": seed, "question": question.id, "kind": question.kind, "arm": arm, "status": row.status,
                    "input_tokens": usage[0], "output_tokens": usage[1], "seconds": round(usage[2], 2), "model": row.model,
                    # the full configuration of this draft, so no result can be read against the wrong setup
                    "effort": loaded.settings.draft_effort, "prompt_version": row.prompt_version,
                    "retrieval_mode": loaded.settings.retrieval_mode, "relevance": loaded.settings.retrieval_relevance,
                    "flags": list(row.flags or []), "error": (row.error or "")[:300] or None}
            if row.status != "failed":
                gold = loaded.refs.get(question.gold_ref) if question.gold_ref else None
                draft = {"status": row.status, "answer": row.answer, "sources": row.sources, "flags": row.flags, "word_count": row.word_count}
                line |= harness.mark_draft(question, draft, _evidence(loaded, question, row), gold)
                line["shown"] = [r["id"] for r in (row.retrieved or [])]
                line["answer"] = row.answer
                line["sme_question"] = row.sme_question
                line["claims"] = row.claims
            async with lock:
                with out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(line) + "\n")
                spent += 1
                if spent % 10 == 0:
                    say(f"  world {seed}: {spent} drafts written")

        try:
            for arm in arms:  # one arm at a time: the product reads its retrieval mode from the settings
                loaded.use(retrieval_mode="none" if arm == "none" else arm)
                await asyncio.gather(*(one(q, arm) for q in world.questions))
        finally:
            await loaded.close()
    if stopped:
        raise LLMError("api_error", f"stopped after {spent} drafts: {stopped[0][:160]}")
    return spent


async def canary(provider, base, seed: int, say=print) -> bool:  # noqa: ANN001
    """Three drafts the model must get right (one clear approved answer, plain search): if it will not use an obviously
    relevant answer, the full run would only measure that, so stop before spending it."""
    import tempfile

    world = make_world(seed)
    ids = {q.id for q in world.questions if q.kind == "exact"}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out = Path(tmp) / "canary.jsonl"
        await run_seed(seed, ("plain",), provider, base, out, set(), 3, say=lambda _m: None, only=ids)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    good = sum(bool(r.get("correct")) for r in rows)
    first = rows[0] if rows else {}
    say(f"canary ({first.get('model')}, effort {first.get('effort')}, prompt {first.get('prompt_version')}, retrieval {first.get('retrieval_mode')}): "
        f"{good} of {len(rows)} clearly answerable questions drafted correctly")
    for r in rows:
        if not r.get("correct"):  # say why, so a failed canary can be diagnosed from one cheap run
            say(f"  {r['question']}: status {r['status']}, shown {r.get('shown')}, flags {r.get('flags')}, "
                f"tokens {r['input_tokens']}/{r['output_tokens']}, model's question for an expert: {(r.get('sme_question') or '')[:160]!r}")
    return len(rows) > 0 and good >= max(1, len(rows) - 1)


def load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(r["seed"], r["question"], r["arm"]) for r in rows if r.get("status") != "failed"}


async def estimate(seeds: int, arms: tuple[str, ...], first_seed: int = 101) -> dict:
    llm = ScriptedLLM()
    for seed in range(first_seed, first_seed + seeds):
        world = make_world(seed)
        with harness.temp_folder() as tmp:
            loaded = await harness.load_world(world, Path(tmp), llm_provider=lambda _s: llm)
            try:
                for arm in arms:
                    loaded.use(retrieval_mode="none" if arm == "none" else arm)
                    for question in world.questions:
                        await draft_requirement(loaded.ctx, loaded.projects[question.project], loaded.requirements[question.id],
                                                world.company, loaded.facts)
            finally:
                await loaded.close()
    calls = len(llm.sizes)
    tokens_in = sum(llm.sizes) // 4
    tokens_out = calls * 350
    return {"calls": calls, "input_tokens": tokens_in, "output_tokens": tokens_out,
            "usd_upper_bound": round(tokens_in / 1e6 * ASSUMED_USD_PER_M_IN + tokens_out / 1e6 * ASSUMED_USD_PER_M_OUT, 3)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evaluation.draft_eval")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=101, help="101 and up are the held-out worlds, not used to design anything")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--model", help="pin the model (default: the one the app is set to use)")
    parser.add_argument("--effort", choices=("low", "medium", "high"), help="thinking effort for drafting (default: the app's setting)")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--pick", help="a small run: only the first question of each of these kinds per world, e.g. stale,reviewed,conflict")
    parser.add_argument("--no-canary", action="store_true", help="skip the 3-call check that the model uses clearly relevant past answers")
    parser.add_argument("--estimate", action="store_true", help="count the calls and size the prompts without calling a model")
    parser.add_argument("--out", default=str(RESULTS / "drafts.jsonl"))
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    if args.estimate:
        print(json.dumps(asyncio.run(estimate(args.seeds, arms, args.first_seed)), indent=1))
        return
    from rfp_assistant.main import _llm_for, base_settings  # reads the app's own settings and keys; nothing is printed

    settings = base_settings()
    if args.model:
        settings = replace(settings, model=args.model)
    if args.effort:
        settings = replace(settings, draft_effort=args.effort)
    provider = _llm_for(settings)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    print(f"model {settings.provider}/{settings.model} (effort {settings.draft_effort}); {len(done)} drafts already done; "
          f"{args.seeds} worlds x {len(args.pick.split(',')) if args.pick else 24} questions x {len(arms)} arms")
    try:
        passed = args.no_canary or asyncio.run(canary(provider, settings, args.first_seed))
    except LLMError as exc:
        print("stopped:", exc.message)
        return
    if not passed:
        print("stopped: the model did not use clearly relevant past answers, so the run would not be a fair comparison. "
              "Check the model, quota and thinking effort first (see evaluation/results/drafts_run1_INVALID*).")
        return
    total = 0
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        try:
            only = None
            if args.pick:
                kinds = [k.strip() for k in args.pick.split(",") if k.strip()]
                questions = make_world(seed).questions
                only = {next(q.id for q in questions if q.kind == k) for k in kinds}
            total += asyncio.run(run_seed(seed, arms, provider, settings, out, done, args.concurrency, only=only))
        except LLMError as exc:
            print("stopped:", exc.message)
            break
    print(f"wrote {total} drafts to {out}")


if __name__ == "__main__":
    main()
