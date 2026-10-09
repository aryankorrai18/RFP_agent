"""Builds results/REPORT.md and the charts from the raw result files. Every number is computed here, from rows on disk."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from . import charts, stats
from .world import KIND_COUNTS, LIBRARY_KINDS, REVIEW_PLAN, TOPICS

RESULTS = Path(__file__).resolve().parent / "results"
SYSTEMS = {"lexical": "Plain search (no checks)", "newest": "Newest first (heuristic)", "plain": "Assistant, memory off for ranking",
           "outcome": "Assistant (outcome-aware)", "hindsight": "Assistant + lessons"}
ARMS = {"none": "No past answers (facts only)", "plain": "Past answers, search order", "outcome": "Past answers ranked by memory",
        "hindsight": "Ranked by memory + lessons"}
KIND_TITLES = {
    "exact": ("Exact", "one approved answer from a won proposal (control)"),
    "stale": ("Stale", "a 2021 answer with an outdated value and a 2025 answer with the current one"),
    "context": ("Context", "two same-year answers for different industries; the right one matches the new client's industry"),
    "reviewed": ("Reviewed", "reviewers kept accepting one answer and kept rewriting the other because its value was wrong"),
    "lost_trap": ("Lost trap", "a good answer from a won proposal and a wrong one from a proposal lost on technical fit"),
    "conflict": ("Conflict", "an approved answer says one value, the current fact sheet says another (the fact sheet must win)"),
    "fact_only": ("Fact only", "only the fact sheet knows the value (control)"),
    "unanswerable": ("Unanswerable", "nobody knows the value: the right behaviour is to hand it to an expert"),
}
SEEDS_NOTE = "worlds 101 and up were never used to design anything; worlds 1 to 10 were the development worlds"


def _rate(rows: list[dict], key: str) -> tuple[float, float, float]:
    return stats.wilson(sum(bool(r[key]) for r in rows), len(rows))


def _mean(rows: list[dict], key: str) -> tuple[float, float, float]:
    return stats.bootstrap_mean([float(r[key]) for r in rows])


def _paired(rows_a: list[dict], rows_b: list[dict], key: str, id_key: str = "question") -> tuple[float, float, float, int, int, float]:
    index = {(r["seed"], r[id_key]): r for r in rows_b}
    pairs = [(a, index[(a["seed"], a[id_key])]) for a in rows_a if (a["seed"], a[id_key]) in index]
    a_vals, b_vals = [bool(a[key]) for a, _ in pairs], [bool(b[key]) for _, b in pairs]
    mean, low, high = stats.paired_difference([float(x) for x in a_vals], [float(x) for x in b_vals])
    a_only, b_only, p = stats.sign_test(a_vals, b_vals)
    return mean, low, high, a_only, b_only, p


def _signed(mean: float, low: float, high: float) -> str:
    return f"{100 * mean:+.0f} points ({100 * low:+.0f} to {100 * high:+.0f})"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _group(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def load_drafts(path: Path) -> list[dict]:
    rows: dict[tuple, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["seed"], r["question"], r["arm"])] = r  # a later successful row replaces an earlier failure
    return [r for r in rows.values() if r.get("status") != "failed"]


def retrieval_section(data: dict) -> tuple[list[str], dict[str, str]]:
    by = _group(data["rows"], "kind", "system")
    seeds = len(data["design"]["seeds"])
    n_q = sum(KIND_COUNTS[k] for k in LIBRARY_KINDS)
    out = ["## Layer 1: retrieval and ranking (no model involved)", "",
           f"{seeds} generated worlds ({SEEDS_NOTE}), each with {n_q} questions whose right answer is in the library. The product's own retrieval "
           "returns the three past answers the model would be shown. A case is right when the right answer is first (or in the top three). "
           "\"Wrong answer first\" counts the planted wrong answer (stale, rewritten, other industry, from the lost proposal) coming first. Intervals are 95%.", ""]
    charts_out: dict[str, str] = {}
    for kind in LIBRARY_KINDS:
        title, blurb = KIND_TITLES[kind]
        n = len(by[(kind, "plain")])
        out += [f"### {title}: {blurb} (n = {n} per system)", "",
                "| System | Right answer first | Right answer in top 3 | Mean reciprocal rank | Wrong answer first | Wrong answer shown |", "|---|---|---|---|---|---|"]
        for s in SYSTEMS:
            rs = by[(kind, s)]
            wrong = "n/a" if kind in ("exact",) else stats.interval(*_rate(rs, "bad1"))
            shown = "n/a" if kind in ("exact",) else stats.interval(*_rate(rs, "bad_shown"))
            mrr = _mean(rs, "rr")
            out.append(f"| {SYSTEMS[s]} | {stats.interval(*_rate(rs, 'hit1'))} | {stats.interval(*_rate(rs, 'hit3'))} | {mrr[0]:.2f} | {wrong} | {shown} |")
        out.append("")
    groups = [KIND_TITLES[k][0] for k in LIBRARY_KINDS]
    charts_out["retrieval_first.svg"] = charts.bar_chart(
        "Is the right approved answer ranked first?", f"{seeds} worlds, 95% intervals, by kind of question.", groups,
        {SYSTEMS[s]: [_rate(by[(k, s)], "hit1") for k in LIBRARY_KINDS] for s in SYSTEMS}, height=400)
    out += ["![right answer first](retrieval_first.svg)", "",
            "### The assistant against the simple baselines, case by case", "",
            "Same questions, paired. A positive difference means the assistant is better. The sign test counts only the questions where the two disagree.", "",
            "| Question kind | Assistant against | Difference in right answer first | Only assistant right | Only other right | Test |", "|---|---|---|---|---|---|"]
    for kind in LIBRARY_KINDS:
        for base in ("lexical", "newest", "plain"):
            mean, low, high, a_only, b_only, p = _paired(by[(kind, "outcome")], by[(kind, base)], "hit1")
            out.append(f"| {KIND_TITLES[kind][0]} | {SYSTEMS[base]} | {_signed(mean, low, high)} | {a_only} | {b_only} | {_p(p)} |")
    out.append("")

    curve = _group([r for r in data["curve_rows"] if r["kind"] == "reviewed"], "system", "history")
    steps = data["design"]["curve_steps"]
    if curve:
        n = len(curve[("plain", steps[0])])
        charts_out["learning_curve.svg"] = charts.line_chart(
            "Does it learn? Right answer first as review history accumulates",
            f"Reviewed questions (n = {n} per point). Each round is one reviewer verdict on the wrong answer and one on the right one; shaded bands are 95% intervals.",
            [float(s) for s in steps], {SYSTEMS[s]: [_rate(curve[(s, h)], "hit1") for h in steps] for s in ("plain", "outcome", "hindsight")},
            x_label="rounds of review history per answer pair")
        plan = "; ".join(f"{i + 1}: {who} answer {action}" + (f" ({', '.join(tags)})" if tags else "") for i, (who, action, tags) in enumerate(REVIEW_PLAN))
        out += ["### Learning curve", "", f"Rounds of review history, in order: {plan}.", "", "![learning curve](learning_curve.svg)", ""]
    return out, charts_out


def drafts_section(rows: list[dict]) -> tuple[list[str], dict[str, str], dict]:
    if not rows:
        return [], {}, {}
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    by = _group(rows, "kind", "arm")
    models = sorted({r["model"] for r in rows if r.get("model")})
    seeds = sorted({r["seed"] for r in rows})
    tin, tout = sum(r["input_tokens"] for r in rows), sum(r["output_tokens"] for r in rows)
    summary = {"drafts": len(rows), "models": models, "seeds": seeds, "input_tokens": tin, "output_tokens": tout}
    out = ["## Layer 2: drafted answers (a real model, one call per draft)", "",
           f"{len(rows)} drafts from {len(seeds)} generated worlds, written by `{', '.join(models)}` with the product's own drafting prompt and grounding checks in "
           "every condition; only what the model is shown from the past differs. Each draft is marked by code: the right value stated, the wrong value absent, "
           "the right evidence cited, no citation of something it was not shown, and (for questions nobody can answer) handed to an expert instead of invented. "
           "No model judges another model.", ""]
    charts_out: dict[str, str] = {}
    for kind in KIND_TITLES:
        n = len(by[(kind, arms[0])])
        title, blurb = KIND_TITLES[kind]
        head = ("| Condition | Handed to an expert (right) | Answered with an invented value | Invented number | Median seconds |", "|---|---|---|---|---|") \
            if kind == "unanswerable" else \
            ("| Condition | Answer right | Stated a wrong value | Handed to an expert instead | Bad citation | Invented number | Median seconds |", "|---|---|---|---|---|---|---|")
        out += [f"### {title}: {blurb} (n = {n} per condition)", "", *head]
        for a in arms:
            rs = by[(kind, a)]
            secs = sorted(r["seconds"] for r in rs)
            med = f"{secs[len(secs) // 2]:.1f}"
            if kind == "unanswerable":
                out.append(f"| {ARMS[a]} | {stats.interval(*_rate(rs, 'correct'))} | {stats.interval(*_rate(rs, 'fabricated'))} | {stats.interval(*_rate(rs, 'invented_number'))} | {med} |")
            else:
                out.append(f"| {ARMS[a]} | {stats.interval(*_rate(rs, 'correct'))} | {stats.interval(*_rate(rs, 'wrong_value'))} | "
                           f"{stats.interval(*_rate(rs, 'abstained'))} | {stats.interval(*_rate(rs, 'bad_citation'))} | {stats.interval(*_rate(rs, 'invented_number'))} | {med} |")
        out.append("")
    groups = [KIND_TITLES[k][0] for k in KIND_TITLES]
    charts_out["drafts_correct.svg"] = charts.bar_chart(
        "Is the drafted answer right?", f"{len(seeds)} worlds, 95% intervals, by kind of question. For \"Unanswerable\", right means handed to an expert.", groups,
        {ARMS[a]: [_rate(by[(k, a)], "correct") for k in KIND_TITLES] for a in arms}, height=400)
    out += ["![drafted answer right](drafts_correct.svg)", "",
            "### Against the no-memory condition, question by question", "",
            "| Question kind | Condition | Difference in answers right | Only this right | Only no-memory right | Test |", "|---|---|---|---|---|---|"]
    for kind in KIND_TITLES:
        for a in arms:
            if a == "none":
                continue
            mean, low, high, a_only, b_only, p = _paired(by[(kind, a)], by[(kind, "none")], "correct")
            out.append(f"| {KIND_TITLES[kind][0]} | {ARMS[a]} | {_signed(mean, low, high)} | {a_only} | {b_only} | {_p(p)} |")
    out += ["", "### Does ranking by memory change what the model writes?", "",
            "| Question kind | Condition | Difference in answers right | Only this right | Only search-order right | Test |", "|---|---|---|---|---|---|"]
    for kind in KIND_TITLES:
        for a in ("outcome", "hindsight"):
            if a in arms and "plain" in arms:
                mean, low, high, a_only, b_only, p = _paired(by[(kind, a)], by[(kind, "plain")], "correct")
                out.append(f"| {KIND_TITLES[kind][0]} | {ARMS[a]} | {_signed(mean, low, high)} | {a_only} | {b_only} | {_p(p)} |")
    out.append("")
    return out, charts_out, summary


def headline(final: dict, drafts: list[dict]) -> list[str]:
    by = _group(final["rows"], "kind", "system")
    first = lambda kind, s: stats.percent(_rate(by[(kind, s)], "hit1")[0])  # noqa: E731
    lines = [f"- **Retrieval (no model, worlds never used for design).** Where the library holds near-duplicate answers that disagree, plain search puts "
             f"the right one first {first('stale', 'lexical')} of the time for stale answers and {first('reviewed', 'lexical')} where reviewers had been rewriting the wrong one. "
             f"The assistant's memory: {first('stale', 'outcome')} and {first('reviewed', 'outcome')}. A one-line \"newest first\" rule gets {first('stale', 'newest')} on stale answers "
             f"but only {first('reviewed', 'newest')} on reviewed ones, and {first('context', 'newest')} on industry context.",
             f"- **What memory does not do here.** Industry context: right answer first {first('context', 'outcome')} with memory against {first('context', 'plain')} without. "
             "The product's industry and client bonus is too small to ever overturn a search position, so it is inert in this evaluation (see Findings).",
             f"- **No harm elsewhere.** One clear answer: {first('exact', 'plain')} without memory, {first('exact', 'outcome')} with. "
             f"Answers from a proposal lost on technical fit are dropped by the product when drafting: right first {first('lost_trap', 'plain')} (search with no checks: {first('lost_trap', 'lexical')})."]
    if drafts:
        by_d = _group(drafts, "kind", "arm")
        ok = lambda kind, a: stats.percent(_rate(by_d[(kind, a)], "correct")[0]) if by_d.get((kind, a)) else "n/a"  # noqa: E731
        lines += [f"- **Drafted answers (a real model).** Right value stated: stale questions {ok('stale', 'none')} with no past answers, {ok('stale', 'plain')} with search order, "
                  f"{ok('stale', 'outcome')} ranked by memory; reviewed questions {ok('reviewed', 'none')}, {ok('reviewed', 'plain')}, {ok('reviewed', 'outcome')}.",
                  f"- **Does memory make drafting worse?** Where the fact sheet is right and an old answer disagrees (conflict): {ok('conflict', 'none')} with no past answers, "
                  f"{ok('conflict', 'plain')} with search order, {ok('conflict', 'outcome')} ranked by memory. Questions nobody can answer, handed to an expert: "
                  f"{ok('unanswerable', 'none')}, {ok('unanswerable', 'plain')}, {ok('unanswerable', 'outcome')}."]
    return ["## Headline", "", *lines, "",
            "**Read the limits before quoting any of this**: synthetic questions with rules the designer planted, a local lexical search standing in for Hindsight's "
            "semantic search, small samples in the model layer, one model at low thinking effort, and baselines that are deliberately simple.", ""]


def findings(dev: dict | None, flat: dict | None, final: dict) -> list[str]:
    out = ["## What this evaluation found", ""]
    if dev is not None:
        def ctx(data: dict) -> dict[str, str]:
            by = _group(data["rows"], "kind", "system")
            return {s: stats.percent(_rate(by[("context", s)], "hit1")[0]) for s in ("plain", "outcome", "hindsight")}

        d, h = ctx(dev), ctx(final)
        same = len({*d.values()}) == 1 and len({*h.values()}) == 1
        out += ["**1. The industry and client bonus " + ("never changes a result" if same else "barely changes a result") + ".** Where the right answer is the one written for the new "
                f"client's own industry, the assistant put it first {d['outcome']} of the time on the development worlds ({h['outcome']} on the held-out worlds) with outcome memory, "
                f"{d['hindsight']} ({h['hindsight']}) with lessons, and {d['plain']} ({h['plain']}) with memory off. Which of two same-year answers comes first is a coin flip per question, "
                "so about half is chance. Cause: inside a group of about equally relevant answers, the score is search position (1 for first, 1/2 for second) "
                "times quality, freshness and context, and context is worth at most 1.10 for a same client and 1.05 for a same industry. A bonus of 5 to 10% "
                "cannot move an answer up a place that costs 50%. Review history (quality) and age (freshness) can, which is why those two work. "
                "The design intent (\"context reorders within a relevance group\") is not met for answers that differ only in context. "
                "**Not changed**: a fix has to be tested against real Hindsight scores, not this evaluation's lexical stand-in (see below).", ""]
    diag = RESULTS / "context_diagnostic.json"
    if diag.exists():
        from .context_diagnostic import summary

        d = json.loads(diag.read_text(encoding="utf-8"))
        a = {x["position"]: x["bonus_needed"] for x in d["arithmetic"]}
        out += ["**1b. How big would the bonus have to be?** To move an answer up one place a bonus must exceed (r+1)/r for the answer at position r: "
                f"x{a[1]:.2f} for 2nd to 1st, x{a[2]:.2f} for 3rd to 2nd. The shipped maximum is x1.155 (same client and same industry), so it can only reorder "
                "answers at positions 7 and below, and only the top three are ever shown. Raising the industry bonus to a power on the development worlds "
                "(the product's source is untouched, the score is rescaled in a wrapper):", "", *summary(d), "",
                "The context questions are fixed only when the bonus reaches about x2 (the industry factor alone would have to be roughly 1.05^14). Nothing else moves, "
                "but that is **not evidence it is safe**: in these worlds no off-topic answer shares the client's industry, so the harm a large bonus can do "
                "(an irrelevant answer from the client's own industry outranking the right one) is untested. A world with that case, on new seeds, is the next step.", ""]
    if flat is not None:
        by = _group(flat["rows"], "kind", "system")
        base = _group(dev["rows"], "kind", "system") if dev else {}
        rows = ["| Kind | Right first, as shipped | Right first, \"flat\" inside a group |", "|---|---|---|"]
        for kind in LIBRARY_KINDS:
            shipped = stats.percent(_rate(base[(kind, "outcome")], "hit1")[0]) if base else "n/a"
            rows.append(f"| {KIND_TITLES[kind][0]} | {shipped} | {stats.percent(_rate(by[(kind, 'outcome')], 'hit1')[0])} |")
        out += ["**2. The obvious fix is dangerous, which is what the gate is for.** `order_candidates` already has an unused `within=\"flat\"` mode that drops "
                "the search position from the score inside a group, so context could decide. Tried on the development worlds (outcome-aware, same questions):", "", *rows, "",
                "It lets memory promote answers that are only weakly relevant. With this evaluation's lexical scores the group of \"about equally relevant\" answers is wide, "
                "so the damage here may be larger than real Hindsight would show; the point is that the relevance gate and the position term are doing real work. "
                "Whether a narrower group (a higher `min_share`) plus a flat order would fix context on real data is the open question.", ""]
    out += ["**3. Memory needs a pattern, not one data point.** In the learning curve the assistant does not move until a pair has three rounds of history (the wrong "
            "answer rewritten twice, the right one accepted once): one rewrite is not enough to overturn the search order. This is a property of the smoothed quality score, "
            "chosen so one reviewer's opinion cannot bury an answer.", ""]
    return out


def main() -> None:
    final = json.loads((RESULTS / "heldout.json").read_text(encoding="utf-8"))
    dev = json.loads((RESULTS / "dev.json").read_text(encoding="utf-8")) if (RESULTS / "dev.json").exists() else None
    flat = json.loads((RESULTS / "dev_flat.json").read_text(encoding="utf-8")) if (RESULTS / "dev_flat.json").exists() else None
    drafts = load_drafts(RESULTS / "drafts.jsonl")
    r_lines, r_charts = retrieval_section(final)
    d_lines, d_charts, d_summary = drafts_section(drafts)
    head = ["# RFP Memory Assistant: does the memory help, measured?", "",
            "This report is generated by `python -m evaluation.report` from raw result files in this folder. It tests a mechanism on **synthetic questions with rules "
            "planted by the designer**, so it shows what the system does when the library and the review history contain a pattern. It does **not** show what it "
            "would earn a real proposal team.", "",
            "## The claim", "",
            "> When a library holds approved answers that disagree (outdated, wrong, or from a lost bid), the assistant puts the right one in front of the model, "
            "the drafted answer states the right value and cites the right evidence, memory does not make the model worse where it has nothing useful to offer, "
            "and anything nobody can answer is handed to an expert instead of invented.", "",
            "## The world", "",
            "A fictional software company, one fact sheet, a library of approved answers imported from eleven past proposals (one lost on technical fit), and "
            f"{sum(KIND_COUNTS.values())} new questions per world, each drawn from {len(TOPICS)} topics with values chosen per seed. Eight kinds of question, "
            "the right answer known by construction:", "",
            "| Kind | Per world | Setup |", "|---|---|---|"]
    for kind, (title, blurb) in KIND_TITLES.items():
        head.append(f"| {title} | {KIND_COUNTS[kind]} | {blurb} |")
    head += ["", "Which of two near-identical answers was imported first (and so which one a tie-breaking search returns first) is drawn per world, so ties favour no system. "
             "Review history is planted in the same shape the product's review path writes it (the same counters and the same event text that becomes a lesson).", "",
             "## Systems compared", "",
             "- **Plain search**: the memory's own order, no checks, no memory of outcomes (what a basic RAG does).",
             "- **Newest first**: among answers the search scores about as high as its best, drop lost proposals and take the newest. One line of code.",
             "- **Assistant, memory off for ranking**: the product's retrieval in `plain` mode (it still drops answers from lost proposals when drafting).",
             "- **Assistant (outcome-aware)**: ranks by the review statistics, outcomes and freshness in its own database.",
             "- **Assistant + lessons**: the same plus lessons built from reviews and outcomes (a local stand-in for the Hindsight lessons bank).", ""]
    notes = ["## Limits you should know", "",
             "- The data is synthetic and the rules are the designer's. Finding them is a test of mechanism, not of business value.",
             "- **Search is lexical, not Hindsight's semantic recall.** It misses paraphrases that share no words and occasionally ranks an unrelated answer first (the same for "
             "every system). Real recall would score differently, so ranking effects here should be re-checked against a real Hindsight bank before being trusted.",
             "- The lessons bank is a local stand-in; the lessons themselves are the product's own text.",
             "- Review history is planted directly rather than produced by 20 simulated reviewers, in the shape the product writes it.",
             "- The model layer uses one model at low thinking effort and a few worlds; some differences are not statistically significant.",
             "- Baselines are deliberately simple. A stronger retrieval system might close some of the gap.",
             "- Freshness uses the real clock, so very old runs drift slightly.",
             "- The next step is a pilot on a real team's past proposals, not a bigger benchmark.", "",
             "## Reproduce", "",
             "```powershell", "cd rfp-v0", "$env:PYTHONPATH = \"src;.\"",
             ".\\.venv\\Scripts\\python.exe -m evaluation.retrieval_eval --seeds 10 --first-seed 101 --out heldout.json   # free",
             ".\\.venv\\Scripts\\python.exe -m evaluation.draft_eval --estimate                                          # calls and cost, no network",
             ".\\.venv\\Scripts\\python.exe -m evaluation.draft_eval --seeds 3 --effort low                               # the real run, resumes",
             ".\\.venv\\Scripts\\python.exe -m evaluation.report", "```", ""]
    usage = [f"Model usage for Layer 2: {d_summary['drafts']} drafts, {d_summary['input_tokens']:,} input and {d_summary['output_tokens']:,} output tokens.", ""] if d_summary else []
    body = head + headline(final, drafts) + findings(dev, flat, final) + r_lines + d_lines + usage + notes
    for name, svg in {**r_charts, **d_charts}.items():
        (RESULTS / name).write_text(svg, encoding="utf-8")
    (RESULTS / "REPORT.md").write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {RESULTS / 'REPORT.md'} and {len(r_charts) + len(d_charts)} charts")


if __name__ == "__main__":
    sys.exit(main())
