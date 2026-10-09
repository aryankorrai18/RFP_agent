"""Builds results/REPORT.md and the charts from the raw result files. Every number is computed here, from rows on disk."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from . import charts, stats
from .world import OBVIOUS_SHARE, P_WIN, SITUATIONS

RESULTS = Path(__file__).resolve().parent / "results"
LABELS = {"popularity": "Popularity (situation-blind)", "naive_rag": "Plain retrieval (outcome-blind)",
          "neighbour_wins": "Similar winners only", "ours_similar": "Deal Intelligence", "ours_hindsight": "Deal Intelligence + lessons"}
ARM_LABELS = {"none": "Plain model (no history)", "longctx": "All deals pasted in", "similar": "Deal Intelligence", "hindsight": "Deal Intelligence + lessons"}
KINDS = (("counterintuitive", "Where history contradicts the obvious play"), ("control", "Where the obvious play is right (control)"))


def _rate(rows: list[dict], key: str) -> tuple[float, float, float]:
    return stats.wilson(sum(bool(r[key]) for r in rows), len(rows))


def _paired(rows_a: list[dict], rows_b: list[dict], key: str) -> tuple[float, float, float, int, int, float]:
    index = {(r["seed"], r["deal"]): r for r in rows_b}
    pairs = [(a, index[(a["seed"], a["deal"])]) for a in rows_a if (a["seed"], a["deal"]) in index]
    a_vals = [bool(a[key]) for a, _ in pairs]
    b_vals = [bool(b[key]) for _, b in pairs]
    mean, low, high = stats.paired_difference([float(x) for x in a_vals], [float(x) for x in b_vals])
    a_only, b_only, p = stats.sign_test(a_vals, b_vals)
    return mean, low, high, a_only, b_only, p


def _signed(mean: float, low: float, high: float) -> str:
    return f"{100 * mean:+.0f} points ({100 * low:+.0f} to {100 * high:+.0f})"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def fix_section(before: dict, after: dict, label: str, held_out: bool) -> list[str]:
    """The same worlds and the same deals, measured with the old ranking rule and with the new one."""
    idx = lambda d: {(r["seed"], r["deal"], r["system"]): r for r in d["rows"]}  # noqa: E731
    b, a = idx(before), idx(after)
    out = [f"| Situation type | System | Measure | Before | After | Change | Deals only the new rule gets right / only the old | Test |", "|---|---|---|---|---|---|---|---|"]
    for kind, _title in KINDS:
        for system in ("ours_similar", "ours_hindsight"):
            keys = [k for k in a if k[2] == system and k in b and a[k]["kind"] == kind]
            for key, name, better in (("hit1", "good play first", True), ("hit3", "good play in top 3", True), ("trap3", "trap in top 3", False)):
                va = [bool(a[k][key]) for k in keys]
                vb = [bool(b[k][key]) for k in keys]
                ra, rb = sum(va) / len(va), sum(vb) / len(vb)
                diff = [float(x) - float(y) for x, y in zip(va, vb, strict=True)]
                mean, low, high = stats.bootstrap_mean(diff)
                new_only, old_only, p = stats.sign_test(va, vb) if better else stats.sign_test([not x for x in va], [not y for y in vb])
                out.append(f"| {kind} | {LABELS[system]} | {name} | {stats.percent(rb)} | {stats.percent(ra)} | {_signed(mean, low, high)} | "
                           f"{new_only} / {old_only} | {_p(p)} |")
    return [f"**{label}** ({'worlds not used to find or design the change' if held_out else 'the worlds the problem was found on'}; "
            f"{len({k[0] for k in a})} worlds, same deals in both columns)", ""] + out + [""]


def headline(final: dict, briefs_path: Path, before: dict | None) -> list[str]:
    """The main numbers in a few lines, computed from the same rows as the tables below."""
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in final["rows"]:
        by[(r["kind"], r["system"])].append(r)
    c = "counterintuitive"
    di, rag, win = by[(c, "ours_similar")], by[(c, "naive_rag")], by[(c, "neighbour_wins")]
    lines = [f"- **Retrieval (no model, {len(di)} cases, worlds never used for design).** Where history contradicts the obvious play, Deal Intelligence "
             f"puts a good play in its top three {stats.percent(_rate(di, 'hit3')[0])} of the time and the trap {stats.percent(_rate(di, 'trap3')[0])}, "
             f"against {stats.percent(_rate(rag, 'hit3')[0])} and {stats.percent(_rate(rag, 'trap3')[0])} for a plain \"what did similar deals do?\" lookup, "
             f"and {stats.percent(_rate(win, 'hit3')[0])} and {stats.percent(_rate(win, 'trap3')[0])} for \"similar winners only\"."]
    if briefs_path.exists():
        rows: dict[tuple, dict] = {}
        for line in briefs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["seed"], r["deal"], r["arm"])] = r
        ready = [r for r in rows.values() if r.get("status") == "ready"]
        arm = lambda name, kind: [r for r in ready if r["arm"] == name and r["kind"] == kind]  # noqa: E731
        if ready:
            plain, pasted, di2, les = arm("none", c), arm("longctx", c), arm("similar", c), arm("hindsight", c)
            lines.append(
                f"- **Written briefs (a real model, {len(plain)} briefs per condition).** A plain model recommends a good play {stats.percent(_rate(plain, 'hit3')[0])} of the time "
                f"and the trap {stats.percent(_rate(plain, 'trap3')[0])}. With Deal Intelligence's memory: good play {stats.percent(_rate(di2, 'hit3')[0])}, trap {stats.percent(_rate(di2, 'trap3')[0])}; "
                f"with lessons: good play {stats.percent(_rate(les, 'hit3')[0])}, trap {stats.percent(_rate(les, 'trap3')[0])}. Pasting every past deal into the prompt "
                f"(about {round((sum(r['input_tokens'] for r in pasted) / max(1, len(pasted))) / max(1.0, sum(r['input_tokens'] for r in di2) / max(1, len(di2))))}x the input tokens) "
                f"gave good play {stats.percent(_rate(pasted, 'hit3')[0])} and trap {stats.percent(_rate(pasted, 'trap3')[0])}.")
            ctl = [(name, arm(name, "control")) for name in ("none", "similar", "hindsight")]
            lines.append("- **No harm where the obvious play is right.** Good play recommended: " + ", ".join(
                f"{ARM_LABELS[n]} {stats.percent(_rate(rs, 'hit3')[0])}" for n, rs in ctl) + ".")
    if before is not None:
        b = [r for r in before["rows"] if r["system"] == "ours_similar" and r["kind"] == c]
        lines.append(f"- **One ranking weakness found and fixed.** Before the fix Deal Intelligence recommended the trap in its top three "
                     f"{stats.percent(_rate(b, 'trap3')[0])} of the time on the held-out worlds; after, {stats.percent(_rate(di, 'trap3')[0])}.")
    return ["## Headline", "", *lines, "",
            "**Read the limits before quoting any of this**: synthetic deals with rules the designer planted, small samples in the model layer "
            "(some differences are not statistically significant), one model at low thinking effort, and baselines that are deliberately simple. "
            "On the simplest outcome-aware baseline (\"similar winners only\") the retrieval advantage is small and not significant; the clearest "
            "gains are over outcome-blind approaches and over a plain model.", ""]


def retrieval_section(data: dict) -> tuple[list[str], dict[str, str]]:
    rows = data["rows"]
    seeds, per = len(data["design"]["seeds"]), data["design"]["n_test"]
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["kind"], r["system"])].append(r)
    systems = [s for s in data["design"]["systems"]]
    out = ["## Layer 1: retrieval and ranking (no model involved)", "",
           f"{seeds} generated worlds, each with 60 closed deals in memory and {per} open deals to advise on "
           f"({seeds * per} cases per system). A case is marked right when the top three recommended plays include a good play, "
           "and marked wrong on the trap when they include the trap play for that situation. Intervals are 95%.", ""]
    charts_out: dict[str, str] = {}
    for kind, title in KINDS:
        n = len(by[(kind, systems[0])])
        out += [f"### {title} (n = {n} cases per system)", "",
                "| System | Good play in top 3 | Good play first | Trap in top 3 | Trap flagged to avoid | Good play wrongly flagged |",
                "|---|---|---|---|---|---|"]
        for s in systems:
            rs = by[(kind, s)]
            flagged = "n/a" if not s.startswith("ours") else stats.interval(*_rate(rs, "flagged"))
            alarm = "n/a" if not s.startswith("ours") else stats.interval(*_rate(rs, "false_alarm"))
            out.append(f"| {LABELS[s]} | {stats.interval(*_rate(rs, 'hit3'))} | {stats.interval(*_rate(rs, 'hit1'))} | "
                       f"{stats.interval(*_rate(rs, 'trap3'))} | {flagged} | {alarm} |")
        out.append("")
        groups = ["Good play in top 3", "Good play first", "Trap in top 3"]
        series = {LABELS[s]: [_rate(by[(kind, s)], k) for k in ("hit3", "hit1", "trap3")] for s in systems}
        charts_out[f"retrieval_{kind}.svg"] = charts.bar_chart(
            f"{title}: what each system recommends", f"{n} cases per system, 95% intervals. Higher is better except the last group.", groups, series)
    out += ["### Deal Intelligence against the simple baselines, case by case", "",
            "Same deals, paired. A positive difference means Deal Intelligence is better. The sign test counts only the deals where the two disagree.", "",
            "| Situation type | Against | Measure | Difference | Only DI right | Only baseline right | Test |", "|---|---|---|---|---|---|---|"]
    for kind, title in KINDS:
        for base in ("naive_rag", "neighbour_wins", "popularity"):
            for key, name in (("hit3", "good play in top 3"), ("trap3", "trap in top 3")):
                a = by[(kind, "ours_similar")]
                b = by[(kind, base)]
                mean, low, high, a_only, b_only, p = _paired(a, b, key)
                if key == "trap3":  # fewer is better, so show "traps avoided"
                    mean, low, high, a_only, b_only = -mean, -high, -low, b_only, a_only
                    name = "trap avoided"
                out.append(f"| {kind} | {LABELS[base]} | {name} | {_signed(mean, low, high)} | {a_only} | {b_only} | {_p(p)} |")
    out.append("")

    curve = defaultdict(list)
    for r in data["curve_rows"]:
        if r["kind"] == "counterintuitive":
            curve[(r["system"], r["n_train"])].append(r)
    sizes = data["design"]["curve_sizes"]
    shown = ("popularity", "naive_rag", "neighbour_wins", "ours_similar")
    charts_out["learning_curve.svg"] = charts.line_chart(
        "Does it learn? Good play in the top three as closed deals accumulate",
        "Situations where history contradicts the obvious play. Same open deals at every size; shaded bands are 95% intervals.",
        [float(s) for s in sizes], {LABELS[s]: [_rate(curve[(s, n)], "hit3") for n in sizes] for s in shown})
    charts_out["learning_curve_trap.svg"] = charts.line_chart(
        "Does it stop recommending the trap as closed deals accumulate?",
        "Share of cases where the trap is among the top three plays (lower is better).",
        [float(s) for s in sizes], {LABELS[s]: [_rate(curve[(s, n)], "trap3") for n in sizes] for s in shown}, y_label="share of deals with the trap recommended")
    out += ["### Learning curve", "", "![good play in top three vs closed deals](learning_curve.svg)", "",
            "![trap recommended vs closed deals](learning_curve_trap.svg)", ""]
    return out, charts_out


def briefs_section(path: Path) -> tuple[list[str], dict[str, str], dict]:
    if not path.exists():
        return [], {}, {}
    rows: dict[tuple, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[(r["seed"], r["deal"], r["arm"])] = r  # a later successful row replaces an earlier failure
    ready = [r for r in rows.values() if r.get("status") == "ready"]
    failed = len(rows) - len(ready)
    if not ready:
        return [], {}, {}
    arms = [a for a in ARM_LABELS if any(r["arm"] == a for r in ready)]
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in ready:
        r["clean"] = not r["flags"]
        by[(r["kind"], r["arm"])].append(r)
    models = sorted({r["model"] for r in ready if r.get("model")})
    seeds = sorted({r["seed"] for r in ready})
    tokens_in, tokens_out = sum(r["input_tokens"] for r in ready), sum(r["output_tokens"] for r in ready)
    summary = {"briefs": len(ready), "failed": failed, "models": models, "seeds": seeds, "input_tokens": tokens_in, "output_tokens": tokens_out}
    out = ["## Layer 2: the written briefs (a real model, one call per brief)", "",
           f"{len(ready)} briefs from {len(seeds)} generated worlds, written by `{', '.join(models)}` at temperature 0 with the same prompt in every "
           f"condition; only what the model is shown about past deals differs. Marked by code against the planted rules, not by another model."
           + (f" {failed} brief(s) failed and are excluded." if failed else ""), ""]
    charts_out: dict[str, str] = {}
    for kind, title in KINDS:
        n = len(by[(kind, arms[0])])
        out += [f"### {title} (n = {n} briefs per condition)", "",
                "| Condition | Good play recommended | Trap recommended | Clean citations | Median seconds | Input tokens / brief |", "|---|---|---|---|---|---|"]
        for a in arms:
            rs = by[(kind, a)]
            secs = sorted(r["seconds"] for r in rs)
            out.append(f"| {ARM_LABELS[a]} | {stats.interval(*_rate(rs, 'hit3'))} | {stats.interval(*_rate(rs, 'trap3'))} | "
                       f"{stats.interval(*_rate(rs, 'clean'))} | {secs[len(secs) // 2]:.0f} | {sum(r['input_tokens'] for r in rs) // max(1, len(rs))} |")
        out.append("")
        charts_out[f"briefs_{kind}.svg"] = charts.bar_chart(
            f"{title}: what the written briefs recommend", f"{n} briefs per condition, 95% intervals.",
            ["Good play recommended", "Trap recommended"], {ARM_LABELS[a]: [_rate(by[(kind, a)], "hit3"), _rate(by[(kind, a)], "trap3")] for a in arms})
    out += ["### Against the plain model, brief by brief", "",
            "| Situation type | Condition | Measure | Difference | Only this right | Only plain model right | Test |", "|---|---|---|---|---|---|---|"]
    for kind, _title in KINDS:
        for a in arms:
            if a == "none":
                continue
            for key, name in (("hit3", "good play recommended"), ("trap3", "trap avoided")):
                mean, low, high, a_only, b_only, p = _paired(by[(kind, a)], by[(kind, "none")], key)
                if key == "trap3":
                    mean, low, high, a_only, b_only = -mean, -high, -low, b_only, a_only
                out.append(f"| {kind} | {ARM_LABELS[a]} | {name} | {_signed(mean, low, high)} | {a_only} | {b_only} | {_p(p)} |")
    out.append("")
    return out, charts_out, summary


def main() -> None:
    final = RESULTS / "heldout_after.json"
    data = json.loads(final.read_text(encoding="utf-8"))
    design = data["design"]
    r_lines, r_charts = retrieval_section(data)
    b_lines, b_charts, b_summary = briefs_section(RESULTS / "briefs.jsonl")
    fix_lines: list[str] = []
    pairs = (("heldout_before.json", "heldout_after.json", "On held-out worlds", True), ("dev_before.json", "dev_after.json", "On the development worlds", False))
    if all((RESULTS / f).exists() for pair in pairs for f in pair[:2]):
        fix_lines = ["## What this evaluation found, and the one change it led to", "",
                     "The first run (worlds 1 to 10) showed a weakness in the product's own ranking. Plays were ordered first by whether the catalogue says "
                     "they address the deal's open objection. Where this company's history contradicts the catalogue (the obvious play keeps losing), "
                     "that label still put the losing play near the top, so Deal Intelligence recommended the trap more often than the simple "
                     "\"similar winners only\" baseline. The change: the catalogue label stops boosting a play once it has been used in at least "
                     "3 similar deals and won fewer than half of them. Both numbers are plain defaults, fixed before looking at new worlds.", ""]
        for before_file, after_file, label, held_out in pairs:
            fix_lines += fix_section(json.loads((RESULTS / before_file).read_text(encoding="utf-8")),
                                     json.loads((RESULTS / after_file).read_text(encoding="utf-8")), label, held_out)
    head = ["# Deal Intelligence: does the memory help, measured?", "",
            "This report is generated by `python -m evaluation.report` from raw result files in this folder. It tests a mechanism "
            "on **synthetic deals with rules planted by the designer**, so it shows what the system does when history contains a "
            "pattern. It does **not** show what it would earn a real sales team.", "",
            "## The claim", "",
            "> When a company's own history says its obvious play loses, Deal Intelligence recommends the play that wins and warns about the one that loses. "
            "A plain model, or a plain \"what did similar deals do?\" lookup, cannot know that.", "",
            "## The world", "",
            f"A fictional seller with ten sales plays. Each deal turns on one objection. Teams reach for the obvious play {OBVIOUS_SHARE:.0%} of the time. "
            f"A good play wins {P_WIN['good']:.0%} of the time, a neutral one {P_WIN['neutral']:.0%}, a trap {P_WIN['trap']:.0%}.", "",
            "| Situation | Type | Obvious play | Good play | Trap |", "|---|---|---|---|---|"]
    for s in SITUATIONS.values():
        head.append(f"| {s.name.replace('_', ' ')} | {s.kind} | {s.obvious} | {', '.join(s.good)} | {', '.join(s.trap)} |")
    head += ["", "In the three counter-intuitive situations the obvious play is the trap (company-specific, like \"our legal fast-track makes buyers "
             "suspicious\"). In the two control situations the obvious play is good; a system that knows nothing should do well there, and memory "
             "must not make it worse. The control traps are arbitrary rules of this world, so read the trap columns for controls with that in mind.", "",
             "## Systems compared", "",
             "- **Popularity**: the plays with the best win rate across all closed deals, whatever the situation.",
             "- **Plain retrieval**: find similar deals (same objection) and recommend what they used most. Ignores whether they won.",
             "- **Similar winners only**: the same, using only deals that won. No contrast with losses and no warnings.",
             "- **Deal Intelligence**: the product's own retrieval, ranking and avoid list, over the local database.",
             "- **Deal Intelligence + lessons**: the same plus lessons built from outcomes (the free local memory; no Hindsight Cloud).", ""]
    notes = ["## Limits you should know", "",
             "- The data is synthetic and the rules are the designer's. A system that finds them has passed a test of mechanism, not of business value.",
             "- Baselines are deliberately simple; a stronger retrieval system might close some of the gap.",
             "- Intervals describe sampling noise across generated deals and worlds, not uncertainty about real sales.",
             "- Layer 2 uses one model at temperature 0; other models would differ.",
             "- The next step is not a bigger benchmark but a pilot on a real team's closed deals.", "",
             "## Reproduce", "",
             "```powershell", "cd deal-intelligence",
             "$env:PYTHONPATH = \"src;.\"",
             ".\\.venv\\Scripts\\python.exe -m evaluation.retrieval_eval 10        # free, about 4 minutes",
             ".\\.venv\\Scripts\\python.exe -m evaluation.brief_eval --estimate    # how many model calls, no network",
             ".\\.venv\\Scripts\\python.exe -m evaluation.brief_eval --seeds 2     # the real run, resumes where it stopped",
             ".\\.venv\\Scripts\\python.exe -m evaluation.report                   # this file and the charts", "```", ""]
    summary_line = []
    if b_summary:
        summary_line = [f"Model usage for Layer 2: {b_summary['briefs']} briefs, {b_summary['input_tokens']:,} input and {b_summary['output_tokens']:,} output tokens.", ""]
    before = json.loads((RESULTS / "heldout_before.json").read_text(encoding="utf-8")) if (RESULTS / "heldout_before.json").exists() else None
    body = head + headline(data, RESULTS / "briefs.jsonl", before) + fix_lines + r_lines + b_lines + summary_line + notes
    for name, svg in {**r_charts, **b_charts}.items():
        (RESULTS / name).write_text(svg, encoding="utf-8")
    # place the images right after their tables
    text = "\n".join(body)
    for kind, _t in KINDS:
        marker = f"### {dict(KINDS)[kind]} (n = "
        text = text.replace(marker, f"![{kind} retrieval](retrieval_{kind}.svg)\n\n{marker}", 1) if False else text
    gallery = ["## Charts", ""] + [f"![{name}]({name})" for name in sorted({**r_charts, **b_charts}) if not name.startswith("learning")] + [""]
    text = text.replace("## Limits you should know", "\n".join(gallery) + "\n## Limits you should know", 1)
    (RESULTS / "REPORT.md").write_text(text, encoding="utf-8")
    print(f"wrote {RESULTS / 'REPORT.md'} and {len(r_charts) + len(b_charts)} charts")


if __name__ == "__main__":
    sys.exit(main())
