# Evaluating Deal Intelligence

This folder answers one question with numbers: **does the memory actually make the recommendations better?**

Read the result first: [results/REPORT.md](results/REPORT.md) (generated from the raw files next to it).

## How it works

A synthetic sales world is generated from a seed (no model call). It has **planted rules**: in three situations the
obvious play (the one the play catalogue says addresses the objection) is a trap in this company's history and a less
obvious play wins; in two control situations the obvious play is right. Teams over-use the obvious play and outcomes are
noisy, so only a system that learns from *outcomes* can find the rule. Because the rules are known, every recommendation
is marked right or wrong **by code**. No model judges another model.

Two layers:

| Layer | What runs | Model calls |
|---|---|---|
| 1. Retrieval and ranking | The product's own retrieval, ranking and avoid list against three simple baselines (popularity, outcome-blind retrieval, similar-winners-only), on many worlds | none |
| 2. Written briefs | The product writes a brief under four conditions with the same model and prompt: nothing about the past, every deal pasted in, similar deals with ranking and warnings, and the same plus lessons | one per brief |

Worlds 1 to 10 were the development worlds, where a weakness in the ranking was found (see the report). Worlds 101 and up
were never used to design anything: the headline numbers come from them.

## Run it

From `deal-intelligence`, on Windows PowerShell:

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python.exe -m evaluation.retrieval_eval --seeds 10 --first-seed 101 --out heldout_after.json
.\.venv\Scripts\python.exe -m evaluation.brief_eval --estimate                 # calls and rough cost, no network
.\.venv\Scripts\python.exe -m evaluation.brief_eval --seeds 3 --effort low     # the real run; resumes where it stopped
.\.venv\Scripts\python.exe -m evaluation.report
```

`--variant before` on the retrieval run switches off the one ranking rule the evaluation led to, so the old behaviour can
be re-measured on new worlds. The brief run reads the model and key from the app's own settings; nothing is printed.

## What this does not show

It shows what the system does when the history holds a pattern. It does not show what it would earn a real team: the data
is synthetic, the rules are the designer's, and the baselines are deliberately simple. The next step is a pilot on a real
team's closed deals.

## Files

`world.py` (the world and its rules), `harness.py` (loads a world into the product, the baselines, the marking),
`retrieval_eval.py`, `brief_eval.py`, `stats.py` (Wilson, bootstrap and exact sign test), `charts.py` (plain SVG),
`report.py`. Tests: `tests/test_evaluation.py`.
