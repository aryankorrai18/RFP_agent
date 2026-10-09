# Evaluating the RFP Memory Assistant

This folder answers one question with numbers: **does the memory make drafted answers better and safer?**

Read the result first: [results/REPORT.md](results/REPORT.md) (generated from the raw files next to it).

## How it works

A synthetic proposal-team world is generated from a seed (no model call): a fact sheet, a library of approved answers imported
from eleven past proposals (one lost on technical fit), and 24 new questions. Each question is one of eight kinds with
**planted rules**: a stale answer next to a current one, two answers where reviewers kept rewriting the wrong one, an answer for
the right industry next to one for another, a wrong answer from a lost bid, an approved answer the current fact sheet
contradicts, a value only the fact sheet knows, one clear answer, and a question nobody can answer. Because the rules are known,
every retrieval and every drafted answer is marked right or wrong **by code**. No model judges another model.

Three layers:

| Layer | What runs | Model calls |
|---|---|---|
| 1. Retrieval and ranking | The product's own retrieval and ranking against simple baselines (plain search, newest first) | none |
| 2. Drafted answers | The product's own drafting (prompt, grounding checks, SME handoff) under four conditions with the same model: no past answers, search order, ranked by memory, ranked by memory plus lessons | one per draft |
| 3. Learning | How the ranking changes as review history accumulates (a curve over rounds of reviewer verdicts) | none |

Worlds 1 to 10 were the development worlds, where weaknesses were found (see the report). Worlds 101 and up were never used to
design anything: the headline numbers come from them.

## What is real and what is a stand-in

Real: importing past proposals, outcome credit, review statistics, lessons, the sync, `retrieve()`, the gated ranking,
`draft_requirement()`, the grounding and evidence checks, the SME handoff, and (in layer 2) the model.

Stand-ins: the two Hindsight banks are local ([memory.py](memory.py)). The answer bank ranks by stemmed, rarity-weighted word
overlap instead of semantic similarity; the lessons bank keeps what the product retains and recalls by tag. Review history is
planted in the shape the product's review path writes it. See the report's limits.

## Run it

From `rfp-v0`, on Windows PowerShell:

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python.exe -m evaluation.retrieval_eval --seeds 10 --first-seed 101 --out heldout.json
.\.venv\Scripts\python.exe -m evaluation.retrieval_eval --seeds 10 --first-seed 1 --out dev.json
.\.venv\Scripts\python.exe -m evaluation.retrieval_eval --seeds 10 --first-seed 1 --no-curve --variant flat --out dev_flat.json
.\.venv\Scripts\python.exe -m evaluation.draft_eval --estimate                  # calls and rough cost, no network
.\.venv\Scripts\python.exe -m evaluation.draft_eval --seeds 3 --effort low      # the real run; resumes where it stopped
.\.venv\Scripts\python.exe -m evaluation.report
```

`python -m evaluation.context_diagnostic` is a second free diagnostic: how large the industry/client bonus would have to be to change a ranking
(arithmetic on the product's own rule, plus a run with the bonus raised in a wrapper; the product is not modified).

`--variant flat` is a diagnostic, not the product: it switches the ordering inside a group of about equally relevant answers to
the unused `within="flat"` mode, to show what the gate protects against. The draft run reads the model and key from the app's own
settings; nothing is printed.

## The model layer as it actually ran (2026-10-07, frozen)

Free-tier sized instead of the planned 288 calls. 42 calls in all, every row recording model, effort, prompt version and retrieval mode:

| File | What |
|---|---|
| `results/drafts_run1_INVALID_*` | The first 288-call attempt: quota ran out and the model (at **low** effort) escalated everything. Invalid, kept as evidence. |
| `results/drafts_run2.log` | Canary at low effort: 0 of 3. |
| `results/drafts_run3.log`, `results/drafts_mini.jsonl` | Canary at the app's own **medium** effort: 3 of 3; then 12 drafts (3 questions x 4 conditions) with prompt v1.0: 5 right, 0 wrong values, 7 escalated. |
| `results/drafts_mini_v1_1.jsonl` | Same 12 with prompt v1.1 (past answers show their proposal date; ranked lists say they are ordered by trust): 8 right, 0 wrong, 4 escalated. |
| `results/drafts_mini_w102.jsonl` | Replication on world 102: 8 right, 0 wrong, 4 escalated. |

Across the 24 v1.1 drafts: when memory ranked the right answer first the model used it (8 of 8); when plain search put the wrong answer
first it escalated (2 of 2), and it also escalated when plain search happened to put the right one first (2 of 2); the fact sheet won
every conflict (8 of 8); no wrong value was ever stated. **Promising, replicated mechanism evidence on synthetic data; not proof of
business effect.** Prompt v1.1 and the ranking are frozen until real data shows a problem.

## What this does not show

It shows what the system does when the library and the review history hold a pattern. It does not show what it would earn a real
team: the data is synthetic, the rules are the designer's, the search is lexical, and the baselines are deliberately simple. The
next step is a pilot on a real team's past proposals, and re-running layer 1 against a real Hindsight bank.

## Files

`world.py` (the world and its rules), `memory.py` (local lexical banks), `harness.py` (loads a world into the product, the
baselines, the marking), `retrieval_eval.py`, `draft_eval.py`, `stats.py` (Wilson, bootstrap and exact sign test), `charts.py`
(plain SVG), `report.py`. Tests: `tests/test_evaluation.py`.
