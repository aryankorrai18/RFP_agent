# V4 evaluation

This folder contains the deterministic portion of the V4 evaluation gate and the recorded live
judge evidence.

Run it with:

```powershell
.\.venv\Scripts\python -m eval.v4.run_offline
```

It deterministically generates 20 synthetic history identities and five held-out RFPs with 200
questions, then compares no retrieval, V1 plain retrieval, V2/V3 outcome ranking, and three
ablations. It makes no Gemini or Hindsight calls.

The report sets `v4_complete` only when the deterministic gate passes and the recorded evidence
contains both a completed blind pairwise judge and human spot-checks. The current Virtusa demo
evidence satisfies all three conditions.

## Live judge and human spot-check (in the app)

On a project page, after **Does memory change the draft?** has finished, the **V4 · blind judge**
panel sends each question whose two drafts differ to a judge model:

- **Blind.** The judge sees "Draft A" and "Draft B", never which one used Hindsight's lessons.
- **Both orders.** Each pair is judged twice, once with each draft first, because judge models
  favour whichever comes first. A win counts only when both orders agree; otherwise the pair is
  "depends on order".
- **Not circular.** The judge gets the official facts and the text of every past answer either
  draft was offered, but not which proposals were won or lost.
- **Judge model.** Picked separately. The panel notes when it also wrote the drafts: a model can
  prefer its own writing, but both drafts in a pair come from the same model, so that favours
  neither side. A different model gives a more independent second opinion.
- **Fixed prompt.** `judge-v2` in `backend/v1/judge.py` (v2 states that a claim found in a past answer counts as supported; v1 judges marked such claims unsupported), stored with every verdict.
- **Cost before spending.** The panel shows the number of calls first. Identical pairs are ties at
  no cost, and a pair is never judged twice under the same model and prompt, so a stopped run is
  finished, not repeated.

Under it, **Human spot-check** shows the same pairs blind, in a fixed shuffled order. After each
pick it reveals which draft used lessons, and the judge panel reports how often you agreed with it.

Record the results (no model calls):

```powershell
.\.venv\Scripts\python -m eval.v4.record_live_judge
```

This writes `live_judge_results.json` with the drafting conditions, every un-blinded verdict, token
counts, the served model versions and the spot-check picks, plus how the run differs from
`protocol-v1.json` (corpus and number of runs).

The first deliberately small live attempt is preserved in `LIVE_SMOKE.md`; Gemini returned 503 in
that historical attempt. The later successful Virtusa run is the current evidence in
`live_judge_results.json`.
