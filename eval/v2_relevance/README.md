# V2 relevance fix

**Principle.** Outcome memory changes the preference among answers that are relevant to the question. It can't override basic relevance: no lesson, however strong, lifts an off-topic answer above a relevant one.

## Why

A live test found it (2026-09-29, `eval/v4/live_learning_loop.json`). After a reviewer rejected the only SCIM answer, an MFA answer took first place for a SCIM question. The V2 score was

    score = relevance (1 / Hindsight rank) x lesson factor (0.25 to 2.5) x freshness x client context

so a candidate ranked 2nd by Hindsight started at half the relevance of the 1st whatever the real gap, and lessons could multiply that by up to 10.

## The rule now (`retrieval_relevance = "gated"`, `backend/v1/ranking.py`)

1. Hindsight returns up to 12 candidates, each with a relevance score (`final`: fused similarity and cross-encoder reranker).
2. The first **group** is every candidate scoring at least `min_share` (default **0.01**) of the best one. The next group is the same over what's left, and so on.
3. Inside a group, the usual V2 score decides (Hindsight's order, lessons, freshness, client context).
4. A candidate never moves above a group that ranked above it.

`retrieval_relevance = "rank"` is the pre-fix rule, kept so earlier results can be reproduced (`RFP_RETRIEVAL_RELEVANCE`, `RFP_RETRIEVAL_RELEVANCE_MIN_SHARE`).

Why `final` and not similarity: on this corpus the raw semantic similarity put an off-topic answer first for 2 of 22 questions (company overview, certifications). Hindsight's `final` score and order never did. The reranker is lopsided (a relevant but less-matched answer can score 0.03 against 0.97), so it's used as a *share of the best*, with a low bar, not as a multiplier.

## Evaluation (no model calls)

- `capture.py` records, once, what Hindsight returns for the 26 questions of the two corpus RFPs in a clean demo workspace: candidates, scores, and the lessons recalled about them (`recall_cache.json`).
- `evaluate.py` scores ranking rules on that frozen input against `samples/corpus_v2/answer_key.json`: 22 questions have a right answer, 4 are deliberate gaps.
- Stress test: for every question, simulate one reviewer rejection of whatever ranks first, and check whether an off-topic answer takes first place although an on-topic one exists.

| Rule | Right answer #1 | Right answer in top 3 | Trap #1 | Off-topic #1 | Off-topic in top 3 (sum) | Off-topic #1 after one rejection |
|---|---|---|---|---|---|---|
| Plain (V1, Hindsight order) | 17 | 22 | 5 | 0 | 11 | n/a |
| **V2 pre-relevance-fix** | 16 | 22 | 3 | **3** | 23 | **2 of 22** |
| V2 gated, share 0.005 | 19 | 22 | 3 | 0 | 15 | 0 of 21 |
| **V2 post-relevance-fix** (gated, share 0.01) | **19** | **22** | **3** | **0** | 15 | **0 of 21** |
| V2 gated, share 0.02 | 19 | 22 | 3 | 0 | 14 | 0 of 21 |
| V2 gated, share 0.05 | 18 | 22 | 4 | 0 | 13 | 0 of 21 |
| V2 gated, memory alone decides inside groups (0.005 to 0.05) | 16 to 17 | 22 | 4 to 5 | 0 to 1 | 13 to 17 | 0 to 1 of 21 |

The share is stable from 0.005 to 0.02; 0.01 is the middle of that range, not a value tuned to one question.

Where the rules disagree on the first answer:

| Question | Right answer | Plain | V2 pre-fix | V2 post-fix |
|---|---|---|---|---|
| 3.2 SCIM (Ashford) | scim_2025 (won) | scim_2023 (lost) | implementation answer (off-topic) | scim_2025 |
| 5.3 Support model | support_2025 | support_2025 | implementation answer (off-topic) | support_2025 |
| MG-05 EU data | residency_2026 | residency_neutral | residency_2026 | residency_2026 |
| MG-09 Accessibility | accessibility | accessibility | AI-training answer (off-topic) | accessibility |

The pre-fix off-topic answers came mostly from the implementation answer, which was in three won proposals, so its lessons outweighed relevance.

What one rejection of the first answer does (21 questions with another on-topic answer):

| Rule | Moves first place to another on-topic answer | Rejected answer stays first | Off-topic answer takes first |
|---|---|---|---|
| V2 pre-fix | 4 | 16 | 2 |
| V2 post-fix | 3 | 18 | 0 |

With the fix, one rejection moves first place only where a comparable relevant answer exists. On this corpus those are identical answers reused in several proposals (pen test, SSO). Elsewhere the rejected answer stays first because the alternatives are worse. For SCIM, the answer from a won bid, once rejected, sits at about neutral (x0.99), while the other SCIM answer came from a bid lost on technical fit (x0.35). The rejection still lowers the answer's standing within about 20 seconds (`eval/v4/live_learning_loop.json`).

## Live regressions (`live_regressions.py`, Hindsight lookups only)

In the demo workspace where the SCIM answer was really rejected (`live_regressions.json`):

- **SCIM:** the rejected answer is first (x0.99), the lost-bid SCIM answer second, and the MFA answer third, no longer first.
- **Pen test:** the three won Ironbridge answers are first to third, and the vague answer from a lost bid is fourth. Memory still promotes what won.

## What this changes elsewhere

- Before/after comparisons and V4 judge verdicts made before the fix are kept and labelled **V2 pre-relevance-fix** (`eval/v4/live_judge_results.json`, `drafting.ranking_version`). The frozen V1 baseline is unaffected: plain retrieval doesn't use this rule.
- A comparison now records the rule it used, and only finishes a stopped run made under the same rule.
- The offline V4 gate (`backend/v4/evaluation.py`) ranks synthetic candidates that have no Hindsight scores, so it still uses the rank rule.

Reproduce: open a clean demo workspace, then `python -m eval.v2_relevance.capture` (Hindsight lookups) and `python -m eval.v2_relevance.evaluate` (no calls).
