# V4 live smoke attempt: 2026-09-28

The low-volume smoke was deliberately limited to one fictional SCIM case.

## External calls

- Hindsight: one live recall succeeded and returned SQLite-backed candidates.
- Configured Gemini 3.5 Flash-Lite: the first drafting request returned 503 high demand after the
  SDK's built-in retries. No draft was returned.
- A no-retry, maximum-128-output-token judge request to the same model also returned 503.
- One no-retry fallback check against Gemini 2.5 Flash-Lite returned 404 because Google no longer
  makes that model available to new users.

No successful Gemini response was produced, so there is no model-reported token usage or live
pairwise verdict to record. Further calls were stopped to protect quota.

## Status

The live Hindsight path passed. Gemini judgment was blocked by temporary provider availability in
this historical attempt. A later Virtusa run completed the two-order judge and human spot-check;
the current status is recorded in `live_judge_results.json` and the offline report now records
`v4_complete: true`.

Retry the cheapest judge-only check later:

    python -m eval.v4.run_live_smoke --judge-only

The full one-case smoke makes one Hindsight call, two drafting calls and one judge call:

    python -m eval.v4.run_live_smoke
