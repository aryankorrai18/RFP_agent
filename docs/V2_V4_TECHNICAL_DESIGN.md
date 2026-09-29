# V2-V4 implementation

## Status

V2 and V3 are implemented as a token-efficient learning loop. V4's corpus, variants, ablations,
metrics and deterministic gate are implemented and frozen offline. The live blind pairwise judge
runs in the app (`backend/v1/judge.py`). The current Virtusa evidence contains 12 blind verdicts
(six pairs in both presentation orders) and six human blind spot-checks; results are in
`eval/v4/live_judge_results.json`, and the combined report records `v4_complete: true`.

No implementation or test in this work calls Gemini, Anthropic, or live Hindsight.

## Version isolation

RFP_RETRIEVAL_MODE selects the behavior:

- none: fact-sheet-only retrieval ablation.
- plain: frozen V1 Hindsight order.
- outcome: Hindsight candidates reranked from exact SQLite signals.

The V1 prompt remains v1.0; ranking changes source selection, not the frozen prompt template.
Every draft stores its retrieved candidates and ranking factors for replay and audit.

## V2: fast feedback

Reviewer action, optional reason tags and a 1-5 rating are attributed only to the ANS sources
actually cited by the draft. Counts are kept in answer_stats. Length feedback is learned as a
client preference and is not treated as an answer-quality failure.

The deterministic score is reciprocal semantic rank times smoothed quality times freshness times
context. The API and UI show the factors and human-readable reasons.

The evidence checker adds supported, partial, unsupported, or unverifiable to every claim using
local lexical coverage. Its provider-neutral output can later be replaced by an entailment judge.

## V3: slow outcomes

Project outcomes add small causal credit:

- won: +0.1
- lost for technical fit or response quality: -0.1
- price, budget, incumbent or relationship loss: 0

Updates are idempotent: changing an outcome applies only the delta from the previous result.
Section debrief scores apply credit only to cited sources in that section. Freshness decays on a
730-day half-life by default. Superseding is explicit, keeps an audit event, and removes the old
answer from live retrieval.

GET /v1/memory/learned exposes the append-only human-readable journal.

## V4: frozen offline gate

Run python -m eval.v4.run_offline. It generates 5 held-out synthetic RFP identities times 40
questions and evaluates no memory, plain retrieval, outcome memory, and ablations for review
signals, context and freshness.

The deterministic gate requires outcome memory to improve right-source top-3 accuracy over plain
retrieval by at least 0.20. The protocol is frozen in eval/v4/protocol-v1.json.

This scaffold is deliberately easy and synthetic. Before claiming complete V4:

1. Freeze a representative, non-synthetic held-out corpus.
2. Run blinded pairwise draft judgments with a fixed judge prompt and model.
3. Human-review a random sample for correctness and leakage.
4. Record cost, token count, model version and raw results.

## Data ownership

SQLite remains authoritative. Hindsight holds two banks per workspace: searchable, verbatim copies of
approved answers (chunks mode, no LLM), and lessons about what happened to them (Hindsight's concise
extraction, observations, Reflect and a mental model). It is never the outcome database: outcomes,
reviews and debriefs are recorded in SQLite first and sent to Hindsight as lessons. Every recalled ID
is revalidated against SQLite before drafting. See [HOW_HINDSIGHT_IS_USED.md](HOW_HINDSIGHT_IS_USED.md).
