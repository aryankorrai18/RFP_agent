# How Hindsight is used

Hindsight is the product's searchable memory and learning layer. It is not the application database and it does not write proposal answers by itself.

## Two banks per workspace

### Approved-answer bank

Contains searchable copies of approved question-answer pairs. When a new requirement is drafted, Hindsight recalls semantically similar answers. The application then:

1. resolves every returned `ANS-*` identifier against SQLite;
2. removes deleted, superseded, or otherwise unavailable answers;
3. groups candidates by semantic relevance;
4. ranks only similarly relevant candidates using quality, freshness, client, industry, and outcome signals;
5. gives the top candidates to the drafting model as citable evidence.

### Lessons bank

Contains experience derived from observable events:

- accepted, edited, rewritten, or rejected drafts;
- reviewer reason tags and ratings;
- won or lost proposals and loss reasons;
- project debriefs;
- explicit answer superseding.

Lessons can change which relevant answer is preferred. They never enter the proposal prompt as factual evidence and cannot override an official company fact.

## Why SQLite remains necessary

SQLite stores the exact state and audit trail. Hindsight is eventually consistent and optimized for recall. Keeping both provides semantic search without giving up deletion, traceability, relational integrity, or deterministic review history.

## Failure behavior

If Hindsight is unavailable:

- application writes continue in SQLite;
- synchronization remains pending and retries in the background;
- drafting falls back to official company facts;
- the UI shows the degraded memory state.

No Hindsight-hosted language model is required. The banks use chunk-based storage and retrieval.
