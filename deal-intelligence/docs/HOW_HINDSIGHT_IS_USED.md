# How Hindsight is used

Hindsight is the product's searchable memory and learning layer. It is not the application database, it does not write briefs, and it cannot make a deal count as similar. SQLite is the system of record; every Hindsight result is checked against it before use.

All deals named here (Halcyon Software's customers, Cedarline Bank, Larkfield Credit Union and the rest) are synthetic demonstration data.

The client is `hindsight-client` 0.10.1, used through its async methods. It works against Hindsight Cloud (`DEAL_HINDSIGHT_URL` plus `DEAL_HINDSIGHT_API_KEY`) or a local server (`http://127.0.0.1:8888`, no key). The two wrappers are `api/v1/memory.py` (bank 1) and `api/v1/lessons.py` (bank 2).

## Hindsight, or the free local backend

Hindsight is the memory layer this project is built for, and everything below describes it. But the app also runs without a Hindsight account. `DEAL_MEMORY_BACKEND` is `auto`, `hindsight` or `local`. `auto` (the default) uses the local backend when there is no Hindsight API key and the URL is Hindsight Cloud, and Hindsight otherwise; a local Hindsight server URL stays Hindsight.

The local backend (`api/v1/memory_local.py`) keeps the same two banks, with the same documents and tags, in one SQLite FTS5 file, `memory.db`, beside the workspace's database. The differences, stated plainly:

| | Hindsight | Local |
|---|---|---|
| Recall | Semantic, ranked by Hindsight | Keyword (bm25 over stemmed word tokens joined with OR), tags filtered exactly |
| Lessons | Hindsight extracts facts and consolidates observations (concise mode, spends credits) | Lessons stored as written; nothing is extracted |
| Reflect and playbook | Written by Hindsight from remembered facts | Computed locally and deterministically from the recorded outcomes: counts and the first sentences of the most relevant lessons, with a footer saying so |
| Cost | Hindsight credits for lessons, Reflect and the playbook | None; runs offline and sends nothing anywhere |
| Needs | A Hindsight account or server | Nothing |

Local recall is weaker than semantic recall: a deal described in different words may rank lower. It matters less than it sounds, because the SQLite re-check, the structural gate and the counts decide what counts, and Hindsight's rank was only ever a tiebreak. `GET /v1/status` reports `hindsight.backend` (`local` or `hindsight`). The rest of this page describes the Hindsight backend.

## Two banks per workspace, and why

| | Bank 1: interactions | Bank 2: lessons |
|---|---|---|
| Default name | `deal-interactions-<workspace id>` | `deal-lessons-<workspace id>` |
| Retain mode | `chunks`: text kept verbatim | `concise`: Hindsight extracts facts |
| Hindsight LLM | Not used | Used, spends Hindsight credits |
| Observations | Off | On: facts consolidate into observations, asynchronously |
| Reflect mission | None | Set |
| Speed | Instant | Retained in a batch with `retain_async=True`; consolidation finishes later |
| Holds | One summary per closed deal; interactions of open deals and same-account history | One outcome lesson per closed deal and one lesson per play used on it |
| Used for | Finding which closed deals resemble an open deal | Breaking ties between plays; the playbook |

They are separate because they do different jobs and cost differently. Bank 1 must be exact, fast and free of model interpretation: a closed-deal summary is evidence that a brief can cite, so it must be the words the application wrote. Bank 2 deliberately lets Hindsight interpret, extract and consolidate, which is slower and spends credits, so it is kept out of the citation path.

Bank 1 is created like this (`HindsightMemory.ensure_bank`):

```python
await client.acreate_bank(bank, name="Deal interactions",
                          retain_extraction_mode="chunks", enable_observations=False)
```

`chunks` is set explicitly so the bank stays verbatim even if the server is later given an LLM. The Memory page and the status card show the bank's actual extraction mode, so a Cloud bank cannot silently run with LLM extraction.

Bank 2 is created like this (`HindsightLessons.ensure_bank`):

```python
await client.acreate_bank(bank, name="Deal lessons", mission=BANK_MISSION, retain_mission=BANK_MISSION,
                          retain_extraction_mode="concise", enable_observations=True,
                          reflect_mission=REFLECT_MISSION)
```

`BANK_MISSION` tells Hindsight to remember each deal's situation, the plays used and whether the deal was won or lost and why. `REFLECT_MISSION` tells it to base every statement on remembered outcomes, name deal and play ids, say when a loss reason such as price is not something a play could have changed, and never invent customer facts. An already-existing bank (HTTP 400 or 409) is not an error.

## Document ids and tags

The document id is the application's own id, so retaining the same id again replaces the document.

| Bank | Document id | Example |
|---|---|---|
| 1 | Closed-deal summary: the deal code | `D-004` |
| 1 | Interaction: the interaction code | `INT-0012` |
| 2 | Outcome lesson | `lesson-outcome:D-018` |
| 2 | Play lesson | `lesson-play:D-018:PLAY-05` |

Bank 1 tags:

- Summaries: `kind:deal_summary`, `industry:<slug>`, `segment:<segment>`, `result:won|lost`, `loss:<reason>`, `objection:<type>` (one per objection), `competitor:<slug>` (one per competitor).
- Interactions: `kind:interaction`, `deal:<code>`, `account:<slug>`.

Bank 2 tags: `kind:deal_outcome` or `kind:play_result`, `signal:positive|negative|neutral`, `deal:<code>`, `result:<won|lost>`, `industry:`, `segment:`, `objection:`, `competitor:`, `loss:`, and for play lessons `play:<code>`.

## What is retained, and when

All writes go to SQLite first and are marked `pending`; the outbox (below) then sends them.

- **A deal summary per closed deal.** Written when a person records an outcome (and when the demo seed inserts closed deals), once the deal has signals. A re-recorded outcome rewrites the summary under the same `D-` id. The summary leads with the situation and ends with the outcome, and it contains no account name, so semantic recall matches the shape of a deal rather than who it was with. A summary of a lost deal looks like this (the seeded Larkfield deal after it is recorded as lost):

  ```text
  A mid-market fintech deal. Objections: SSO (stayed unresolved), pricing (raised). The champion went silent.
  The economic buyer was not engaged. Plays used: PLAY-05 (Discount offer). Outcome: the deal was lost,
  because of an unresolved objection.
  ```

- **Interactions of open deals,** as they are added (`INT-` ids, text prefixed with the kind, date, author and subject). Interactions of a closed deal are sent only if the same account also has an open deal (same-account history); otherwise they stay `skipped` and are never sent. Deleting a deal is a soft delete in SQLite and queues a `delete_document` call for the deal's summary and each of its interactions.
- **Lessons per outcome and per play,** built from what SQLite records about each closed deal: one outcome lesson, plus one lesson for each play used. They are rewritten if the outcome is re-recorded and dropped from SQLite if the deal is reopened or the play is unticked.

Lessons are written in plain sentences. For example, the outcome lesson and one play lesson for the Larkfield loss:

```text
Deal D-018, a mid-market fintech deal, was lost because of an unresolved objection. An SSO objection stayed
unresolved; a pricing objection raised. The champion went silent. The economic buyer was not engaged.
Plays used: PLAY-05 (Discount offer).

Play PLAY-05 (Discount offer) was used in deal D-018, a mid-market fintech deal. An SSO objection stayed
unresolved; a pricing objection raised. The champion went silent. The economic buyer was not engaged.
The deal was lost because of an unresolved objection, a problem a play could plausibly have changed, which
counts against the play.
```

Bank 1 documents are retained with:

```python
await client.aretain(bank, content=item.content, document_id=item.code, tags=list(item.tags),
                     timestamp=item.timestamp, context=item.context, metadata=dict(item.metadata))
```

Lessons are retained in batches of 20:

```python
await client.aretain_batch(bank, items=items, retain_async=True)
# each item: content, timestamp, document_id, tags, context, metadata={"lesson_key", "signal"}
```

## Recall by tags, validated against SQLite

To find similar deals, the application builds the recall query from the open deal's extracted signals. For the seeded Cedarline Bank deal it is a plain description of the situation, with no account name:

```text
A mid-market fintech deal at the evaluation stage. Objections: SSO (unresolved), security review (raised),
feature gap (addressed). Competing against Brightline. A champion is engaged. The economic buyer is not engaged.
```

and the call is:

```python
await client.arecall(bank, query=query, tags=["kind:deal_summary"], tags_match="all_strict", budget="mid")
```

`all_strict` restricts recall to documents that carry every listed tag, so interactions and untagged documents never come back as deal summaries. Results are de-duplicated by `document_id` (a long document can return as several chunks) and the first `max(k * 4, 12)` are kept, where `k` is `DEAL_SIMILAR_DEALS_K` (default 10). Hindsight's relevance scores are not calibrated, so no threshold is applied.

Each hit is then validated against SQLite. It is dropped if SQLite no longer has the deal, the deal is not closed, the deal is the open deal itself, or the structural gate rejects it. A closed deal that is not yet in Hindsight is added from SQLite so a fresh outcome counts immediately. Hindsight's rank is used only to order deals that are otherwise equal.

## Lesson signal rules

Each lesson carries a signal, set from the recorded outcome in code (`signal_for_outcome`):

| Outcome | Signal |
|---|---|
| Won | `positive` |
| Lost for a quality reason: `security_compliance`, `feature_gap`, `unresolved_objection` | `negative` |
| Lost for any other reason: `price`, `no_decision`, `timing`, `champion_left`, `competitor` | `neutral` |

A loss to price, timing or a plain competitor says little about whether a play was good, so it gives plays no credit or penalty. The play lesson says so in its own text ("which no play could have changed, so this says little about the play").

## How lessons are used

Lessons are a **tiebreak and evidence on plays**, only in `hindsight` mode, and only for plays that already came from gated similar deals:

```python
await lessons.recall(query, tags=sorted(f"play:{code}" for code in candidate_codes),
                     limit=max(len(codes) * 4, 20))
# arecall(bank, query=query, tags=tags, tags_match="any", budget="mid", max_tokens=3000)
```

The hits are validated: a lesson counts only if SQLite still has its document id and every `deal:` tag on it names a closed deal that still exists. A lesson removed from SQLite is therefore ignored even if Hindsight still holds it. Several facts extracted from one lesson count once, and higher-ranked lessons count slightly more (weight `1 / (1 + 0.1 * (rank - 1))`). The result is a net positive or negative signal per play with up to five lesson ids as evidence, shown as "Hindsight lessons: 3 positive, 0 negative".

That signal is the fifth ordering key for plays, after the open-objection match, shared problem keys, won/used ratio and times used (see [Architecture](ARCHITECTURE.md)). It cannot lift a play or a deal past the gate.

Separately from lessons, the plays-to-avoid list is not a Hindsight feature: it is counted in code (`avoid_plays`) over the gated similar deals, from SQLite facts. A play used in at least 3 similar closed deals and in none of the similar won ones is "likely to backfire". It leaves the recommended plays, goes into the prompt's evidence block, and is enforced after the model answers. Memory's part is finding the similar deals it is counted over.

Lessons are **never cited in a brief and never in the drafting prompt.** The prompt contains this deal's records, the flags, the gated similar closed deals, candidate plays with their counts, the warnings and the plays to avoid, and nothing from the lessons bank. The brief checker rejects any cited id that looks like a lesson (`lesson_cited`). The brief cites interactions (`INT-`) and closed deals (`D-`) only.

## Reflect and the playbook

The lessons bank backs one Hindsight **mental model**, the deal playbook, answering "which plays, taken at which point, tend to win which kinds of deals, which objections and stakeholder situations tend to lose them, and what should an account executive do differently". The model id is `deal-playbook`.

- **Reading is free.** `GET /v1/memory/playbook` calls `aget_mental_model(bank, "deal-playbook", detail="content")`. If there is none yet (HTTP 404) it reports `missing` and spends nothing.
- **Creating and refreshing are explicit and spend Hindsight credits.** `POST /v1/memory/playbook/refresh` first syncs pending lessons, then creates the model if needed and refreshes it. The UI asks for a second click ("Confirm: spends Hindsight credits"). Nothing refreshes it automatically.

```python
await client.acreate_mental_model(bank, name="Deal playbook", source_query=PLAYBOOK_QUERY,
                                  id="deal-playbook", max_tokens=900)
await client.arefresh_mental_model(bank, "deal-playbook")
```

On the local backend there is no mental model: the playbook is computed from the stored lessons each time it is read, as counts per play, objection, competitor, industry, segment and loss reason, with a footer ("Computed locally from recorded outcomes, not by Hindsight."). Reading and refreshing are both free.

`lessons.py` also contains a Reflect helper (`areflect(..., budget="low", tags_match="any", include_facts=True)`) with a "deals like this one" query: it describes the open deal's kind (segment, industry), asks what is remembered about similar deals, which won or lost, why and which plays made a difference, asks for at most 10 bullet points, and tells Hindsight to mark any explanation of why a single play mattered as its inference rather than a recorded fact. The tags are the deal's `industry:`, `segment:` and `objection:` tags, matched with `any`.

That Reflect call now backs the **Reflect panel** on the Memory evidence tab. `GET /v1/deals/{id}/memory-says` returns the last answer from the `deal_reflections` table (state `missing` until someone asks), and reading it is free. `POST /v1/deals/{id}/memory-says` syncs pending lessons, asks the lessons bank again, and stores the text and the facts it was based on. On Hindsight Cloud the POST spends Hindsight credits, so the UI asks for a second click; on the local backend it is computed locally and free. Like the lessons, the Reflect text is interpretation: it is shown beside the evidence, never cited and never in the drafting prompt.

`retrieval.py` also contains a tag-filtered recall of one deal's own interactions (`kind:interaction` plus `deal:<code>`). No HTTP route calls it in this version, so it is not part of any brief; a brief reads this deal's own interactions from SQLite. Closed-deal recall is the only bank 1 recall that drives a brief.

## The outbox and outage behaviour

Bank 1 is kept in step by an outbox (`sync.py`). Each deal and interaction row has a status: `pending`, `retained`, `pending_delete`, `deleted` or `skipped`. A sync sends pending rows in order, marks each `retained` on success, and stops at the first `MemoryUnavailable`, recording the attempt count and error on the row. If a row changed while it was being synced it stays `pending`, so the newer content is pushed next time. Lessons use the same pattern with their own pending, retained and failed counts.

Syncs run right after a write that changes memory, then every 30 seconds, backing off to 300 seconds while they fail. `POST /v1/sync` runs one on demand and the status card shows what is waiting.

If Hindsight is down, nothing the user does is lost or blocked: SQLite is written first, a brief is still written from similar deals found in SQLite with the same gate, and the brief says so in `degraded`. If only the lessons bank is down, plays are ranked from similar deals only and `degraded` says so. A deleted deal can never reach a brief, because retrieval re-checks SQLite.

## Why the headline demo flip does not depend on the lessons bank

The demo's headline is that a recorded outcome changes the next brief: the warning goes from "2 of 3 similar deals lost" to "3 of 4", and the newly closed deal is among the cited deals. (The seeded pricing deal D-024 moves the same way: "Discount offer: used in 4 similar deals, 0 won" becomes 5 once Larkfield is recorded as lost with that play.) That comes from SQLite and bank 1 alone. SQLite records the outcome at once; the closed deal then counts either because bank 1 has its summary (retained in chunks mode, no Hindsight LLM) or, if it has not synced yet, because retrieval adds it from SQLite. The counts are computed by the gate and the contrast code over those deals.

The lessons bank is asynchronous: extraction and observation consolidation finish some time after the lesson is accepted, and the app does not wait for them. Lessons only reorder plays that are otherwise tied, so a lesson that has not been consolidated yet cannot change the warnings, the similar deals or the counts. The Memory page and the status card show how many lessons are still waiting.

## What Hindsight adds that a plain database would not

- **Semantic recall of situations.** The recall query is a description of the deal, and closed deals are stored as descriptions of situations. Hindsight ranks them by meaning, so a deal phrased differently can still surface, and the gate then decides whether it counts. A database query needs exact keys.
- **Observation consolidation.** In the lessons bank, Hindsight extracts facts from lessons and consolidates them into observations and a playbook that can be read and refreshed. Hand-written SQL does not do that.
- **Tags as scoped filters.** `kind:`, `deal:`, `play:`, `industry:`, `objection:` and the rest let one bank serve several questions, with strict filtering (`all_strict`) for summaries and broad matching (`any`) for lessons.

This is stated without overselling. With 19 closed deals, a SQLite query on the structured keys finds the same similar deals, which is exactly what the degraded path does (and what the memory self-check uses). Hindsight's contribution at this size is not that it finds deals a database cannot; it is the shape of the system: targeted retrieval that stays small as history grows, lessons that consolidate, and a clean separation of evidence (verbatim, citable) from interpretation (not cited). The comparison on the Deals page shows the no-history, all-deals-in-the-prompt and Hindsight arms side by side so the effect can be judged on the data, and its caveat says that at this size pasting every summary also works.

A live finding, stated plainly. On the seeded pricing deal D-024 with a real model (Gemini), the No history arm recommended the discount play. The arm with every closed deal in the prompt and the Hindsight memory arm both avoided it. So at this size, stuffing every summary also works. What memory adds is targeted retrieval (the prompt carries only the gated similar deals), counted evidence ("used in 4 similar deals, 0 won") with citations, a rule enforced in code rather than hoped for in a prompt, and scale. On the SSO deal Cedarline the recommended plays were identical across the three arms, and the difference was the counted warnings and the cited deals. That was one run per arm and model wording varies; it is not a benchmark.

## Memory self-check

`GET /v1/memory/quality` leaves one closed deal out at a time, rebuilds the recommendations from the others using only SQLite (`build_recommendations(..., source="sqlite", exclude_deal_ids=...)`: no model, no Hindsight, no lessons bank), and compares them with what really happened: for won deals, whether a top-3 recommended play was really used; for lost deals, whether the unresolved objection was warned about and whether a play flagged as backfiring was really used. On the 19-deal seed: won n=9, covered 7, hit 5; lost n=10, warned 6 of the 8 that had an unresolved objection, backfire play flagged in 4. After Larkfield is recorded (20 closed deals): lost n=11, warned 7 of 9, flagged in 4. It is a sanity check on a small sample, not a benchmark, and it exercises the structural path the gate and the counts provide; it does not measure Hindsight's recall.
