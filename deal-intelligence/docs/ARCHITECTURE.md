# Architecture

All deals, companies and people named in this document and in the demo are synthetic.

## Runtime flow

1. The browser calls the FastAPI `/v1` API (same origin; `frontend/app.html` is served at `/`).
2. Uploaded files and pasted notes become interactions in SQLite, de-duplicated by content hash. The interactions of an open deal are marked `pending` in the outbox. Interactions of a closed deal are marked `skipped` unless the account also has an open deal (account history).
3. **Read this deal** makes one model call that extracts objections (with a status: raised, addressed, unresolved), competitors, promises, stakeholders and plays used. The output is validated in code before anything is written: an invented objection type, play code or evidence id never reaches the tables. The seeded demo deals already carry signals, so they need no call.
4. A brief request first computes the deterministic flags from SQLite (no model).
5. Retrieval builds a recall query from the open deal's extracted signals (segment, industry, objections and their status, competitors, champion, economic buyer), never from the account name, and recalls closed-deal summaries from the memory tagged `kind:deal_summary`. The memory is Hindsight Cloud, a local Hindsight server, or the free local SQLite memory (see "Memory backends" below); the flow is the same for all three.
6. Every recalled id is re-checked against SQLite (the deal exists, is active, is closed, and is not the open deal itself). Closed deals not yet in the memory are added from SQLite so a deal closed a moment ago still counts.
7. The structural gate keeps only deals that resemble the open deal. Contrast, warnings, plays to avoid and play ranking are then computed in code over the gated deals.
8. One model call writes the brief from three data blocks: this deal, the flags, and the evidence (which includes the plays to avoid). The model's output is never trusted.
9. The output is checked in code: citations must exist and be of the right class, recommended plays must be among the offered candidates, a next step that names a play to avoid is dropped and flagged `avoided_play`, lessons cannot be cited. Findings are stored with the brief. The flags, warnings and avoid list are added to the stored brief whatever the model said.
10. A person records the outcome. SQLite is written first (result, loss reason, plays used, play statistics, journal events), the deal's summary is marked `pending` for re-retention, and lesson rows are rebuilt from the closed deals.
11. A background sync pushes the summary to the interactions bank and the lessons to the lessons bank, immediately after the write and then every 30 seconds. On the local backend the same outbox writes to the local `memory.db`.
12. The next brief recalls the updated memory. `GET /v1/deals/{id}/brief-diff` compares the saved evidence of two briefs of the same deal with no model call.

Two things sit beside this flow and make no model call. `GET /v1/deals/{id}/memory-says` returns the cached Reflect text about deals like this one (`POST` asks the lessons bank again and stores the answer in the `deal_reflections` table). `GET /v1/memory/quality` is the leave-one-out self-check (see below).

## Trust boundaries

- **SQLite** is the exact system of record: deals, interactions, stakeholders, signals, plays, play statistics, lessons, briefs, jobs and the journal.
- **Interactions bank** (bank 1) is a searchable copy of closed-deal summaries and open-deal interactions. On Hindsight it proposes candidates by meaning; on the local backend it proposes them by keyword. Either way it cannot make a deal count.
- **Lessons bank** (bank 2) holds lessons derived from recorded outcomes. It breaks ties between plays and never enters the drafting prompt. It also answers the Reflect question and the playbook.
- **Model provider** (Anthropic, Gemini or Groq) extracts signals and writes the brief. It cannot record an outcome, change memory or cite a lesson.
- **Deterministic code** produces the flags, the gate, the counts, the warnings, the plays to avoid, the play ranking, the citation checks, the diff and the self-check.
- **The human** closes deals. Nothing becomes a closed deal, a lesson or a statistic until a person records the outcome (or the demo seed, which is disclosed as synthetic, inserts it).

## Main modules

| Module | Responsibility |
|---|---|
| `src/deal_intelligence/main.py` | Application lifecycle, workspaces, model selection, security headers |
| `src/deal_intelligence/config.py` | Settings, environment, retrieval modes, `DEAL_MEMORY_BACKEND` and `resolve_memory_backend` |
| `src/deal_intelligence/workspaces.py` | Per-company registry: database, uploads, two bank names |
| `src/deal_intelligence/api/v1/router.py` | HTTP routes and request bodies |
| `src/deal_intelligence/api/v1/db.py` | SQLite tables and the vocabulary the gate is built on |
| `src/deal_intelligence/api/v1/deals.py` | Deal creation, uploads and notes |
| `src/deal_intelligence/api/v1/signals.py` | Deterministic flags, the facts a brief may state, and the one extraction call |
| `src/deal_intelligence/api/v1/retrieval.py` | Memory recall, SQLite validation, retrieval modes, plays to avoid, `source` and `exclude_deal_ids`, degrade paths |
| `src/deal_intelligence/api/v1/ranking.py` | The structural gate, warnings, contrast candidates, play ranking and `avoid_plays` (pure functions) |
| `src/deal_intelligence/api/v1/memory.py` | Interactions bank wrapper (bank 1, Hindsight) |
| `src/deal_intelligence/api/v1/lessons.py` | Lessons bank wrapper (bank 2, Hindsight), lesson text and tags, reflect, playbook |
| `src/deal_intelligence/api/v1/memory_local.py` | Free local backend: `LocalMemory` and `LocalLessons` over SQLite FTS5 (`memory.db`), deterministic Reflect and playbook |
| `src/deal_intelligence/api/v1/reflection.py` | Cached Reflect text about deals like this one (`deal_reflections`) |
| `src/deal_intelligence/api/v1/quality.py` | Leave-one-out memory self-check (SQLite only) |
| `src/deal_intelligence/api/v1/sync.py` | Outbox that keeps bank 1 in step with SQLite |
| `src/deal_intelligence/api/v1/outcomes.py` | Recording a result, play statistics, play credit |
| `src/deal_intelligence/api/v1/briefs.py` | Brief generation and citation validation |
| `src/deal_intelligence/api/v1/brief_prompts.py` | The brief prompt (`brief-v2`) and its hash |
| `src/deal_intelligence/api/v1/experiment.py` | Three-arm comparison and the brief diff |
| `src/deal_intelligence/api/v1/demo.py` | Seeds the synthetic demo workspace with no model calls |
| `src/deal_intelligence/api/v1/jobs.py` | Resumable in-process background jobs |
| `src/deal_intelligence/api/v1/context.py` | Shared runtime context, choice of memory backend, and the sync loop |
| `src/deal_intelligence/providers/` | Anthropic, Gemini and Groq adapters, error explanations, model choice |
| `frontend/app.html` | Browser interface |

## The structural gate

Defined in `ranking.py` (`passes_gate`):

> A closed deal counts as similar to an open deal only if it shares at least `min_shared_keys` problem keys with it (an objection type or a competitor), OR it has the same segment AND the same industry.

`min_shared_keys` defaults to 1 (`DEAL_MIN_SHARED_KEYS`). A deal is never similar to itself. Hindsight's rank is only the final tiebreak, and lesson evidence never moves a deal or a play past the gate.

Kept deals are ordered by problem keys shared (most first), then total shared keys (most first), then Hindsight rank, then deal code, and the first `DEAL_SIMILAR_DEALS_K` (default 10) are kept. Each similar deal carries the keys that matched (for example `objection:sso`, `competitor:brightline`, `industry:fintech`, `segment:mid_market`) so the UI can show why it was chosen.

## Contrast candidates and warnings

Contrast candidates are the plays used in similar **won** deals, minus the plays the open deal has already done. For each, the counts run over every similar deal that used the play, won or lost: times used, won, lost on a quality reason (`security_compliance`, `feature_gap`, `unresolved_objection`) and lost on another reason. A win despite the pattern is counted like any other deal.

Warnings are base rates over the gated similar deals, written as counts:

- one per objection that is unresolved on the open deal;
- one if the champion has gone silent, or if there is no champion;
- one per competitor named on the open deal.

Each reads like "SSO objection is unresolved: 2 of 3 similar closed deals were lost (1 won despite it)" and lists the deals it counted.

## Plays that address an open objection rank first

Each play in the catalogue lists the objection types it is meant to resolve. The open deal's open objections are those whose status is not `addressed`. Plays are ordered by:

1. the number of the deal's open objections the play addresses (most first);
2. the most problem keys a winning deal using the play shared with this deal;
3. won/used ratio among similar deals;
4. times used among similar deals;
5. net lesson evidence from the lessons bank (a tiebreak);
6. the best Hindsight rank of the deals that used the play;
7. play code.

At most 8 plays are offered to the brief. The brief may recommend at most 3 next steps, each a play from the offered list.

## Plays likely to backfire

Defined in `ranking.py` (`avoid_plays`), run over the same gated similar deals the warnings use:

> A play is "likely to backfire" if it was used in at least 3 similar closed deals (`AVOID_MIN_USED = 3`) and in none of the similar won deals.

Each entry carries honest counts and the deals they come from, for example "Discount offer: used in 4 similar deals, 0 won." with `source_deals` D-008, D-019, D-020, D-021 on the seeded pricing deal. A deal counts a play once. A play the open deal already used is still listed, as a warning about what the team did. Entries are ordered most used first, then by play code.

What the list does:

- The plays are removed from the recommended plays (`Recommendations.avoid` and `Recommendations.plays` never overlap) and from the candidate plays offered to the brief.
- The list goes into the `<evidence>` block of the brief prompt, with an instruction not to recommend those plays. Because it is inside the evidence block, `prompt_hash` stays identical across modes (see below). The prompt version is `brief-v2`.
- It is enforced in code: a next step that names an avoided play is dropped and flagged `avoided_play`. If that leaves no next step, the brief is asked once more, naming the violation.
- It is stored in the brief's content (`avoid`) and evidence, is part of `memory_state`, and is compared across arms (`differs.avoided_play_recommended_by`) and between two briefs of a deal (`added_avoid`, `removed_avoid`, `changed_avoid`).
- Only the `similar` and `hindsight` modes carry a list; `none` and `longctx` never do, so the No history arm and the all-deals arm can recommend such a play, and the comparison says which arms did.

The rule is deliberately strict. A play that won in even one similar deal is not listed, so on the SSO deal Cedarline the discount play (used in 3 similar deals before Larkfield is recorded, 1 won) is ranked low rather than flagged. The counts show use and outcome, not cause.

## Memory backends

`DEAL_MEMORY_BACKEND` is `auto`, `hindsight` or `local` (`config.resolve_memory_backend`). `auto` picks local when there is no Hindsight API key and the URL is Hindsight Cloud; a local Hindsight server URL stays Hindsight. `context.build_context` then builds either the Hindsight wrappers or the local ones, and the rest of the application does not know which it has.

`memory_local.py` implements the same two interfaces as `memory.py` and `lessons.py` over one SQLite file, `memory.db`, in the workspace's own folder (beside `deals.db`):

- Documents are rows of a SQLite FTS5 table (Porter stemming) tagged with the same tags as on Hindsight. A recall query is cut into lowercase word tokens, stop words and one-letter tokens are dropped, each token is quoted and the tokens are joined with OR (at most 48), and results are ranked by bm25. Interactions recall needs every requested tag (like `all_strict`); lessons recall matches any tag.
- Recall is keyword search, not semantic search, so it is weaker than Hindsight's: a deal described in different words may rank lower. The gate, the SQLite re-check and the counts are unchanged and decide what counts.
- Reflect and the playbook are computed deterministically from the stored lessons, as counts (won, lost for a reason a play could change, no verdict on the plays) with the first sentences of the most relevant lessons, and a footer saying they were computed locally. Nothing is invented and no model is called.
- It is always reachable (`healthy` is true, extraction mode `local`), so the Hindsight-unavailable degrade path does not occur. Only an unusable file raises `MemoryUnavailable`, which degrades like an outage.
- `GET /v1/status` reports `hindsight.backend`, with `cloud` false and mode `local` on this backend. SQLite remains the system of record; `memory.db` holds a searchable copy, like a Hindsight bank, and nothing is recalled from it without the SQLite re-check.

## Memory self-check

`quality.memory_quality` takes each closed deal in turn as the target, calls `build_recommendations(..., source="sqlite", exclude_deal_ids={that deal})` so the deal is neither a neighbour nor a source of counts, and compares the result with what really happened. `source="sqlite"` skips the memory and the lessons bank entirely, so the check uses no model and no Hindsight and is not marked degraded. It reports:

- for won deals: how many had a recommendation, and in how many a play from the top 3 was really used;
- for lost deals: how many that had an unresolved objection were warned about it, and in how many a play flagged as backfiring was really used.

On the seed (19 closed deals) the offline run gave: won n=9, covered 7, hit 5; lost n=10, warned 6 of 8 that had an unresolved objection, backfire play flagged in 4. After Larkfield is recorded as lost (20 closed deals): won 9, 7, 5; lost n=11, warned 7 of 9, flagged in 4. These are counts on a small sample, a sanity check and not a benchmark; the response says so in `caveat`.

## Honest counts

- Counts are shown as counts with the number of closed deals in memory beside them, and the UI calls the sample small.
- The demo seed has 19 closed deals (9 won, 10 lost), so a count like "3 of 4" is real but rests on very few deals.
- Briefs, the evidence tab and the comparison state `n`. The comparison adds a caveat: at this size, putting every summary in the prompt also works.

## Degrade paths

- **Hindsight (bank 1) down or unreachable** (the Hindsight backend only; the local backend is always reachable). Retrieval does not fail. Similar deals are found from the signals in SQLite with the same gate, and the recommendations carry a `degraded` message that is stored in the brief and shown in the UI.
- **Lessons bank down or switched off (`DEAL_LESSONS=false`).** Mode `hindsight` falls back to `similar` behaviour: plays are ranked from similar deals only, `lessons_used` stays false, and `degraded` says why.
- **Both down.** Both messages are reported in `degraded`; the brief is still written from SQLite-only evidence.
- **Writes during an outage.** SQLite is always written first. The outbox keeps the row `pending`, stops at the first failure, and retries on the next sync (30 seconds, backing off to 300 seconds while failures continue). A delay can postpone when a deal becomes findable in Hindsight; it cannot let a deleted deal reach a brief, because retrieval re-checks SQLite.
- **Model provider failure.** The brief is stored as failed with an explained error; nothing else changes. A failure on one arm of a comparison stops the comparison and keeps the arms already written.

## Workspace isolation and demo reset

Every workspace owns one SQLite database, one uploads directory, and two Hindsight banks named `<bank>-<workspace id>` (by default `deal-interactions-<id>` and `deal-lessons-<id>`). On the local backend the two banks are tables of one `memory.db` in the workspace's own folder, and each row carries its bank name, so two workspaces never see each other's documents. One team's memory never answers for another.

Creating a demo workspace makes a new workspace with new banks and seeds it with no model calls, so repeated demo runs never inherit the observations an earlier run consolidated. The demo is reset by creating a new demo workspace. Deleting a workspace removes its database, uploads (and, with them, a local `memory.db`) and, best effort, its two Hindsight banks. The original `main` workspace and the active workspace cannot be deleted. Switching workspaces is refused (HTTP 409) while a background job is running.

## Background jobs

Signal extraction, brief writing and the three-arm comparison are jobs in the application process. They are queued, running, completed, failed or cancelled; a job still running when the process dies is marked interrupted and resumed at startup. A resumed comparison does not rewrite an arm that already finished. A person can stop a job. The UI polls `GET /v1/jobs/{id}`.

Separately, the Hindsight sync runs after every write that changes memory and on a timer. It never makes the request that triggered it wait.

## Retrieval modes

| Mode | UI label | What the brief sees |
|---|---|---|
| `none` | No history | This deal's own records only |
| `longctx` | All deals in the prompt | Every closed-deal summary pasted into the prompt, no retrieval, no warnings, ranked plays or plays to avoid |
| `similar` | Similar deals | Recall from the memory, SQLite validation and the gate; plays, warnings and plays to avoid from contrast; no lessons |
| `hindsight` | Hindsight memory | As `similar`, plus lesson evidence as a tiebreak between plays |

The default is `hindsight` (`DEAL_RETRIEVAL_MODE`). The mode keeps its name, and the UI its label, on the local backend too: it then recalls from the local memory. In a comparison every arm uses the same deal, model, system prompt and temperature (0.0); only the evidence block differs (the plays to avoid are inside it), and the shared `prompt_hash` shows it.
