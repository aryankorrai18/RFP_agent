# RFP Memory Assistant project handbook

Updated 2026-09-29. How Hindsight is used, in detail: [HOW_HINDSIGHT_IS_USED.md](HOW_HINDSIGHT_IS_USED.md).

## 1. The project in one minute

RFP Memory Assistant turns an incoming request for proposal into a grounded first draft. It:

1. Reads PDF, Word, Excel, text, or Markdown RFPs.
2. Extracts every requirement and lets a person correct that list.
3. Finds relevant official company facts and approved answers from older proposals.
4. Uses a model to draft one answer per requirement.
5. Rejects invalid citations, highlights weak evidence, and routes real knowledge gaps to an SME.
6. Requires a person to accept, edit, rewrite, or reject every answer.
7. Exports the approved response and learns from review feedback and eventual proposal outcomes.

The central product claim is not that AI knows everything. It is that AI can produce a useful,
reviewable draft while remaining inside an explicit evidence boundary.

## 2. The trust model

SQLite is the source of truth. Hindsight is the memory: it finds past answers and remembers what
happened to them. It is not the primary database and not the drafting model.

The drafting model (Gemini, Claude or Groq) is used for:

- extracting requirements from new RFPs;
- extracting question-and-answer pairs from old proposals;
- drafting answers from the sources it is explicitly given;
- the V4 blind judge, when a person runs it.

Hindsight is used for (two banks per workspace):

- the **answer bank**: a verbatim, searchable copy of approved answers in chunks mode (no LLM), and
  finding the candidate answers for each new requirement;
- the **lessons bank**: outcomes, reviews, debriefs, superseded answers and client preferences,
  stored with Hindsight's concise extraction and observations. Recalled lessons decide which relevant
  candidates are preferred. Reflect writes a client brief, and a mental model keeps a playbook of
  what wins.

Local deterministic code is used for:

- revalidating every Hindsight result against SQLite;
- removing deleted or superseded answers;
- ranking: relevance groups from Hindsight's scores first, then lessons, freshness and client context;
- citation validation, word limits, evidence checks, and export gating;
- storing review actions, ratings, outcomes, debriefs, and the learning journal.

Human approval is used for:

- deciding which extracted historical pairs enter the answer library;
- correcting extracted requirements before drafting;
- accepting or changing every generated answer;
- recording the final win, loss, and section debrief.

## 3. Versions

| Version | What it proves | Main behavior |
|---|---|---|
| V0, frozen | A grounded first draft is useful at all | Fact-sheet-only quick draft; no answer library |
| V1 | Past approved answers improve coverage | Hindsight retrieval, projects, review, and export |
| V2 | Fast human feedback can improve retrieval cheaply | Review reasons and ratings become Hindsight lessons; relevance-grouped ranking; client preferences; lexical evidence checks |
| V3 | Slow business outcomes can improve future work | Win/loss credit, debriefs, freshness, superseding, inspectable memory journal |
| V4 | The effect is real when measured blind | Frozen 200-case offline gate; in the app, a before/after comparison, a blind pairwise judge and a human spot-check |

V2's ranking changed once after a live test: memory could lift an off-topic answer to first place.
Memory now only reorders answers that are about equally relevant ("relevance groups"). On the corpus
answer key, the right answer comes first for 19 of 22 questions, against 17 for plain retrieval and
16 for the first V2 rule, and no off-topic answer comes first ([eval/v2_relevance](../eval/v2_relevance/README.md)).
Results made before the change are kept and labelled "V2 pre-relevance-fix".

V4 is complete for the current hackathon evidence. The 200-case offline gate passes. The Virtusa
demo's six before/after pairs were judged blind in both presentation orders (12 verdicts) and all
six received a human blind spot-check. Lessons won one pair, no lessons won one, and four tied;
the lesson-aware drafts improved mean question coverage and specificity while accuracy remained
5/5. `eval/v4/live_judge_results.json` is the machine-readable record, and the offline report sets
`v4_complete: true`.

Frozen means the baseline prompts, corpus, and expected behavior stay unchanged so later versions
can be compared fairly. Visual styling may change without invalidating the baseline.

## 4. User-facing pages

### Main workspace: http://127.0.0.1:8001/

The root page opens the V1-V4 workspace. It contains Projects, Library, Memory, and a link to the
Quick Draft baseline. The top status area shows the configured model, Hindsight health and mode,
and library synchronization state.

### Projects: http://127.0.0.1:8001/#/projects

Use this page to:

- see every RFP project and its current state;
- upload a new RFP;
- optionally upload that project's approved facts as a JSON fact sheet;
- provide project name, client, and industry context;
- load the fictional sample RFP;
- open an existing project.

Creating a project starts a background requirement-extraction job.

Project states are uploaded, extracting, requirements extracted, drafting, in review, approved,
exported, or failed.

### Project detail: http://127.0.0.1:8001/#/projects/PROJECT_ID

The page changes by workflow stage.

Before drafting:

- review every extracted requirement;
- upload or replace the project-specific JSON fact sheet;
- edit, remove, or add requirements;
- set section, mandatory status, and word limit;
- start Draft all answers.

During drafting:

- see resumable background-job progress;
- see warnings when Hindsight is unavailable.

If requirement extraction fails because Gemini returns a temporary 503, the failed project keeps
its uploaded RFP. Open it and select **Retry extraction**; there is no need to create the project
again. The same screen also lets you attach the fact sheet before retrying.

The project fact sheet must be a .json file shaped like this:

    {
      "company": "Meridian Financial",
      "facts": [
        {
          "id": "FACT-001",
          "topic": "Security",
          "statement": "Customer data is encrypted at rest with AES-256.",
          "valid_from": null,
          "valid_to": null
        }
      ]
    }

Upload it in the **New project** panel, or on a failed/requirements-extracted project before
drafting. If none is uploaded, the app uses the path configured by RFP_FACT_SHEET, which defaults
to data/fact_sheet.json.

During review:

- filter All, To review, Final, Needs SME, Check warnings, or Failed;
- inspect the draft, word count, claims, FACT and ANS citations;
- open the exact past answers retrieved for that requirement;
- inspect outcome score, quality, freshness, context, and ranking reasons on newly generated drafts;
- accept, edit, rewrite, regenerate with instructions, or reject;
- add an optional feedback reason and one-to-five rating.

After every answer is final:

- export Word;
- export Excel when the original RFP was an Excel file;
- record won, lost, no decision, or unknown;
- add an optional loss reason;
- add section-level debrief scores and comments.

Export remains locked until every answer is accepted, edited, or rewritten. A rejection reopens the
answer instead of silently treating it as final.

### Library: http://127.0.0.1:8001/#/library

Use this page to:

- browse all approved reusable answers;
- search by meaning through Hindsight;
- inspect synchronization and track-record state;
- import a past proposal with client, industry, date, result, and loss reason;
- load the three fictional sample proposals;
- open an imported proposal;
- delete an obsolete answer.

Nothing enters the answer library automatically. After extraction, the proposal page requires a
person to keep, edit, or drop every question-and-answer pair. Only confirmed pairs become approved
answers.

The sample proposals are:

- Northwind Bank, finance, won;
- Meridian Health Partners, healthcare, lost on price;
- Cobalt Insurance, insurance, won.

### Past-proposal confirmation: http://127.0.0.1:8001/#/library/PROPOSAL_ID

Use this page to inspect the metadata and extraction status, correct questions and answers, keep or
drop each pair, confirm the final set, or discard an unconfirmed import.

A confirmed proposal cannot be casually discarded because its answers may already influence
drafting. Those answers must instead be explicitly deleted or superseded.

### Memory: http://127.0.0.1:8001/#/memory

This is the audit trail, not a chat-memory page. It shows review, client-preference, outcome,
debrief, and superseding signals with their related answer, project, and timestamp.

The page explains how a human decision becomes a deterministic ranking signal. It also makes the
trust boundary explicit: SQLite owns exact records; Hindsight only finds candidates.

### Quick Draft: http://127.0.0.1:8001/quick

This is the visually updated but behaviorally frozen V0 baseline. It accepts one RFP and an optional
JSON fact sheet, extracts requirements, drafts only from official FACT sources, displays claims and
citations, saves V0 run JSON, and lets the user download the result.

It does not use the answer library, review workflow, outcomes, or Hindsight. Use it to demonstrate
why memory matters: SCIM, admin MFA, implementation timelines, and similar details can remain
Needs SME in V0 but become grounded in the full workspace when an approved past answer exists.

### API documentation

- Swagger UI: http://127.0.0.1:8001/docs
- ReDoc: http://127.0.0.1:8001/redoc
- OpenAPI JSON: http://127.0.0.1:8001/openapi.json
- Basic health: http://127.0.0.1:8001/health

## 5. End-to-end data flow

Historical knowledge flow:

    Past proposal
      -> model extracts Q and A pairs
      -> person keeps, edits, or drops every pair
      -> approved answers are written to SQLite
      -> outbox copies searchable chunks to Hindsight

New RFP flow:

    New RFP
      -> optional project JSON fact sheet (otherwise the workspace's company facts)
      -> model extracts requirements
      -> person corrects the requirement list
      -> Hindsight's answer bank finds candidate answers with relevance scores
      -> SQLite removes invalid, deleted, or superseded candidates
      -> Hindsight's lessons bank recalls what happened to those candidates
      -> ranking: relevance groups first, then lessons, freshness, and client context
      -> model receives official facts plus only the selected approved answers
      -> deterministic checks validate citations, word limits, and evidence
      -> person reviews every answer
      -> accepted text enters the answer library
      -> reviews, outcome, and debrief become lessons in Hindsight and improve later ranking

If Hindsight is unavailable, SQLite remains intact, writes remain pending in the outbox, and
drafting falls back to the official fact sheet with a visible warning. Failed synchronization uses
exponential backoff instead of repeatedly hammering the service.

## 6. Ranking and learning

For each requirement:

1. Hindsight's answer bank returns up to 12 candidates with relevance scores; SQLite drops dead ones.
2. Hindsight's lessons bank is asked what it remembers about those candidates. Positive and negative
   lessons give each candidate a lesson factor from 0.25 to 2.5.
3. Candidates are grouped by relevance: the first group is every candidate scoring at least 1% of the
   best one on Hindsight's `final` score, the next group the same over what's left.
4. Inside a group: Hindsight's order x lesson factor x freshness (730-day half-life) x client and
   industry context. A group never overtakes an earlier one, so memory can't lift an off-topic answer.
5. The top 3 are offered to the drafting model. Lessons themselves never enter the prompt.

With Hindsight's lessons bank switched off or unreachable, step 2 uses the local SQLite signals
instead (smoothed quality from accepts, edits, rewrites, rejections, ratings, and outcome credit),
with a visible warning.

Only approved ANS sources actually cited by a draft receive review credit. A retrieved answer that
the model ignored is not rewarded or punished. A review of a draft that cited no past answer becomes
a lesson about the question and client instead, and "too vague", "too long", "too short" and "tone"
also become drafting instructions for that client.

Outcome credit is intentionally small:

- won: positive credit;
- lost for technical fit or response quality: negative credit;
- lost for price, budget, incumbent, or relationship: neutral credit.

This avoids pretending that every commercial loss means an answer was bad.

## 7. Grounding and safety behavior

- Every generated claim must cite an allowed FACT or ANS source.
- A citation outside the supplied source set is invalid.
- Unsupported requirements become Needs SME rather than invented answers.
- Current official facts outrank contradictory historical answers.
- The lexical evidence checker marks supported, partial, unsupported, or unverifiable claims.
- Word limits are checked.
- Deleted answers are excluded immediately from SQLite-backed retrieval even if their Hindsight
  deletion is still pending.
- Superseded answers remain auditable but are no longer live retrieval candidates.
- Uploads are extension-checked, size-limited, content-length-limited, and stored under generated
  names rather than user-controlled paths.
- API errors use structured codes and messages.
- Sensitive API responses are marked no-store and browser safety headers are enabled.

The lexical evidence check is deliberately conservative and is not equivalent to a semantic
entailment model.

## 8. Storage and important files

| Location | Purpose |
|---|---|
| data/workspaces.json | Workspace registry: which workspaces exist and which is open |
| data/workspaces/<id> | A workspace's database, uploads and company facts |
| data/rfp.db | SQLite source of truth for the original workspace, main |
| data/uploads | Stored uploaded source documents |
| data/fact_sheet.json | Fictional official company facts |
| runs | V0 Quick Draft result JSON |
| samples | Fictional sample RFP, questionnaire, and past proposals |
| baselines/v0 | Frozen V0 results and verdict |
| baselines/v1 | Frozen plain-retrieval V1 runs and analysis |
| backend/v1 | Library, retrieval, ranking, jobs, learning, outcomes, export, and API |
| backend/v4 | Deterministic evaluation implementation |
| eval/v4 | Frozen protocol, offline/live runners, live judge and learning-loop results |
| eval/v2_relevance | The relevance-fix evaluation: frozen Hindsight input, rules, results |
| samples/corpus_v2 | The demo company's 8 past proposals, 2 new RFPs, answer key and demo seed |
| frontend/app.html | Main V1-V4 workspace |
| frontend/index.html | Quick Draft V0 page |
| tests | Offline test suite using fake model and memory clients |

The environment file, runtime database, uploads, V0 runs, virtual environments, and caches are
ignored by source control.

## 9. Current prepared demo state

On 2026-09-29 there are three workspaces:

| Workspace | What's in it | Use |
|---|---|---|
| `larkspur-data-clean-demo` | The demo seed only: 8 past proposals (4 won, 4 lost), 45 answers, their lessons | **The stage demo.** No test has touched it |
| `larkspur-data-demo` | The demo seed plus the live learning-loop tests: the Ashford RFP, a "too vague" preference for Ashford, a rejected SCIM answer | Evidence for `eval/v4/live_learning_loop.json` |
| `main` | The data from before workspaces: 11 past proposals, 67 answers, 6 projects including the frozen V1 baseline runs, and the judged before/after comparisons (Ashford, Meridian) | Baselines and V4 judge results. Its fact sheet is read-only |

Switch with the workspace button in the top bar. Switching is refused while a background job is running.

## 10. Recommended five-to-seven-minute demo

> **Being rewritten.** This script predates workspaces and the Hindsight lessons bank. The next demo
> runs in `larkspur-data-clean-demo` (section 9), and its strongest moment is "reject as too vague,
> then the next draft is specific". Until the rewrite, use section 9 and HOW_HINDSIGHT_IS_USED.md.

### Opening, 20 seconds

Say:

“RFP teams repeatedly answer the same questions, but ordinary generation either ignores historical
knowledge or invents details. This system drafts from controlled evidence, learns from human
decisions and proposal outcomes, and keeps every recommendation inspectable.”

### Step 1: system status, 20 seconds

Open the main workspace and point to the configured model, Hindsight healthy in chunks mode, the
synchronized library count, and outcome-aware retrieval.

Say that chunks mode means Hindsight is indexing approved text without running another Hindsight
LLM.

### Step 2: Library, 45 seconds

Open Library. Show the 21 approved answers, three confirmed sample proposals, won and lost
metadata, one answer's track record, and a semantic search for “SCIM provisioning”.

Explain that every result is still checked against SQLite before reaching the model.

### Step 3: Project 4, two minutes

Open Harborview Credit Union RFP. Show:

- progress and review counts;
- requirement 3.2 for SCIM, citing ANS-0012;
- requirement 3.3 for encryption, combining official facts and a past answer;
- requirement 2.2 or 7.1 as Needs SME;
- the retrieved-answer disclosure and its ranking explanation;
- the fixed fact-sheet panel.

This contrast is the core demo: known history fills legitimate gaps, while unknown facts are still
refused.

### Step 4: Human review, one minute

On an unfinished grounded answer, show the optional reason and rating and explain Accept, Edit,
Rewrite, Regenerate, and Reject. Either accept one answer or only describe the action if you do not
want to change demo data.

Explain that only cited library sources receive the feedback signal.

### Step 5: Memory, 40 seconds

Open Memory and show the review-signal timeline. Explain that it is an auditable business-learning
ledger rather than hidden model memory.

### Step 6: Quick Draft comparison, 40 seconds

Open Quick Draft. Explain that it uses the same evidence-first presentation but only the fact sheet:
it has no historical-answer memory and is preserved as the comparison baseline.

Use SCIM and administrator MFA as the V0 versus V1 story.

### Step 7: evaluation proof, 30 seconds

In a terminal run:

    .\.venv\Scripts\python.exe -m eval.v4.run_offline

Point out:

- 200 cases;
- outcome top-three accuracy 1.0;
- plain top-three accuracy 0.0 on this deliberately competitive synthetic corpus;
- required improvement 0.20, observed improvement 1.0;
- deterministic gate passed;
- the live Virtusa blind judge ran in both presentation orders (1 better with lessons, 1 without,
  4 ties), and all 6 pairs received a human blind spot-check;
- the combined report records `v4_complete: true`.

### Closing, 20 seconds

Say:

“The model writes, Hindsight finds candidates, SQLite owns truth, deterministic checks enforce the
boundary, and people make every final decision.”

## 11. Demo modes and token cost

### Prepared-state demo: recommended

Browse Library, Projects, Memory, and Quick Draft without starting a new run. This uses no Gemini
generation calls. Library searches recall from both Hindsight banks. The answer bank uses no
Hindsight LLM (chunks mode); new lessons, the client brief and the playbook use Hindsight credits.

Running the offline V4 evaluation and pytest also uses no provider tokens.

### Fresh full-workflow demo

This intentionally spends provider calls:

- loading three new sample proposals makes roughly three extraction calls;
- creating the sample project makes one requirement-extraction call;
- drafting its 20 requirements makes roughly 20 drafting calls;
- running Quick Draft on the 20-question sample makes one extraction call and roughly 20 drafting
  calls.

Use drafting concurrency 1 if the Gemini free tier is rate-limited:

    $env:RFP_DRAFT_CONCURRENCY = "1"

Do not run both the full workspace draft and Quick Draft during the same presentation unless that
comparison is worth the extra cost and waiting time.

## 12. Starting and stopping

From the parent directory:

    powershell -ExecutionPolicy Bypass -File rfp-v0\run.ps1

Then open:

    http://127.0.0.1:8001

The current session may already have the server running. Starting a second copy on the same port
will be refused. Use Ctrl+C in the server terminal to stop a foreground instance.

For another port:

    powershell -ExecutionPolicy Bypass -File rfp-v0\run.ps1 -Port 8002

After changing Python code, restart the server. Model keys saved to the environment file are
re-read without a restart. Hindsight URL, key, bank, database path, and upload directory are used
to construct the startup context and therefore require a restart.

## 13. Configuration

Required for live drafting:

- GEMINI_API_KEY, GROQ_API_KEY or ANTHROPIC_API_KEY.

Required for Hindsight Cloud:

- RFP_HINDSIGHT_URL;
- RFP_HINDSIGHT_API_KEY;
- optionally RFP_HINDSIGHT_BANK.

Important controls:

- RFP_LLM_PROVIDER selects auto, gemini, groq, or anthropic;
- RFP_MODEL selects the model;
- RFP_DRAFT_CONCURRENCY controls parallel calls;
- RFP_RETRIEVAL_MODE selects none, plain, outcome, or hindsight (the default);
- RFP_RETRIEVAL_RELEVANCE selects gated (default: memory reorders only relevant answers) or rank
  (the V2 pre-relevance-fix rule), and RFP_RETRIEVAL_RELEVANCE_MIN_SHARE sets the group share (0.01);
- RFP_HINDSIGHT_LESSONS_BANK names the lessons bank, and RFP_LESSONS=false switches lessons off;
- RFP_WORKSPACES_FILE points at the workspace registry;
- RFP_RETRIEVAL_TOP_K defaults to 3;
- RFP_RETRIEVAL_FRESHNESS_HALF_LIFE_DAYS defaults to 730;
- RFP_EVIDENCE_CHECK enables the local lexical check;
- RFP_MAX_REQUIREMENTS defaults to 150;
- RFP_MAX_UPLOAD_MB defaults to 20;
- RFP_MAX_DOCUMENT_CHARS defaults to 400000.

Never show the environment file or provider keys during a demo.

## 14. Testing

### Full offline suite: recommended after every change

From the rfp-v0 directory:

    .\.venv\Scripts\python.exe -m pytest -q

Expected current result: 235 passing tests. The suite replaces Gemini, Groq, Anthropic, and Hindsight
with fakes, so it uses no provider tokens.

Coverage includes parsing, upload limits, structured errors, frozen prompts and samples, grounding,
citations, all three model adapters and their error explanations, library and lesson synchronization,
outage fallback, projects, resumable jobs, reviews, exports, outcomes, debriefs, ranking and the
relevance rule, the before/after comparison, the blind judge and spot-check, workspaces and the demo
seed, company facts, evidence checks, V4 evaluation, and security headers.

### Dependency and syntax checks

    .\.venv\Scripts\python.exe -m pip check
    .\.venv\Scripts\python.exe -m compileall -q backend eval

### V4 offline gate

    .\.venv\Scripts\python.exe -m eval.v4.run_offline

This is deterministic and token-free.

### Health checks

    Invoke-RestMethod http://127.0.0.1:8001/health
    Invoke-RestMethod http://127.0.0.1:8001/v1/status

Healthy demo expectations:

- status is ok;
- llm_credentials is env;
- hindsight.healthy is true;
- hindsight.extraction_mode is chunks (the answer bank);
- lessons.pending is 0 and lessons.last_error is empty;
- hindsight.baseline_ok is true;
- library.pending_sync is 0.

### Manual UI smoke checklist

At desktop width and a narrow mobile width:

1. Projects, Library, Memory, and Quick Draft all open.
2. Navigation indicates the current page.
3. Theme toggle works.
4. No page has horizontal scrolling.
5. Library search returns answers.
6. Project 4 shows all 20 requirements.
7. FACT citations scroll to the fact.
8. ANS citations open the retrieved-answer detail.
9. Review filters change the visible cards.
10. Export stays locked while answers remain unfinished.
11. No red error alert appears.

### Optional low-token live judge

This makes one Gemini judge call with no SDK retries:

    .\.venv\Scripts\python.exe -m eval.v4.run_live_smoke --judge-only

### Optional full one-case live smoke

This makes one Hindsight recall, two Gemini draft calls, and one Gemini judge call:

    .\.venv\Scripts\python.exe -m eval.v4.run_live_smoke

The live smoke is only a wiring check. It does not complete V4.

## 15. API summary

V0:

- POST /v0/draft
- GET /v0/fact-sheet
- GET /v0/runs
- GET /v0/runs/{run_id}

Library and memory:

- POST /v1/library
- GET /v1/library/proposals
- GET /v1/library/proposals/{proposal_id}
- POST /v1/library/proposals/{proposal_id}/confirm
- POST /v1/library/proposals/{proposal_id}/discard
- GET /v1/library/answers
- DELETE /v1/library/answers/{code}
- POST /v1/library/answers/{code}/supersede
- GET /v1/memory/learned
- GET /v1/memory/lessons
- GET /v1/memory/playbook, POST /v1/memory/playbook/refresh
- GET /v1/memory/learning-curve
- POST /v1/sync

Projects:

- POST /v1/projects
- GET /v1/projects
- GET /v1/projects/{project_id}
- POST /v1/projects/{project_id}/retry
- PUT /v1/projects/{project_id}/fact-sheet
- PUT /v1/projects/{project_id}/requirements
- POST /v1/projects/{project_id}/draft
- GET /v1/jobs/{job_id}
- POST /v1/requirements/{requirement_id}/review
- POST /v1/requirements/{requirement_id}/regenerate
- GET /v1/projects/{project_id}/export
- PUT /v1/projects/{project_id}/outcome
- POST /v1/projects/{project_id}/debrief
- GET, POST /v1/projects/{project_id}/brief (client brief, Hindsight Reflect)
- GET, POST /v1/projects/{project_id}/comparison (before/after)
- GET, POST /v1/projects/{project_id}/comparison/judge (blind judge; GET shows the cost first)
- GET, POST /v1/projects/{project_id}/comparison/spot-check
- POST /v1/jobs/{job_id}/cancel

Workspaces and company:

- GET /v1/workspaces, POST /v1/workspaces (kind demo or company), POST /v1/workspaces/{id}/activate
- DELETE /v1/workspaces/{id} (its database, uploads, fact sheet, and both Hindsight banks; refused for "main" or the active workspace)
- GET /v1/workspace
- GET, PUT /v1/company
- GET, PUT /v1/models

System:

- GET /health
- GET /v1/status

## 16. Honest limitations

The app is hackathon-demo ready, not public-production ready.

Current intentional limits:

- single user and no login; workspaces separate data and memory but are not a security boundary;
- SQLite and one in-process background worker;
- no multi-instance coordination;
- no production audit-log service, tracing, or cost dashboard;
- English only;
- duplicate questions are not merged;
- Excel is model-read rather than mapped by a customer-specific schema;
- external model and Hindsight availability can affect fresh runs;
- lexical evidence checking is not full semantic entailment;
- V4's offline gate uses a deliberately easy synthetic corpus; the live corpus is small (two RFPs, 37
  questions) and fictional;
- the live blind judge ran and found no clear draft-quality gain from lessons (31 of 37 identical or
  tied); the human spot-check remains open;
- one rejection changes the first answer only when a comparable relevant answer exists.

Before public launch, add authentication, organization isolation, managed Postgres, durable queues,
migrations, backups, centralized observability, rate limits, privacy and retention controls,
deployment TLS and CSP hardening, formal accessibility testing, and representative real-world
evaluation.

## 17. Best judge questions and answers

### Why use Hindsight if SQLite is the database?

Hindsight is the memory: its answer bank finds relevant past answers by meaning, and its lessons bank
remembers what happened to them (won, lost, rejected, too vague) and says so when asked about a
candidate or a client. SQLite remains authoritative for the exact text, live or deleted state,
reviews, scores, outcomes, and audit trail.

### Does Hindsight use another LLM here?

The answer bank doesn't: chunks mode, observations off, approved text stored verbatim. The lessons
bank does, on Hindsight's side: concise extraction turns each lesson into recallable facts,
observations consolidate them, Reflect writes the client brief, and a mental model keeps the
playbook. None of that text reaches the drafting prompt.

### Can memory make the wrong answer win?

Not across topics. Memory reorders only answers that Hindsight scores as about equally relevant; an
off-topic answer can't be lifted above a relevant one. We added that rule after a live test broke
without it, and measured it: 0 off-topic first answers on the corpus, also after a simulated
rejection of every first answer.

### What prevents hallucinations?

The model receives only current facts and selected approved answers, must cite claims, is checked
against its allowed source IDs, receives a lexical evidence check, and cannot bypass human review.
Unsupported gaps are shown as Needs SME.

### What actually learns?

The base model is not fine-tuned. Reviews, outcomes and debriefs become Hindsight lessons that change
which approved answers are offered; review reasons such as "too vague" also become short drafting
instructions for that client. Live: a rejection reached Hindsight within seconds, and the next draft
for that client changed from a generic sentence to the specific, won answer.

### Why is the baseline frozen?

It prevents moving the goalposts. V1-V4 improvements can be compared against the same prompt,
corpus, and expected V0/V1 behavior.

### Is V4 finished?

Yes for the current hackathon evidence: the offline gate passes, the live blind judge has run in
both presentation orders, all six pairs received a human blind spot-check, and the combined report
records `v4_complete: true`. The report still discloses that this was one live project rather than
the protocol's three repeated live runs.
