# RFP Memory Assistant

Upload a buyer's RFP and get a draft answer for every question. Every statement must cite an official company fact (`FACT-…`) or an approved past answer (`ANS-…`). Anything the sources can't support is marked **Needs SME** (needs an expert), never presented as fact. A person reviews every answer.

It learns from what happens next. Which answers reviewers accept or reject, and which proposals are won or lost, become lessons in [Hindsight](https://hindsight.vectorize.io). The lessons decide which past answers are offered for the next RFP. Three rules keep this honest:

1. **Hindsight is memory, not just search.** One bank holds the approved answers; a second holds what happened to them.
2. **SQLite is the exact system of record.** Hindsight holds searchable copies and learned experience, and every answer it returns is checked against SQLite.
3. **Memory can't override relevance.** Lessons reorder answers that are about equally relevant; they never lift an off-topic answer to the top.

- **How Hindsight is used:** [docs/HOW_HINDSIGHT_IS_USED.md](docs/HOW_HINDSIGHT_IS_USED.md)
- **Pages, data flow, demo and judge Q&A:** [docs/PROJECT_HANDBOOK.md](docs/PROJECT_HANDBOOK.md)
- **Evidence:** [eval/v2_relevance](eval/v2_relevance/README.md) (ranking), [eval/v4](eval/v4/README.md) (live judge, learning-loop timings)

Two ways to start, from the workspace button in the top bar: **Try the demo** (a fictional vendor with 8 won and lost past proposals, loaded with no model calls) or **Start from scratch** (your own company facts, an empty memory that fills as you work). See [Two ways to start](#two-ways-to-start-demo-or-your-own-company).

### Why the names say v0, v1 and v4

The project was built in stages, each measured against a frozen baseline before the next was added. The names stayed:

- `rfp-v0/` is the project folder: the whole product lives here, not just V0.
- `backend/v1/` is the application (library, projects, memory, ranking, API). V2 and V3 were built inside it.
- `backend/v4/`, `eval/v4/` and `eval/v2_relevance/` are evaluation. `baselines/v0/` and `baselines/v1/` are frozen baselines.
- `/quick` (API `/v0/…`) is the V0 fact-sheet-only baseline page, kept for comparison. The app itself is at `/` (API `/v1/…`).

### How it was built

| | What it adds | Hypothesis | Page |
|---|---|---|---|
| **V0** (frozen) | Drafts from the fact sheet only. No database. | H1: are drafts a useful start? Passed provisionally, see [baselines/v0/BASELINE.md](baselines/v0/BASELINE.md). | http://127.0.0.1:8001/quick |
| **V1** | An **answer library**: import past proposals, confirm their Q&A pairs, and draft from the fact sheet plus the top 3 past answers found by plain retrieval (Hindsight). Projects, review, export, and accepted answers flow back into the library. | The **plain-retrieval baseline** that V2's outcome memory must beat. | http://127.0.0.1:8001 |
| **V2** | Outcome memory: review feedback and results become lessons in a Hindsight lessons bank; ranking by relevance groups, then lessons; client preferences; client briefs (Reflect) and a playbook (mental model); lexical evidence checks. | Better sources should rise without another model call. | http://127.0.0.1:8001 |
| **V3** | Win/loss outcomes, section debriefs, freshness decay, explicit superseding, and an inspectable memory journal. | Slow business outcomes improve future retrieval without hiding causality. | http://127.0.0.1:8001 |
| **V4** | Frozen 200-case offline gate; in the app, a before/after comparison, a blind pairwise judge and a human spot-check. | Is the effect real, measured blind? | python -m eval.v4.run_offline · project page |

- Design: [docs/HOW_HINDSIGHT_IS_USED.md](docs/HOW_HINDSIGHT_IS_USED.md) · [docs/V2_V4_TECHNICAL_DESIGN.md](docs/V2_V4_TECHNICAL_DESIGN.md) · [baselines/v0](baselines/v0/BASELINE.md) · [baselines/v1](baselines/v1/BASELINE.md)
- Product: [../PRD_RFP_MEMORY_ASSISTANT.md](../PRD_RFP_MEMORY_ASSISTANT.md)

## Setup

Python 3.12 (3.10+ works).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the keys (below)
```

### 1. A model key (V0 and V1)

| Provider | Key variable | Default model | Where to get a key |
|---|---|---|---|
| **Gemini** (free tier works) | `GEMINI_API_KEY` | `gemini-3.5-flash-lite` | aistudio.google.com → Get API key |
| **Groq** (free tier works) | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | console.groq.com → API Keys |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-5` | console.anthropic.com → API Keys |

`RFP_LLM_PROVIDER=auto` (the default) uses Anthropic if its key is set, then Gemini, then Groq. The server re-reads `.env` on every request, so a saved key works without a restart.

**Gemini free tier:**
- Per-minute request limits are low. Drafting runs 2 questions at a time on Gemini and retries automatically when rate-limited. If you still see rate-limit errors, set `RFP_DRAFT_CONCURRENCY=1`.
- Google's free-tier terms allow it to use your inputs to improve its products. **Use only the fictional samples or non-confidential documents on a free key.**

**Groq free tier:**
- Groq's limits are per minute and count tokens, so a long RFP can hit them even at 2 questions at a time. The app waits and retries; if it still fails, set `RFP_DRAFT_CONCURRENCY=1` or pick a model with a higher limit in the model button (the list comes from your key).
- Groq can't read scanned PDFs (use Gemini or Claude for those). Text-based PDFs, Word and Excel files work.
- The frozen V0/V1 baselines were measured on Gemini; results on Groq's models will differ.
- As with any free key, **use only fictional or non-confidential documents.**

**When a model call fails:**
- Every failure is explained in plain words, with what to do: quota used up, too many requests per minute, model not available to your key, key missing or rejected, provider unreachable or overloaded, request too large, file the provider can't read, answer declined, unusable output. The raw provider message stays one click away under **Error details**.
- When switching model is the fix (quota used up, retired model), the note has a **Choose another model** button, and the model button and status bar turn red until a call succeeds or you switch.
- A drafting job, comparison or quick draft **stops after the first error every remaining call would repeat** (quota used up, model not available, key rejected) instead of spending the rest of its calls on it. The questions it skipped show as "no draft yet"; after switching model, **Redraft** does them.
- Gemini's free tier has a per-minute and a per-day quota. The app tells them apart: a per-minute limit is retried, a used-up daily quota is not.

### 2. Hindsight (V1's answer search)

V1 stores everything in SQLite (`data/rfp.db`), which is the source of truth. Hindsight holds a search copy of the approved answers and is used **only to find** them: no Hindsight LLM, no Reflect. The app creates the bank in `chunks` mode, and every search result is re-checked against SQLite, so a deleted answer never reaches a draft.

**Option A: Hindsight Cloud (default).** Put your key in `.env`:

```
RFP_HINDSIGHT_URL=https://api.hindsight.vectorize.io
RFP_HINDSIGHT_API_KEY=<your key>
```

Unlike the model key, Hindsight settings are read once at startup: **restart the server after changing them.** The status bar at the top of the V1 page shows whether Hindsight is reachable and which extraction mode the bank really uses. It should say `chunks`. If it shows a warning, the bank was created with LLM extraction, which V1's baseline must not use: point `RFP_HINDSIGHT_BANK` at a new bank name. Like the Gemini free tier, send only fictional or non-confidential content to a cloud service.

**Option B: local Hindsight (no key, no Docker).** Install it into its own venv, then run it bound to localhost:

```powershell
python -m venv .hindsight-venv
.\.hindsight-venv\Scripts\python -m pip install hindsight-api==0.10.1
$env:PYTHONUTF8 = "1"; $env:HINDSIGHT_API_LLM_PROVIDER = "none"
.\.hindsight-venv\Scripts\hindsight-api --host 127.0.0.1 --port 8888
```

Then set `RFP_HINDSIGHT_URL=http://127.0.0.1:8888` and leave `RFP_HINDSIGHT_API_KEY` empty. `PYTHONUTF8=1` is required on Windows. The embedded Postgres keeps its data in `%USERPROFILE%\.pg0`.

**If Hindsight is down,** V1 keeps working: answers are saved to SQLite and marked *syncing*
(retried in the background with exponential backoff and at startup), and drafts fall back to the
fact sheet only, with a warning on each affected answer.

## Run

The **New project** panel accepts the RFP plus an optional project-specific .json fact sheet. You
can also attach or replace that file on a failed or requirements-extracted project before drafting.
Without one, the project uses data/fact_sheet.json (or RFP_FACT_SHEET). If Gemini returns a
temporary 503 during extraction, reopen the failed project and select **Retry extraction**; the
original RFP remains stored.

Easiest on Windows (works from any folder; creates the venv on first run):

```powershell
powershell -ExecutionPolicy Bypass -File rfp-v0\run.ps1
```

Or manually, from inside `rfp-v0` with the venv activated:

```bash
uvicorn backend.main:app --port 8001
```

Port 8001, because 8000 is often taken. Don't use `--reload` on Windows; it hangs.

### Two ways to start: demo or your own company

The engine is the same either way; what differs is the starting memory. Each **workspace** has its own database, uploads, company facts and pair of Hindsight banks, so a demo and a real company never mix. Switch or create one with the workspace button in the top bar.

- **Try the demo** (New workspace → Open the demo workspace). Loads a fictional vendor, Larkspur Data, with 8 past proposals from 2023 to 2026: 4 won and 4 lost, with competing answers to the same questions. The 45 answers are taken exactly as they appear in each document (`samples/corpus_v2/demo_seed.json`), so **no model calls** are made. Hindsight then stores the answers and turns the won/lost results into lessons, which uses Hindsight credits. The Ashford RFP and the Meadowgate questionnaire are ready to draft, with an answer key in `samples/corpus_v2/answer_key.json`.
- **Start from scratch** (New workspace → your company's name). An empty workspace:
  1. **Company**: enter your official facts (or paste them one per line). Drafts cite them as FACT-001, FACT-002…, and a fact outranks any past answer.
  2. **Library** (optional): import old proposals with their won/lost result.
  3. **Projects**: upload your first RFP. Anything your facts and library can't support comes back as "needs an expert".
  4. Each answer you approve joins the library, and each outcome you record becomes a Hindsight lesson, so later RFPs start from what worked.

The data that existed before workspaces is the first workspace, **main** ("Larkspur Data (sample history)"), left where it was. Its fact sheet is the bundled `data/fact_sheet.json`, which the frozen baselines were measured against, so the Company page shows it read-only. The registry is `data/workspaces.json` and new workspaces live in `data/workspaces/<id>/` (both git-ignored). Switching is refused while a background job is running. Any workspace but `main` can be permanently deleted (its database, uploads, fact sheet and both Hindsight banks) from the workspace switcher, once it isn't the one currently open.

### V1-V3 walkthrough (http://127.0.0.1:8001)

1. **Library → Load the sample library** imports three fictional past proposals (Northwind Bank, Meridian Health, Cobalt Insurance). Open each, check the extracted question-and-answer pairs (edit or drop any), and click **Add … answers to the library**.
2. **Projects → Try the sample RFP.** Check the extracted requirements (fix, remove or add), then **Draft all answers**. A progress bar tracks the background job. If the server restarts mid-way, the job resumes where it stopped.
3. **Review** each answer: optionally select a reason and 1-5 rating, then **Accept**, **Edit**, **Rewrite**, **Regenerate...** or **Reject**. Retrieval rows explain relevance, quality, freshness and client/industry context.
4. When every answer is final, **Export Word** (or **Export Excel** for a spreadsheet RFP, with answers written next to their questions).
5. Record the win/loss and optional section debrief. Accepted answers and their exact track record influence the next project.
6. Open **Memory** to inspect what the system learned and why. The **learning curve** there is counted from real reviews (accepted as written, light edit, heavy edit, rejected, plus needs-an-expert and unsupported-claim rates), one point per reviewed project; the counts sit beside the score and the mode and model are shown so you can tell whether conditions matched.
7. On a project, **Does memory change the draft?** drafts every question twice with the same model, prompt and facts, once with plain search and once with Hindsight's lessons, and shows both answers, which past answers each cited (from won or lost proposals) and what Hindsight remembered. It costs two model calls per question, and those drafts are kept apart from review. Drafts are made at temperature 0 on Gemini and Groq so the two states differ only in what memory offered (Claude ignores it; even at 0 a provider may not be perfectly repeatable, so treat one run as evidence, not proof). If you stop a comparison, running it again finishes it: drafts already made under the same model and temperature are carried over, not paid for twice. Running again after a comparison finished drafts everything fresh, because memory may have changed. For the corpus RFPs, `python samples/corpus_v2/score_comparison.py <project_id>` scores each state against the answer key.
8. Under the comparison, **V4 · blind judge** asks a second model which draft is better, without saying which used lessons, in both presentation orders (see `eval/v4/README.md`). It shows the number of calls before you start. **Human spot-check** lets you pick blind too, then reveals which was which and how often you agreed with the judge.

## Test

```bash
pytest
```

The tests run offline. The AI and Hindsight are replaced by fakes, so they cost nothing and need no key.

The frozen V4 retrieval evaluation is also offline. Run: python -m eval.v4.run_offline

It evaluates 200 synthetic held-out cases across none, plain, outcome, and three ablations. It
also reads the recorded live evidence in `eval/v4/live_judge_results.json`. The current Virtusa
demo run includes a two-order blind judge and six human spot-checks, so the report records
`v4_complete: true`.

## Live acceptance tests (cost a small amount of API usage)

**V0** (fact sheet only; frozen, see [baselines/v0/BASELINE.md](baselines/v0/BASELINE.md)): on `/quick`, run `samples/sample_rfp.docx`. SCIM (3.2), customer references (2.2), Symitar (6.2), pricing (7.1) and MFA (8.3) must be **Needs SME**, because the fact sheet deliberately has no facts for them.

**V1** (design §11): with the three sample proposals confirmed, run `samples/sample_rfp.docx` as a project. V1 passes when:

1. Questions with a matching past answer show it among the top 3 retrieved, for example SCIM from Northwind, penetration testing and admin MFA from Meridian, and SLA credits from Cobalt.
2. Drafts cite `ANS-…` as well as `FACT-…`, and no citation is invalid.
3. Where a past answer contradicts the fact sheet (Northwind's headcount, Cobalt's "TLS 1.1"), the draft follows the fact sheet.
4. Questions with no fact and no past answer (references, Symitar, pricing) are still **Needs SME**.
5. Every requirement can be reviewed, and the Word export lists the answers in RFP order.
6. An accepted answer is retrievable in the next project.

Checks 1–3 are automated by the harness, which drives the same API as the UI. Results are in [baselines/v1/BASELINE.md](baselines/v1/BASELINE.md).

```bash
python baselines/v1/run_baseline.py library     # import the 3 samples (refused if they're already imported)
python baselines/v1/run_baseline.py confirm     # confirm every extracted pair, then sync to Hindsight
python baselines/v1/run_baseline.py run 3       # 3 projects, drafted but never reviewed
python baselines/v1/analyze.py                  # score them against the checks above
```

Checks 4–6 are the UI walkthrough above.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` / `GROQ_API_KEY` / `ANTHROPIC_API_KEY` | — | The key for the provider you use |
| `RFP_LLM_PROVIDER` | `auto` | `auto`, `gemini`, `groq` or `anthropic` |
| `RFP_MODEL` | `gemini-3.5-flash-lite` / `llama-3.3-70b-versatile` / `claude-opus-5` | Model for all call types. The **model button in the top bar** overrides it without editing `.env` (saved in `data/model_choice.json`; "Reset to .env" removes the override) |
| `RFP_EXTRACTION_EFFORT` / `RFP_DRAFT_EFFORT` | `low` / `medium` | Effort per call type (Gemini: thinking level) |
| `RFP_DRAFT_CONCURRENCY` | `2` (Gemini, Groq) / `8` (Anthropic) | Parallel drafting calls |
| `RFP_MAX_REQUIREMENTS` | `150` | Above this, the request is rejected (never truncated) |
| `RFP_MAX_UPLOAD_MB` | `20` | Upload size limit |
| `RFP_MAX_DOCUMENT_CHARS` | `400000` | Extracted-text limit |
| `RFP_FACT_SHEET` | `data/fact_sheet.json` | Default fact sheet |
| `RFP_RUNS_DIR` | `runs` | Where V0 results are saved |
| `RFP_HINDSIGHT_URL` | `http://127.0.0.1:8888` | Hindsight server (Cloud: `https://api.hindsight.vectorize.io`) |
| `RFP_HINDSIGHT_API_KEY` | — | Hindsight Cloud key |
| `RFP_HINDSIGHT_BANK` | `rfp-library` | Memory bank for approved answers |
| `RFP_RETRIEVAL_TOP_K` | `3` | Past answers offered per question |
| `RFP_RETRIEVAL_MODE` | `outcome` | `none`, frozen-V1 `plain`, or V2/V3 `outcome` |
| `RFP_RETRIEVAL_FRESHNESS_HALF_LIFE_DAYS` | `730` | Freshness decay used by outcome ranking |
| `RFP_EVIDENCE_CHECK` | `true` | Token-free lexical evidence-support check |
| `RFP_DB_PATH` / `RFP_UPLOADS_DIR` | `data/rfp.db` / `data/uploads` | V1 data (git-ignored). Once workspaces exist, the active workspace's paths, fact sheet and Hindsight banks are used instead; these values only seed workspace "main" on first run |
| `RFP_WORKSPACES_FILE` | `data/workspaces.json` | Workspace registry (which workspaces exist, which is open) |

## Layout

```
backend/        main.py (routes, V0 pipeline endpoints) · service.py · parser.py · llm.py (Claude) · llm_gemini.py (Gemini) · llm_groq.py (Groq) · provider_errors.py (what a failed call means, stop-early, model health)
                grounding.py · prompts.py (v0.3 frozen, v1.0) · schemas.py · config.py
backend/v1/     db.py (SQLite models) · memory.py (Hindsight) · retrieval.py · sync.py (outbox) · jobs.py (background jobs)
                ranking.py · evidence.py · learning.py · outcomes.py · library.py · projects.py · export.py · api.py
backend/v4/     deterministic frozen-corpus evaluation
eval/v4/        protocol manifest and token-free evaluation runner
frontend/       app.html (V1-V3 product) · index.html (V0 quick draft). No build step.
data/           fact_sheet.json (fictional company, deliberately incomplete) · rfp.db + uploads/ (created on first run)
samples/        sample RFP, questionnaire, 3 past proposals, and make_samples.py (won't overwrite without --force)
baselines/v0/   frozen V0 results
tests/          offline tests with fake AI and fake Hindsight
```

## Known limits

- Single user, no login (Beta adds accounts and Postgres).
- Background jobs run inside the app process: one server instance only.
- Duplicate questions aren't merged. Excel sheets are read by the model, not by column mapping.
- The default evidence checker is lexical and conservative; a live entailment judge is still required for the final V4 gate.
- Hindsight remains a candidate generator/search index. SQLite is the source of truth and stores all exact learning signals.
- The V4 offline gate validates retrieval mechanics on synthetic data, not real-world draft quality.
- English only.
