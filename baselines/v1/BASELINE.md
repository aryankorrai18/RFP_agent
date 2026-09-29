# V1 Baseline: plain retrieval (frozen 2026-09-28)

This is the number V2's outcome memory must beat (hypothesis H2). V1 drafts from the fact sheet plus the **top 3 past answers found by plain similarity search** (Hindsight, chunks mode, no Hindsight LLM, no Reflect).

## Verdict

| Acceptance check (V1 design §11) | Result |
|---|---|
| 1. A matching past answer is among the top 3 retrieved | ✅ **14/14 in every run, and ranked first in all 42 cases** |
| 2. Drafts cite `ANS-…` as well as `FACT-…`, and check 1 passes | ✅ 7, 12 and 10 drafts cite past answers; **0 invalid citations**; every `ANS` cited was the expected one (no irrelevant past answer cited) |
| 3. A question with no fact and no past answer still goes to Needs SME | ✅ references (2.2), Symitar (6.2) and pricing (7.1): **Needs SME in 3/3 runs** |
| 4. Every requirement can be reviewed, and export is a Word file in RFP order | ⏳ covered by the offline tests; **the live UI walkthrough is still to do** |
| 5. An accepted answer is retrievable on the next project | ⏳ covered by the offline tests; **the live UI walkthrough is still to do** |

Also verified: where a past answer contradicts the fact sheet, **the fact sheet won in 6/6 cases**: 2.1 says 240 employees, not the outdated "around 200", and 3.3 says TLS 1.2, not the outdated "TLS 1.1".

## Frozen configuration

| Item | Value |
|---|---|
| Prompt version | **v1.0**. Drafting rules `c6df7ac57d50b9fb`, pair extraction `1c74cba59c166142` + `38b8283a6f6fa9f5`, draft instruction `f9f7f91b33221d87`, requirement extraction `d6384bb3bfb55e43` (unchanged from V0). Guarded by `tests/test_prompts_and_data.py::test_frozen_v1_prompts_are_unchanged` |
| Provider / model | Gemini, `gemini-3.5-flash-lite` (free tier); extraction `low`, drafting `medium`; draft concurrency 2 |
| Memory | Hindsight Cloud, `https://api.hindsight.vectorize.io`, bank `rfp-library`, `retain_extraction_mode: chunks` (verified to be honoured, see V1 design §13.1) |
| Retrieval | `budget="mid"`, tag `kind:rfp_answer` (`all_strict`), top k = 3, **no score threshold**, every hit re-checked against SQLite |
| SDKs | hindsight-client 0.10.1, sqlalchemy 2.0.54, google-genai 2.25.0, fastapi 0.141.1 |
| Harness | [`run_baseline.py`](run_baseline.py) drives the HTTP API exactly as the UI does; [`analyze.py`](analyze.py) scores the runs. Projects are drafted and **never reviewed**, so every run sees the same library |

## Frozen corpus

| File | Parsed-text SHA-256 (first 16 hex) | Chars |
|---|---|---|
| `samples/sample_rfp.docx` (20 requirements; same as V0) | `5b547309c22f853f` | 2,329 |
| `samples/past_northwind_bank_2025.docx` (6 pairs, won) | `30d6ce7c688283f7` | 1,608 |
| `samples/past_meridian_health_2026.docx` (5 pairs, lost on price) | `d498716ab038b641` | 1,137 |
| `samples/past_cobalt_insurance_2025.docx` (4 pairs, won) | `5915ba787509b28e` | 880 |
| `data/fact_sheet.json` (byte hash, same as V0) | `06eab6e0b9401955` | — |

**Library:** 15 answers, `ANS-0001`–`ANS-0015`. Every extracted pair was confirmed as *kept*. All 15 answers were verified word for word against their source documents.

## Results

| Run file | Grounded | Needs SME | Flagged | Failed | Drafts citing `ANS` | Answer words | Time |
|---|---|---|---|---|---|---|---|
| V0 baseline (`../v0/sample_rfp.prompt-v0.3.json`) | 13 | 7 | 0 | 0 | — | 398 | 52 s |
| `run-20260928-151803.json` | 16 | 4 | 0 | 0 | 7 | 504 | 148 s |
| `run-20260928-152100.json` | 17 | 3 | 0 | 0 | 12 | 530 | 177 s |
| `run-20260928-152310.json` | 16 | 4 | 0 | 0 | 10 | 446 | 131 s |

Time covers extraction, retrieval and drafting. Token usage wasn't recorded in V1 runs.

### Per question (G = grounded, S = Needs SME; `+ANS` = cited a past answer)

| Ref | Topic | V0 | Run 1 | Run 2 | Run 3 | Past answer ranked first |
|---|---|---|---|---|---|---|
| 2.1 | Company overview | G | G | G | G | ANS-0010 (conflicts: ~200 people) |
| 2.2 | Customer references | S | S | S | S | none (true gap) |
| 3.1 | SSO with Entra ID | G | G | G +ANS | G +ANS | ANS-0011 |
| **3.2** | **SCIM provisioning** | **S** | **G +ANS** | **G +ANS** | **G +ANS** | ANS-0012 |
| 3.3 | Encryption | G | G +ANS | G +ANS | G +ANS | ANS-0013 (ANS-0003 second: conflicts, TLS 1.1) |
| 3.4 | SOC 2 | G | G | G | G | none |
| 3.5 | Pen test: how often, by whom | G | G | G | G | ANS-0005 (names the firm), **never used** |
| 4.1 | Data residency | G | G +ANS | G +ANS | G +ANS | ANS-0008 |
| 4.2 | Backup / DR | G | G | G +ANS | G +ANS | ANS-0004 |
| 4.3 | Contract end | G | G | G +ANS | G | ANS-0009 |
| **5.1** | **Uptime remedies** | **S** | **S** | **G +ANS** | **S** | ANS-0001 (credit percentages) |
| 5.2 | Support hours | G | G +ANS | G +ANS | G +ANS | ANS-0002 |
| **5.3** | **Implementation timeline** | **S** | **G +ANS** | **G +ANS** | **G +ANS** | ANS-0014 |
| 6.1 | APIs and connectors | G | G +ANS | G +ANS | G +ANS | ANS-0015 |
| 6.2 | Symitar real-time events | S | S | S | S | none (true gap) |
| 7.1 | Pricing | S | S | S | S | none (true gap) |
| 8.1 | RBAC | G | G | G | G | none |
| 8.2 | Audit logs 12 months | G | G | G +ANS | G +ANS | ANS-0007 |
| **8.3** | **Admin MFA** | **S** | **G +ANS** | **G +ANS** | **G +ANS** | ANS-0006 |
| 8.4 | Breach notification | G | G | G | G | none |

## Known weaknesses, kept as evaluation cases

These continue the V0 list (EC-01 to EC-06 in [../v0/BASELINE.md](../v0/BASELINE.md)).

| ID | Case | Evidence | What V2/V4 must measure |
|---|---|---|---|
| **EC-07** | **Retrieved detail dropped when the fact sheet partly covers the topic** | 3.5 asks "by whom". ANS-0005 (Ironbridge Security) was ranked first in 3/3 runs, yet every draft said "an independent third party" from FACT-009 | Use of retrieved relevant detail per answer (recall of sources) |
| **EC-08** | **The SME decision on commercial terms varies** | 5.1: ANS-0001 has the credit tiers (10% / 25%). Grounded from it in 1 run, Needs SME asking for those same percentages in 2. This extends EC-03 | Stability of Needs SME across runs; whether a won past answer should settle commercial terms (a V2 outcome-memory question) |
| **EC-09** | **Library use varies from run to run** | Drafts citing a past answer: 7, 12, 10. 3.1, 4.2, 4.3 and 8.2 cited one in some runs only | Citation rate across ≥3 runs; report mean and spread, never a single run |
| **EC-10** | **Uncalibrated scores fill top 3 with irrelevant answers** | Relevant hits score ~0.99 and irrelevant ones ~0.00005, but the top 3 always has 3 entries. The drafter ignored every irrelevant one in these runs | Irrelevant-citation rate (0 here); calibrate a threshold in V4 on labelled data |

## Limits of this baseline (read before V2)

- **Retrieval is saturated on this corpus.** The right answer ranked first in 42/42 cases, and only one question (3.3) had two competing past answers. On this corpus, V2's outcome memory **can't beat V1 at retrieval**, only at how answers are used. Before V2 is judged, the synthetic corpus needs **competing answers**: won vs lost versions of the same answer, current vs outdated, different clients and industries. That's what H2 is actually about.
- The gains over V0 come from three questions V0 had to send to an SME (3.2, 5.3, 8.3), plus 5.1 in one run. That's real value, but it's a small sample.
- One RFP, one model, a free tier, and three runs. Treat differences of ±1 question as noise.
