# RFP Templates for the RFP Memory Assistant

Use these templates to create your own test documents. The app accepts **`.md`, `.txt`, `.docx`, `.pdf` and `.xlsx`**, so you can fill in a template below, save it as its own `.md` file, and upload it directly.

There are three templates, one for each thing the app reads:

| # | Template | Uploaded where | Required? |
|---|---|---|---|
| A | **RFP** (the buyer's questions) | Projects → New project | **Yes** |
| B | **Past proposal** (questions *with* your old answers) | Library → Add a past proposal | Optional, but without it drafts come from the fact sheet only |
| C | **Fact sheet** (your company's official facts) | Edit `rfp-v0/data/fact_sheet.json` | **Yes** (a sample one ships with the app) |

> Use **fictional or non-confidential** content only. Drafts go to Google Gemini on a free key, and approved answers go to Hindsight Cloud.

---

## A. RFP template (the buyer's document)

Copy everything between the lines into a new file, e.g. `my_rfp.md`, and replace the `<...>` parts.

---

```markdown
# <Buyer organisation name>: Request for Proposal

**RFP title:** <e.g. Customer Data Platform>
**Issued:** <YYYY-MM-DD>
**Submission deadline:** <YYYY-MM-DD, time and time zone>
**Contact:** <procurement contact name and email>

## 1. Instructions to vendors

Answer every question in sections 2 onwards. Keep each answer within its stated word limit.
Questions marked MANDATORY must be answered; an unanswered mandatory question disqualifies the response.

## 2. Company information

2.1 Provide an overview of your company, including year founded, number of employees and office locations. (Maximum 150 words)
2.2 Provide <number> customer references from <industry> organisations, with contact details.

## 3. Security

3.1 MANDATORY: Describe how your solution supports single sign-on with our identity provider (<IdP name>) via <SAML 2.0 / OIDC>. (Maximum 100 words)
3.2 MANDATORY: Does your solution support automatic user provisioning and de-provisioning via SCIM 2.0?
3.3 Describe how customer data is encrypted at rest and in transit.
3.4 Provide your most recent SOC 2 Type II report or describe your current certifications.
3.5 How often is your platform penetration tested, and by whom?

## 4. Data and hosting

4.1 MANDATORY: Where will our data be stored? Confirm whether data can be kept within <region>.
4.2 Describe your backup and disaster-recovery arrangements, including RPO and RTO.
4.3 What happens to our data when the contract ends?

## 5. Service and support

5.1 State your uptime commitment and what remedies apply if it is not met.
5.2 Describe your support hours and channels, including for critical incidents.
5.3 Describe your implementation approach and a typical timeline. (Maximum 120 words)

## 6. Integration

6.1 Describe the APIs and prebuilt connectors your platform provides.
6.2 <A question about integrating with one of the buyer's own systems>

## 7. Commercials

7.1 Provide a <number>-year pricing proposal, including implementation fees.

## 8. Compliance matrix

| Requirement | Compliant (Y/N) | Comments |
|---|---|---|
| 8.1 The solution must provide role-based access control. | | |
| 8.2 Audit logs must be retained for at least <N> months. | | |
| 8.3 Multi-factor authentication must be enforced for all administrative access. | | |
| 8.4 The vendor must notify us of any confirmed security breach within <N> hours. | | |
```

---

### How to write questions so they extract well

| Do | Why |
|---|---|
| Number every question (`3.1`, `SEC-04`, …) | The number becomes the requirement's reference and orders the Word export |
| Put each question on its own line | One line gives one requirement |
| Write `MANDATORY:` at the start where it applies | It is picked up as the "Mandatory" flag |
| Write `(Maximum N words)` at the end where it applies | It is picked up as the word limit, and drafts over it are flagged |
| Group questions under section headings | The heading becomes the requirement's section |
| Keep instructions, deadlines and contacts in their own sections, not as numbered questions | These should not become requirements (the extractor is told to skip them) |
| For Excel RFPs: one question per row, with a header row such as `ID | Question | Response` | Export writes each answer into the response column next to its question |

**Limits:** up to 150 requirements per RFP, 20 MB per file, about 400,000 characters of text. A scanned PDF (image only) works but costs more model tokens.

**Tip for good tests:** include some questions your fact sheet and library *can* answer, some they *partly* answer, and at least 2–3 they *can't* (like pricing or references). A correct run marks those **Needs SME** instead of inventing answers.

---

## B. Past proposal template (for the answer library)

This is a proposal you (the vendor) submitted before, with **both the question and your answer**. The app extracts each question-and-answer pair, you confirm them, and they become `ANS-…` library entries.

```markdown
# <Vendor name>: Response to <Buyer name> RFP (<Month Year>)

## Security

**3.1 Question:** Which single sign-on standards and identity providers do you support?
**Response:** <Your answer as actually submitted.>

**3.2 Question:** Describe how user accounts are created and removed automatically.
**Response:** <Your answer.>

## Service levels

**SL-1 Question:** What availability do you guarantee, and what happens if you miss it?
**Response:** <Your answer.>

<...repeat for every question you answered...>
```

When you upload it, the form also asks for:

| Field | Values | Used for |
|---|---|---|
| Client | e.g. `Northwind Bank` | Shown next to each answer; context for ranking |
| Industry | e.g. `finance`, `healthcare` | Context for ranking |
| Submitted | date, `YYYY-MM-DD` | Freshness: older answers count less over time (2-year half-life) |
| Result | `won`, `lost`, `no_decision`, `unknown` | Outcome learning |
| Loss reason | e.g. `price`, `technical fit`, `response quality`, `incumbent` | `technical fit` or `response quality` lowers the answers' credit. `price`, `budget`, `incumbent` or `relationship` doesn't, because it isn't the answer's fault |

**Tips:**
- Write answers exactly as submitted. Pairs are copied verbatim.
- To test outcome memory, make **competing versions**: the same question answered differently in a won proposal and a lost one, or an old answer that contradicts the current fact sheet (the fact sheet must win).

---

## C. Fact sheet template (`rfp-v0/data/fact_sheet.json`)

These are the company's **current official facts**. Drafts may cite them as `FACT-…`, and when a past answer disagrees with the fact sheet, the fact sheet wins.

```json
{
  "company": "<Your company name>",
  "facts": [
    {"id": "FACT-001", "topic": "Company", "statement": "<One self-contained, checkable fact.>", "valid_from": "2026-01-01", "valid_to": null},
    {"id": "FACT-002", "topic": "Security / SSO", "statement": "<Another fact.>", "valid_from": "2025-11-01", "valid_to": null},
    {"id": "FACT-099", "topic": "Old fact", "statement": "<An outdated fact.>", "valid_from": "2024-01-01", "valid_to": "2025-12-31"}
  ]
}
```

- **One fact per statement**, written as a full sentence. It must make sense on its own.
- **IDs** must be unique: `FACT-001`, `FACT-002`, …
- **`valid_to`** in the past means the fact is expired and is never offered to drafts. Use `null` for current facts.
- Leave out anything you don't want the assistant to claim. Missing topics become **Needs SME**, which is the correct behaviour.
- Back up the sample file before replacing it, and restart the server after editing.

---

## Everything the app needs as input

### To run at all
| Input | Where | Required |
|---|---|---|
| Gemini API key (`GEMINI_API_KEY`), or `ANTHROPIC_API_KEY` | `rfp-v0/.env` | **Yes** |
| Hindsight Cloud URL and key (`RFP_HINDSIGHT_URL`, `RFP_HINDSIGHT_API_KEY`) | `rfp-v0/.env` | Recommended. Without it, drafts use the fact sheet only |
| Fact sheet | `rfp-v0/data/fact_sheet.json` | **Yes** (sample included) |

### Per project (drafting an RFP)
| Input | Required |
|---|---|
| The RFP file (template A) | **Yes** |
| Project name | No (defaults to the file name) |
| Client, industry | No, but they improve ranking context |
| Your review of each draft: Accept / Edit / Rewrite / Reject, with an optional rating and reason tags | Needed to finish and export; teaches the library |
| Outcome once known: `won` / `lost` / `no_decision`, loss reason, optional section debrief scores | Optional; teaches outcome memory |

### Per library entry
| Input | Required |
|---|---|
| A past proposal file (template B) | **Yes** |
| Client, industry, submitted date, result, loss reason | Optional, but they drive freshness and outcome learning |
| Your keep / edit / drop decision on every extracted pair | **Yes**: nothing enters the library without it |
