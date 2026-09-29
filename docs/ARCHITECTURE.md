# Architecture

## Runtime flow

1. The browser calls the FastAPI `/v1` API.
2. Uploaded documents are validated, stored in the active workspace, and parsed.
3. The selected model extracts the complete bidder-response requirements.
4. Retrieval searches approved answers in Hindsight and validates every result against SQLite.
5. Outcome, review, freshness, client, and industry signals rank relevant candidates.
6. The model drafts from the active fact sheet and selected approved answers.
7. Deterministic grounding and evidence checks add citations, warnings, or a company-input gate.
8. Human reviews and proposal outcomes are written to SQLite and synchronized as lessons.

## Trust boundaries

- **SQLite** is the exact system of record for projects, answers, reviews, outcomes, and audit data.
- **Hindsight answer bank** provides semantic retrieval over approved answer text.
- **Hindsight lessons bank** stores experience derived from reviews, outcomes, and debriefs.
- **Model providers** extract and draft, but cannot approve responses or directly change memory.
- **Company facts** override contradictory historical answers.
- **Human review** is mandatory before a response becomes final.

## Main modules

| Module | Responsibility |
|---|---|
| `backend/main.py` | Application lifecycle, workspaces, model selection, security headers |
| `backend/v1/api.py` | HTTP contracts and response models |
| `backend/v1/projects.py` | Requirement extraction, drafting, review, and project state |
| `backend/v1/library.py` | Past-proposal import and approved answer library |
| `backend/v1/retrieval.py` | Hindsight recall and SQLite validation |
| `backend/v1/ranking.py` | Relevance-aware outcome ranking |
| `backend/v1/learning.py` | Review signals and client preferences |
| `backend/v1/outcomes.py` | Win/loss credit, debriefs, and superseding |
| `backend/v1/lessons.py` | Hindsight lesson storage and recall |
| `backend/v1/experiment.py` | Before/after memory comparison |
| `backend/v1/judge.py` | Blind model judge and human spot-check |
| `frontend/app.html` | Production browser interface |

## Workspace isolation

Every workspace owns:

- one SQLite database;
- one uploads directory;
- one company fact sheet;
- one Hindsight answer bank;
- one Hindsight lessons bank.

Switching workspaces is refused while a background job is active. Runtime workspace data is ignored by Git.

## Background jobs

Requirement extraction, proposal extraction, drafting, comparisons, and judge runs are resumable jobs. A blocking provider failure stops calls that would repeat the same error, preserving completed work and tokens.
