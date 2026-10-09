# Hindsight agents

Three small apps that share one idea: agents get better when they remember what happened.

| Folder | What it is | Port |
|---|---|---|
| `rfp-v0/` | RFP Memory Assistant: grounded answers to RFPs and security questionnaires, learning from won and lost proposals | 8001 |
| `deal-intelligence/` | Deal Intelligence: pre-call deal briefs grounded in the deal's own notes and in what memory recalls from similar won and lost deals | 8002 |
| `agent-hub/` | Agent Hub: a chat that picks the agent and does the work (attach a file, it creates the deal or project, runs the steps and shows the result), asking before anything spends model calls | 8003 |

## Run everything

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 start     # start the three apps
powershell -ExecutionPolicy Bypass -File .\run.ps1 status    # what is running
powershell -ExecutionPolicy Bypass -File .\run.ps1 stop      # stop only what this script started
```

Open the hub at <http://127.0.0.1:8003>. Each app has its own `.env.example`; copy it to `.env` and add your own keys (never commit `.env`).

## Learn how it works

