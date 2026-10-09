# Agent Hub

One front door for the agents. Open the hub and chat: it picks the agent and does the work in the chat. Attach an RFP and it creates the project, drafts the answers and hands you the document; ask for a deal brief and it reads the deal and writes it. Each agent stays its own app with its own data and memory; the hub talks to them over their HTTP APIs, server-side (no keys or agent URLs in the browser).

## How it works

- **The chatbot layer (`planner.py`, `llm.py`).** With a Gemini key in `agent-hub/.env`, a small language model reads every message that is not a plain yes/no, together with the recent conversation and where the chat is working, and picks one of the hub's tools with its arguments (list or count deals, brief, ask about a deal, ask across the pipeline or the library, draft a follow-up, record an outcome, create a deal, choose a workspace...), or replies in words, or asks one clarifying question. It never answers questions about your data itself: the answer comes from the agents' checked, cited endpoints through the tool it picked. It does not decide spending either: every tool that makes an agent use model calls still asks first, in code. One understanding call per message, shown as a counter in the header, with a per-chat cap (`HUB_ROUTER_CAP`, default 300). Without a key, or when the model fails or the cap is reached, the hub tells you once and falls back to the phrase matching below, so it keeps working.

- `agents.json` is the registry: every agent with its URL, description, examples and routing keywords. Two are live (Deal Intelligence, RFP Memory Assistant); the rest are listed as planned.
- `router.py` and `intents.py` understand a message without a model: which agent, which deal ("Cedarline", "D-004", token match with typos), won or lost and why (eight loss reasons with plain-English synonyms), yes/no/cancel. Free, instant, testable.
- `engine.py` holds one state machine per conversation plus an append-only event log in SQLite (`data/hub.db`, git-ignored; uploads go to `data/uploads/`, 20 MB and `.pdf .docx .xlsx .txt .md .eml .csv` only). Long steps run as background tasks that poll the agent's job and append progress events. Running job ids are stored, so after a hub restart the job is re-attached instead of started again.
- `flows/deal.py`, `flows/rfp.py` are the two workflows; `clients.py` is the only code that talks to the agents.

### Workspaces

The first time a chat needs an agent that has more than one workspace, the hub asks which one to use (with deal counts to help) and then carries on with what you asked; it asks once per agent per chat, and never when you already chose, saved a default, or the agent has only one. Each agent keeps its work in workspaces (a company's deals, a company's RFP library). A chat chooses one per agent ("show workspaces", "use the Halcyon local demo workspace", "use the Globex workspace for RFPs", "use the default workspace") and the hub sends it with every request as `X-Workspace`. Nothing is switched inside the agents, so their own screens are unaffected and two chats can work in two workspaces at the same time; the choice only lives in that chat and shows as a chip at the top. "Remember these workspaces as Accenture" saves the chat's choices as the start of every new chat (and shows the company name in the chip); "forget the saved workspaces" undoes it. When a deal is not in the chat's workspace but is in another, the hub says where and offers to use it. Creating and deleting workspaces stays in the agents' own screens.

### The cost rule

Every step that spends model calls first posts a confirmation with the exact cost ("This will use 12 model calls (one per requirement)") and two buttons. Each question or draft asks first too; "Yes, and don't ask again for questions in this chat" lets later questions in that one chat run without asking (each still counts as one call, and a new chat asks again). A confirmation fires at most once, expires when the conversation moves on, and the calls you approve are a budget the engine will not exceed. Free steps (reading a deal, creating one, accepting grounded answers, exporting) run without asking; steps that change memory at no model cost (recording an outcome) still ask. The hub never redrafts on its own.

### What the chat does

- Lists and counts are free: "What are those deals?", "list the open deals",  "How many deals do we have?", "how many open deals", "how many deals did we lose to price?" are answered from the deal list with no model call (a condition the list cannot test, like "stalled", goes to the model after asking). If the chat's deals workspace is empty, the hub says so and points at the workspaces that have deals instead of spending a call.
- Company-wide questions: "Why do we lose fintech deals?" goes to Deal Intelligence, which counts across every deal in the chat's workspace and answers with citations and the counts it used; "What have we told clients about SOC 2 before?" goes to the RFP assistant, which answers from the company facts and approved past answers; a question that needs both asks both (one call each, you are told first). Each agent answers only from its own data; a named deal still gets the single-deal answer.
- Deals: ask a question about a deal ("What did the buyer say about SSO on Cedarline?", one model call, answered from its notes and similar closed deals, with citations; a deal named in the sentence or the one the chat is already about), draft a follow-up email or call agenda for the brief's top next step (one model call, nothing is sent), brief a deal (reads its emails and notes first when needed), create a deal from attached files, add files or a note to a deal, record won/lost with a reason (the deal's plays are sent explicitly so they are never wiped), offer to brief the open deals again.
- RFPs: attach a document, check the active workspace and company facts, extract the requirements (1 call), draft all answers (1 call per requirement), accept the grounded ones, offer the Word (and, for an Excel RFP, Excel) export through a download link, record how the bid ended.

### What the chat does not do

It does not edit requirements, write SME answers, regenerate or redraft answers, accept flagged answers, create, switch or delete workspaces, or change models. It tells you and links to the agent's own screen for those. It also stops (with a message) if the active workspace changes while a step is waiting for your yes.

## Backups

From the repository folder:

```powershell
.\backup.ps1 create                     # a dated copy of every app's data folder under backups\, checked straight away
.\backup.ps1 list
.\backup.ps1 verify backups\<folder>     # re-reads every file against its SHA-256 and runs PRAGMA integrity_check on every database
.\run.ps1 stop; .\backup.ps1 restore backups\<folder>
```

A backup covers databases, uploads, workspace registries and fact sheets. SQLite files are copied with SQLite's online backup, so a backup taken while the apps run is consistent. `restore` refuses while an app is running, refuses a backup that does not verify, and moves what it replaces to a `.before-restore-<time>` folder instead of deleting it. Not included: `.env` files (your keys; keep them in a password manager) and Hindsight Cloud (a copy of what is in these databases). A backup holds password hashes and chats, so treat the `backups` folder like the data itself. The admin page's Security check shows whether each specialist can be read without the hub.

## Sign-in and companies

For one person on their own computer the hub is open. Once the first account exists (`HUB_AUTH=auto`, the default) sign-in is required; `HUB_AUTH=on` always requires it and `off` never does.

- **A company** is a name plus the workspaces it may use in each agent (its own deals, its own RFP library). **A user** belongs to exactly one company. The company decides which workspaces the hub will ever send to an agent: the person's words cannot widen it. A company with one workspace per agent is simply placed in it; with several, the hub asks which, among those only. Another company's workspaces are not listed, cannot be chosen, and requests for them are refused in the client before they reach an agent. Saving "default workspaces" is switched off for company users.
- **The admin page (`/admin`).** Open <http://127.0.0.1:8003/admin> (there is also an Admin link in the chat header). Before any account exists it shows a first-run form, available only from the computer the hub runs on: name your company, tick the workspaces it may use in each agent, and make your administrator account. That switches sign-in on and signs you in. After that, an administrator can add and edit companies and their workspaces, add people (a role of member or administrator, with a generated or typed password), change a person's company or role, reset a password, disable or re-enable an account, and read the activity log (who changed what, never a password). Changing a person's company or role, resetting a password and disabling all end their open sessions at once. The only active administrator cannot be demoted or disabled, so nobody can lock themselves out. Workspace ids are checked against what the agents actually have. **Each company gets its own workspaces.** When you add a company, "Create its own workspaces" is ticked by default: the hub asks Deal Intelligence and the RFP Memory Assistant to create one new, empty workspace each (named "<Company> - My deals" and "<Company> - RFPs") and gives them to the company, so its data is separate from every other company's. Only this click does it (never a chat), it is written to the activity log, an agent that is not running stops it before anything is made, and a failure part-way undoes what was created. Creating a workspace makes an agent switch to it, so the hub switches the agent back to the workspace it had open. **Deleting a company keeps its workspaces and data by default.** The delete panel lists what the company has and offers "Also permanently delete its data"; that needs ticking and typing the company name, and the server only deletes exactly the workspaces the panel listed. A workspace another company also uses, an agent's original (main) workspace, and one an agent currently has open are always kept, with the reason shown. A company that still has people cannot be deleted. **Administrators are not a company.** Every administrator belongs to the built-in "Hub administrators" team, which can never be given data, renamed or deleted; making someone an administrator moves them there, and a member must belong to a customer company. An administrator's chat answers usage questions only (no model, no company data), and the refusal is in the backend: the team has no workspaces, so the agent clients refuse every data request. Older hubs' admins are moved into the team at start. **Usage.** The admin page has a Usage card, and an administrator can ask the chat ("how many companies are working here, how many tokens did they use, how much storage"). Both show, per company: people, chats and messages, model calls (the hub's own plus each agent's) with input and output tokens, and the storage its workspaces and uploads take, plus the workspaces no company has been given, so the totals add up. It is read from records, never from a model: each agent records every model call it makes in a `model_usage` table and reports it with the workspace's storage at `GET /v1/usage`, and the hub keeps its own ledger of the calls it makes to read a message. Counting starts when the ledgers were added (the card says the date); earlier calls were not recorded. A non-administrator asking in the chat is told it is for administrators. Calls and tokens only: no money figure is shown, so nothing goes stale when prices change. **Reviewing drafts in the chat.** After drafting, "Review them one by one" (or "review the answers") shows each unfinished answer with its draft, the evidence it cites and the checks it failed, with Accept, Edit (type the final text, then pick why it changed), Reject (with a reason) and Write the answer (for ones sent to an expert). Each decision is one call to the RFP assistant's own review route, so it scores the review and learns from it exactly as its own screens do; typed text is taken word for word (never read by a model or as a command), and every decision is in the activity log with who made it. **Feeding the memory.** Everything a company's memory needs can be given in the chat: "here is our fact sheet" (or "Add these company facts:" and one per line), a past proposal with "past proposal we won for Acme Bank (banking), submitted 2025-03-10" (or "lost on technical fit"), and "Add our sales plays:" with one play per line or a file (.json, .csv, .txt, .md). Each is read by plain code (the proposal by one model call), shown, and saved only when the person clicks; facts are added to the existing ones, the read-only sample workspace is refused, and every save is written to the activity log.
- **Accounts are made by administrators**, not by self sign-up. The same things can be done from a terminal (add `--admin` to make an administrator). From `agent-hub`:

```powershell
.\admin.ps1 workspaces            # the workspace ids each agent has
.\admin.ps1 company add "Accenture" --deals accenture-deals --rfp accenture-rfp
.\admin.ps1 user add dana@accenture.com --company Accenture --name Dana   # asks for the password twice
.\admin.ps1 user list
.\admin.ps1 company set Accenture --deals a,b   # change what a company may use
```
  Passwords are asked for, never taken from the command line (use `--password-env VAR` to script it), are at least 10 characters, and are stored as salted scrypt hashes.
- **Sessions** are a random token in an HttpOnly, SameSite=Lax cookie (Secure with `HUB_COOKIE_SECURE=1` or https); only the token's SHA-256 is stored, so a copy of the database cannot be replayed. Signing out, a new password or disabling the account ends the session. Five failed sign-ins for an email and address lock it out for 15 minutes; an unknown email and a wrong password look identical.
- **Chats belong to their owner.** Someone else's chat (or one made before sign-in existed) answers 404, exactly like a missing one. A request from another website (a different `Origin`) is refused.
- **Closing the agents to clients.** The agents have no sign-in of their own. Put a secret in each agent's `.env` (`DEAL_SERVICE_TOKEN`, `RFP_SERVICE_TOKEN`) and the same one in the hub's (`HUB_DEAL_SERVICE_TOKEN`, `HUB_RFP_SERVICE_TOKEN`): every `/v1` call without it is refused, so only the hub can reach the agent. Their own screens stop working while it is set, which is what you want on a shared machine; keep them off the network.

Not covered: hosting itself (https, a reverse proxy, backups), password reset by email, and rate limiting beyond the sign-in lockout.

## API

All JSON, no CORS:

| Route | |
|---|---|
| `GET /api/auth/me`, `POST /api/auth/login` `{email, password}`, `POST /api/auth/logout` | whether sign-in is on and who is signed in; sets and clears the session cookie |
| `GET /`, `GET /health`, `GET /api/agents` | the page, a health check, the agents with a live status |
| `POST /api/chat` | the old stateless keyword demo (`{message}`) |
| `POST /api/conversations` | `{id}` |
| `GET /api/conversations/{id}` | `{id, state, events, next, busy}`; 404 when unknown |
| `POST /api/conversations/{id}/messages` | multipart `text`, repeated `files`; 422 with neither, 413 over 20 MB; returns `{accepted, n}` at once |
| `GET /api/conversations/{id}/events?after=n` | `{events, next, busy}`; poll it |
| `POST /api/conversations/{id}/actions` | `{action_id}` (a button click) -> `{accepted}` |
| `GET /api/conversations/{id}/downloads/{token}` | the exported file, proxied with its `Content-Disposition` |

An event is `{n, role, kind: text|cards|progress|error, text?, cards?, actions?, progress?, files?, created_at}`; a card is generic (`title, subtitle, tone, sections[heading, text, bullets, chips, rows], links`) so the page is not tied to a flow.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Open <http://127.0.0.1:8003>. Start the agents from their own folders (`deal-intelligence\run.ps1` on 8002, `rfp-v0\run.ps1` on 8001); if one is stopped the chat says how to start it.

## Add an agent

Add an entry to `agents.json` (`status: live`, `url`, `health_path`, `open_path`). Chat flows for a new agent need a client and a flow module; routing and the agents strip need nothing else.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Offline: `tests/fakes.py` has small fake Deal and RFP apps (real response shapes, scriptable failures) mounted through `httpx.ASGITransport`, and every request they receive is logged. The tests assert that nothing spends model calls without a confirmation and that no workspace-changing route or `PUT /v1/models` is ever called.

## Not in this version

Shared sign-in or hosting for other people, a model fallback for messages that match no flow, and embedding the agents' screens. Each agent keeps its own workspace and memory.
