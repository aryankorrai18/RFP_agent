"""Small in-memory stand-ins for the Deal Intelligence and RFP Memory Assistant APIs.

They return the real response shapes (taken from the two apps' routers and docs/API.md) and can be
scripted to fail. Every request is logged, whether or not the route exists, so tests can assert what
the hub did and did not call. They are mounted through httpx.ASGITransport: no sockets, no models."""

from __future__ import annotations

import copy
import re
from contextvars import ContextVar

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

SPENDING = (
    ("POST", re.compile(r"^/v1/projects$")),
    ("POST", re.compile(r"^/v1/projects/\d+/draft$")),
    ("POST", re.compile(r"^/v1/deals/\d+/signals$")),
    ("POST", re.compile(r"^/v1/deals/\d+/brief$")),
    ("POST", re.compile(r"^/v1/deals/\d+/ask$")),
    ("POST", re.compile(r"^/v1/deals/\d+/followup$")),
    ("POST", re.compile(r"^/v1/portfolio/ask$")),
    ("POST", re.compile(r"^/v1/library/ask$")),
)
FORBIDDEN = (  # nothing here may ever be called by the hub
    ("POST", re.compile(r"^/v1/workspaces")),
    ("DELETE", re.compile(r"^/v1/workspaces")),
    ("PUT", re.compile(r"^/v1/workspace")),
    ("PUT", re.compile(r"^/v1/models")),
    ("PUT", re.compile(r"^/v1/projects/\d+/requirements$")),
    ("POST", re.compile(r"^/v1/requirements/\d+/regenerate$")),
    ("PUT", re.compile(r"^/v1/projects/\d+/fact-sheet$")),
    ("DELETE", re.compile(r".*")),
)


class FakeError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status, self.code, self.message = status, code, message


class FakeAgent:
    """Shared plumbing: request log, an on/off switch for 'the agent is not running', error envelope."""

    def __init__(self) -> None:
        self.down = False
        self.calls: list[tuple[str, str]] = []
        self.model_calls = 0
        self.app = FastAPI()

        self.workspace_headers: list[tuple[str, str | None]] = []  # (path, X-Workspace) of every request
        self.service_tokens: list[str | None] = []  # X-Service-Token of every request
        self.require_token: str | None = None  # when set, /v1 calls without it are refused (like DEAL_SERVICE_TOKEN)

        @self.app.middleware("http")
        async def log(request: Request, call_next):  # noqa: ANN001, ANN202
            self.calls.append((request.method, request.url.path))
            if self.require_token and request.url.path.startswith("/v1/") and request.headers.get("x-service-token") != self.require_token:
                return JSONResponse(status_code=401, content={"error": {"code": "service_token", "message": "Only the hub may call this."}})
            wanted = (request.headers.get("x-workspace") or "").strip() or None
            self.workspace_headers.append((request.url.path, wanted))
            self.service_tokens.append(request.headers.get("x-service-token"))
            if wanted and wanted not in self.workspace_ids() and request.url.path != "/health":
                return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": f"No workspace {wanted!r}."}})
            self.select_workspace(wanted)  # None = the workspace the agent has open (the transport shares our task context)
            return await call_next(request)

        @self.app.exception_handler(FakeError)
        async def _error(_: Request, exc: FakeError) -> JSONResponse:
            return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})

        @self.app.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        # What the admin page's provisioning does to an agent. Like the real agents, creating switches to the new workspace
        # and removing refuses the active one and "main".
        self.extra: list[dict] = []  # workspaces created through the API
        self.active_override: str | None = None

        @self.app.post("/v1/workspaces", status_code=201)
        async def create_workspace(request: Request) -> dict:
            body = await request.json()
            name = (body.get("name") or "").strip()
            if not name:
                raise FakeError(422, "invalid_request", "Give the workspace a name.")
            stem = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30].strip("-") or "workspace"
            taken, ws_id, n = set(self.workspace_ids()), stem, 2
            while ws_id in taken:
                ws_id, n = f"{stem}-{n}", n + 1
            space = {"id": ws_id, "name": name, "kind": body.get("kind", "company")}
            self.extra.append(space)
            self.on_new_workspace(space)
            self.active_override = ws_id
            return {"workspace": {**space, "active": True}, "seeded": None}

        @self.app.post("/v1/workspaces/{workspace_id}/activate")
        async def activate_workspace(workspace_id: str) -> dict:
            if workspace_id not in self.workspace_ids():
                raise FakeError(404, "not_found", f"No workspace {workspace_id!r}.")
            self.active_override = workspace_id
            return {"active": workspace_id, "workspaces": self.workspace_list()}

        @self.app.delete("/v1/workspaces/{workspace_id}")
        async def delete_workspace(workspace_id: str) -> dict:
            space = next((w for w in self.base_workspaces() + self.extra if w["id"] == workspace_id), None)
            if space is None:
                raise FakeError(404, "not_found", f"No workspace {workspace_id!r}.")
            if space["kind"] == "main":
                raise FakeError(422, "invalid_request", "The original workspace can't be removed.")
            if workspace_id == self.current_active():
                raise FakeError(422, "invalid_request", "Switch to another workspace before removing this one.")
            self.extra = [w for w in self.extra if w["id"] != workspace_id]
            self.on_removed_workspace(workspace_id)
            self.removed.append(workspace_id)
            return {"active": self.current_active(), "workspaces": self.workspace_list()}

        self.removed: list[str] = []

        # What GET /v1/usage reports per workspace: {"calls", "failed", "input_tokens", "output_tokens", "bytes", "since"}.
        self.usage_data: dict[str, dict] = {}

        @self.app.get("/v1/usage")
        async def usage(request: Request) -> dict:
            workspace = (request.headers.get("x-workspace") or "").strip() or self.current_active()
            u = self.usage_data.get(workspace, {})
            return {"workspace": workspace,
                    "model": {"calls": u.get("calls", 0), "failed": u.get("failed", 0), "input_tokens": u.get("input_tokens", 0),
                              "output_tokens": u.get("output_tokens", 0), "by_purpose": {}, "since": u.get("since"), "last": None},
                    "storage": {"database_bytes": u.get("bytes", 0), "uploads_bytes": 0, "uploads_files": 0, "other_bytes": 0,
                                "total_bytes": u.get("bytes", 0)}}

    def workspace_ids(self) -> list[str]:
        return []

    def base_workspaces(self) -> list[dict]:
        return []

    def default_workspace_id(self) -> str:
        return ""

    def on_new_workspace(self, space: dict) -> None:
        return None

    def on_removed_workspace(self, workspace_id: str) -> None:
        return None

    def current_active(self) -> str:
        return self.active_override or self.default_workspace_id()

    def workspace_list(self) -> list[dict]:
        active = self.current_active()
        return [{**w, "created_at": "2026-01-01", "active": w["id"] == active} for w in self.base_workspaces() + self.extra]

    def select_workspace(self, workspace_id: str | None) -> None:
        return None

    def transport(self) -> httpx.AsyncBaseTransport:
        agent = self

        class Switchable(httpx.AsyncBaseTransport):
            def __init__(self) -> None:
                self.inner = httpx.ASGITransport(app=agent.app)

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                if agent.down:
                    raise httpx.ConnectError("connection refused", request=request)
                return await self.inner.handle_async_request(request)

        return Switchable()

    def matching(self, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> list[tuple[str, str]]:
        return [c for c in self.calls if any(c[0] == m and p.search(c[1]) for m, p in patterns)]

    def spending_calls(self) -> list[tuple[str, str]]:
        return self.matching(SPENDING)

    def forbidden_calls(self) -> list[tuple[str, str]]:
        return self.matching(FORBIDDEN)

    def called(self, method: str, pattern: str) -> list[tuple[str, str]]:
        return [c for c in self.calls if c[0] == method and re.search(pattern, c[1])]


# -- Deal Intelligence ------------------------------------------------------------------------------------

BRIEF_CONTENT = {
    "summary": "Cedarline is waiting on legal; the economic buyer has not been met.",
    "summary_sources": ["INT-0001"],
    "this_deal": [{"text": "Legal asked for a data-residency clause.", "source_ids": ["INT-0002"]}],
    "memory": [{"text": "Similar deals stalled at legal.", "source_ids": ["D-010"]}],
    "next_steps": [{"play_code": "PLAY-03", "name": "Executive sponsor", "rationale": "Bring a VP to the next call.",
                    "source_ids": ["INT-0001"], "counts": {}, "reasons": ["won 3 of 4 similar deals"]}],
    "missing_info": ["Who signs?"],
    "flags": [{"code": "stale", "severity": "medium", "text": "No contact for 21 days.", "evidence": ["INT-0003"]}],
    "warnings": [{"kind": "objection", "text": "Security review objections went unresolved.", "similar_lost": 3,
                  "similar_total": 5, "source_deals": ["D-010", "D-011"]}],
    "avoid": [{"play_code": "PLAY-07", "name": "Discount offer", "text": "Discount offer: used in 4 similar deals, 0 won.",
               "source_deals": ["D-012"]}],
    "similar": ["D-010"], "mode": "hindsight", "n_closed": 19, "degraded": False,
}


class FakeDeal(FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.workspace = {"id": "ws-demo", "name": "Halcyon Demo", "kind": "demo"}  # the one the agent has open
        self.other_workspace = {"id": "ws-other", "name": "Brightwater Team", "kind": "company"}
        self.empty_workspace = {"id": "ws-empty", "name": "My deals", "kind": "main"}  # a workspace with no deals yet
        self.plays_by_ws: dict[str, list[dict]] = {}
        self.play_bodies: list[dict] = []
        self.stores = {w["id"]: {"deals": {}, "briefs": {}, "jobs": {}}
                       for w in (self.workspace, self.other_workspace, self.empty_workspace)}
        self._ws = ContextVar("fake_deal_workspace", default=self.workspace["id"])
        self.job_polls = 2  # a job reports "running" this many times before it completes
        self.go_down_after_job_start = False  # the agent "crashes" right after it accepts a job
        self.fail_jobs: str | None = None  # a job kind ("signals" or "brief") that should fail with error_info
        self.outcome_bodies: list[dict] = []
        self.created_bodies: list[dict] = []
        self.update_bodies: list[dict] = []
        self.note_bodies: list[dict] = []
        self.ask_bodies: list[dict] = []
        self.portfolio_bodies: list[dict] = []
        self.portfolio_answer: dict = {"found": True, "answer": "Two of the lost deals went on price.", "sources": ["D-006"],
                                       "grounded": True, "findings": [], "model": "fake"}
        self.followup_bodies: list[dict] = []
        self.answer: dict = {"found": True, "answer": "Dana said SSO is a blocker for the pilot.", "sources": ["INT-0001"],
                             "grounded": True, "mode": "hindsight", "degraded": False}
        self._seed()
        self._routes()

    def workspace_ids(self) -> list[str]:
        return list(self.stores)

    def base_workspaces(self) -> list[dict]:
        return [self.workspace, self.other_workspace, self.empty_workspace]

    def default_workspace_id(self) -> str:
        return self.workspace["id"]

    def on_new_workspace(self, space: dict) -> None:
        self.stores[space["id"]] = {"deals": {}, "briefs": {}, "jobs": {}}

    def on_removed_workspace(self, workspace_id: str) -> None:
        self.stores.pop(workspace_id, None)

    def select_workspace(self, workspace_id: str | None) -> None:
        self._ws.set(workspace_id or self.workspace["id"])

    @property
    def deals(self) -> dict[int, dict]:
        return self.stores[self._ws.get()]["deals"]

    @property
    def briefs(self) -> dict[int, dict]:
        return self.stores[self._ws.get()]["briefs"]

    @property
    def jobs(self) -> dict[int, dict]:
        return self.stores[self._ws.get()]["jobs"]

    # data ------------------------------------------------------------------------------------------------

    def _seed(self) -> None:
        plays = [{"code": "PLAY-03", "name": "Executive sponsor"}]
        specs = [
            (1, "Cedarline Renewal", "Cedarline Systems", "ready", 3, []),
            (2, "Juniper Expansion", "Juniper Health", "none", 2, []),
            (3, "Larkfield Platform", "Larkfield Bank", "ready", 4, plays),
            (4, "Larkfield Security Add-on", "Larkfield Bank", "none", 1, []),
            (5, "Brightwater Pilot", "Brightwater Co", "none", 0, []),
        ]
        for did, name, account, signals, n, used in specs:
            self.deals[did] = self._deal(did, name, account, signals, n, used)
        self.deals[6] = {**self._deal(6, "Oakhurst Renewal", "Oakhurst", "ready", 2, []), "result": "lost", "loss_reason": "price"}
        other = self.stores[self.other_workspace["id"]]["deals"]  # a second team's deals, with different ids and names
        other[1] = self._deal(1, "Zephyr Pilot", "Zephyr Labs", "ready", 3, [])
        other[2] = self._deal(2, "Cedarline Pilot", "Cedarline Systems", "ready", 2, [])

    @staticmethod
    def _deal(did: int, name: str, account: str, signals: str, n: int, plays: list[dict]) -> dict:
        return {
            "id": did, "code": f"D-{did:03d}", "name": name, "account": account, "industry": None, "segment": None,
            "amount": None, "stage": "proposal", "owner": None, "opened_on": "2026-01-05", "closed_on": None,
            "result": "open", "loss_reason": None, "signals_status": signals, "signals_error": None,
            "_interactions": [
                {"id": i, "code": f"INT-{i:04d}", "kind": "email", "occurred_on": "2026-02-01", "author": "x", "subject": "s",
                 "text": "t", "hindsight_status": "retained"} for i in range(1, n + 1)
            ],
            "_plays": plays,
            "last_activity": None,
        }

    def summary(self, deal: dict) -> dict:
        return {**{k: v for k, v in deal.items() if not k.startswith("_")}, "interactions": len(deal["_interactions"])}

    def detail(self, deal: dict) -> dict:
        signals = None
        if deal["signals_status"] == "ready":
            signals = {"objections": [], "competitors": [], "pricing": {}, "promises": [], "source": "model",
                       "plays_used": deal["_plays"]}
        return {
            **{k: v for k, v in deal.items() if not k.startswith("_")}, "interactions": deal["_interactions"],
            "stakeholders": [], "signals": signals, "flags": [],
        }

    def _deal_or_404(self, deal_id: int) -> dict:
        if deal_id not in self.deals:
            raise FakeError(404, "not_found", f"No deal {deal_id}.")
        return self.deals[deal_id]

    def _new_job(self, kind: str, deal_id: int) -> int:
        job_id = len(self.jobs) + 101
        self.jobs[job_id] = {"id": job_id, "kind": kind, "target_id": deal_id, "status": "running", "done": 0, "total": 1,
                             "error": None, "warning": None, "polls": 0, "payload": {}, "error_info": None}
        return job_id

    def _finish(self, job: dict) -> None:
        deal = self.deals[job["target_id"]]
        if self.fail_jobs == job["kind"]:
            job.update(status="failed", error="429 quota exceeded", error_info={
                "reason": "quota", "title": "The model quota is used up", "detail": "The key has no requests left today.",
                "action": "Wait for the quota to reset or switch the key.", "switch_model": True})
            if job["kind"] == "signals":
                deal["signals_status"] = "failed"
            return
        job.update(status="completed", done=1)
        if job["kind"] == "signals":
            deal["signals_status"] = "ready"
        else:
            self.briefs[deal["id"]] = {
                "id": job["id"], "deal_id": deal["id"], "mode": "hindsight", "status": "ready", "error": None,
                "content": copy.deepcopy(BRIEF_CONTENT), "flags": [], "evidence": {}, "memory_state": "ready",
            }

    # routes ------------------------------------------------------------------------------------------------

    def _routes(self) -> None:
        app = self.app

        @app.get("/v1/workspaces")
        async def workspaces() -> dict:
            return {"active": self.current_active(), "workspaces": self.workspace_list()}

        @app.get("/v1/plays")
        async def list_plays() -> dict:
            return {"plays": self.plays_by_ws.get(self._ws.get(), [])}

        @app.post("/v1/plays")
        async def save_plays(request: Request) -> dict:
            body = await request.json()
            self.play_bodies.append(body)
            plays = self.plays_by_ws.setdefault(self._ws.get(), [])
            known = {"sso", "security_review", "pricing", "integration", "timeline", "legal_terms", "data_residency", "support", "feature_gap"}
            added, updated, ignored = [], [], set()
            for item in body["plays"]:
                wanted = item.get("addresses") or []
                ignored |= {a for a in wanted if a not in known}
                row = next((p for p in plays if p["name"].lower() == item["name"].lower()), None)
                fields = {"name": item["name"], "description": item.get("description") or item["name"],
                          "category": item.get("category") or "process", "addresses": [a for a in wanted if a in known]}
                if row is None:
                    row = {"code": f"PLAY-{len(plays) + 1:02d}", **fields}
                    plays.append(row)
                    added.append(row["code"])
                else:
                    row.update(fields)
                    updated.append(row["code"])
            return {"added": added, "updated": updated, "ignored_objections": sorted(ignored), "plays": plays}

        @app.get("/v1/deals")
        async def list_deals() -> dict:
            return {"deals": [self.summary(d) for d in self.deals.values()]}

        @app.get("/v1/deals/{deal_id}")
        async def get_deal(deal_id: int) -> dict:
            return self.detail(self._deal_or_404(deal_id))

        @app.post("/v1/deals", status_code=201)
        async def create_deal(
            name: str = Form(...), account: str = Form(...), amount: str | None = Form(None), segment: str | None = Form(None),
            industry: str | None = Form(None), files: list[UploadFile] = File(default_factory=list),
        ) -> dict:
            if segment and segment not in ("smb", "mid_market", "enterprise"):  # the real app refuses anything else
                raise FakeError(422, "invalid_request", f"Unknown segment {segment}.")
            parsed = []
            for f in files:  # parse everything first: a bad file must leave no deal behind
                data = await f.read()
                if data.startswith(b"CORRUPT"):
                    raise FakeError(422, "parse_error", f"{f.filename} could not be read.")
                parsed.append(f.filename)
            self.created_bodies.append({"name": name, "account": account, "amount": amount, "segment": segment, "files": parsed,
                                        **({"industry": industry} if industry else {})})
            did = max(self.deals) + 1
            deal = self._deal(did, name, account, "none", len(parsed), [])
            deal.update(industry=industry, segment=segment)
            self.deals[did] = deal
            return self.detail(deal)

        @app.patch("/v1/deals/{deal_id}")
        async def update_deal(deal_id: int, request: Request) -> dict:
            deal = self._deal_or_404(deal_id)
            body = await request.json()
            if body.get("segment") and body["segment"] not in ("smb", "mid_market", "enterprise"):
                raise FakeError(422, "invalid_request", "Unknown segment.")
            self.update_bodies.append({"deal_id": deal_id, **body})
            deal.update({k: v for k, v in body.items() if k in ("name", "account", "industry", "segment", "amount", "owner")})
            return self.detail(deal)

        @app.post("/v1/deals/{deal_id}/files")
        async def add_files(deal_id: int, files: list[UploadFile] = File(...)) -> dict:
            deal = self._deal_or_404(deal_id)
            for f in files:
                await f.read()
                n = len(deal["_interactions"]) + 1
                deal["_interactions"].append({"id": n, "code": f"INT-{n:04d}", "kind": "email", "text": "t"})
            return self.detail(deal)

        @app.post("/v1/deals/{deal_id}/notes")
        async def add_note(deal_id: int, request: Request) -> dict:
            deal = self._deal_or_404(deal_id)
            self.note_bodies.append(await request.json())
            n = len(deal["_interactions"]) + 1
            deal["_interactions"].append({"id": n, "code": f"INT-{n:04d}", "kind": "call_note", "text": "t"})
            return self.detail(deal)

        @app.post("/v1/deals/{deal_id}/signals", status_code=202)
        async def signals(deal_id: int) -> dict:
            deal = self._deal_or_404(deal_id)
            if not deal["_interactions"]:
                raise FakeError(409, "no_interactions", "This deal has no interactions yet.")
            self.model_calls += 1
            deal["signals_status"] = "extracting"
            job_id = self._new_job("signals", deal_id)
            self.down = self.go_down_after_job_start
            return {"job_id": job_id}

        @app.post("/v1/deals/{deal_id}/brief", status_code=202)
        async def brief(deal_id: int) -> dict:
            deal = self._deal_or_404(deal_id)
            if deal["signals_status"] != "ready":
                raise FakeError(409, "signals_missing", "Run signals first.")
            self.model_calls += 1
            job_id = self._new_job("brief", deal_id)
            self.down = self.go_down_after_job_start
            return {"job_id": job_id}

        @app.get("/v1/deals/{deal_id}/brief")
        async def get_brief(deal_id: int) -> dict:
            self._deal_or_404(deal_id)
            if deal_id not in self.briefs:
                raise FakeError(404, "no_brief", "No brief yet.")
            return self.briefs[deal_id]

        @app.post("/v1/deals/{deal_id}/ask")
        async def ask(deal_id: int, request: Request) -> dict:
            self._deal_or_404(deal_id)
            body = await request.json()
            self.ask_bodies.append({"deal_id": deal_id, **body})
            self.model_calls += 1
            return {"deal_id": deal_id, "question": body["question"], **self.answer}

        @app.post("/v1/portfolio/ask")
        async def ask_portfolio(request: Request) -> dict:
            body = await request.json()
            self.portfolio_bodies.append(body)
            if not self.deals:
                raise FakeError(409, "no_deals", "This workspace has no deals yet.")
            self.model_calls += 1
            open_n = sum(1 for d in self.deals.values() if d["result"] == "open")
            lost_n = sum(1 for d in self.deals.values() if d["result"] == "lost")
            return {**self.portfolio_answer, "question": body["question"], "stats": {
                "deals": len(self.deals), "open": open_n, "won": 0, "lost": lost_n, "win_rate": 0.0,
                "loss_reasons": {"price": lost_n} if lost_n else {}}, "deals_considered": len(self.deals),
                "deals_total": len(self.deals), "truncated": False}

        @app.post("/v1/deals/{deal_id}/followup")
        async def followup(deal_id: int, request: Request) -> dict:
            deal = self._deal_or_404(deal_id)
            body = await request.json()
            self.followup_bodies.append({"deal_id": deal_id, **body})
            if deal["result"] != "open":
                raise FakeError(409, "deal_closed", "A follow-up is only drafted for an open deal.")
            if deal_id not in self.briefs:
                raise FakeError(409, "no_brief", "Write the brief first.")
            self.model_calls += 1
            return {"deal_id": deal_id, "kind": body.get("kind", "email"), "subject": "Next steps on the data-residency clause",
                    "body": "Hi [name],\n\nFollowing up on the clause.\n\nThanks", "sources": ["INT-0001"], "grounded": True,
                    "play_code": "PLAY-03", "step_name": "Executive sponsor"}

        @app.get("/v1/jobs/{job_id}")
        async def get_job(job_id: int) -> dict:
            if job_id not in self.jobs:
                raise FakeError(404, "not_found", f"No job {job_id}.")
            job = self.jobs[job_id]
            if job["status"] == "running":
                job["polls"] += 1
                if job["polls"] > self.job_polls:
                    self._finish(job)
            return {k: v for k, v in job.items() if k != "polls"}

        @app.put("/v1/deals/{deal_id}/outcome")
        async def outcome(deal_id: int, request: Request) -> dict:
            deal = self._deal_or_404(deal_id)
            body = await request.json()
            self.outcome_bodies.append({"deal_id": deal_id, **body})
            if body["result"] == "lost" and not body.get("loss_reason"):
                raise FakeError(422, "invalid_request", "loss_reason must be one of competitor, price, ...")
            used = body["plays_used"] if body.get("plays_used") is not None else [p["code"] for p in deal["_plays"]]
            deal.update(result=body["result"], loss_reason=body.get("loss_reason"), closed_on="2026-03-01")
            lost = body["result"] == "lost" and body.get("loss_reason") in ("feature_gap", "security_compliance", "unresolved_objection")
            return {
                "deal": self.summary(deal),
                "credit": [{"play_code": c, "delta": -1.0 if lost else 0.0, "times_used": 4, "won": 2, "lost_quality": 1, "lost_other": 0} for c in used],
                "events": [f"Recorded {deal['code']} as {body['result']}."], "lessons_added": 1,
                "memory_note": "This reason is not about the plays used, so they get no credit either way.", "plays_used": used,
            }


# -- RFP Memory Assistant -------------------------------------------------------------------------------

QUESTIONS = [
    "Describe your information security program.", "Do you hold SOC 2 Type II certification?",
    "How is customer data encrypted at rest?", "What is your incident response time commitment?",
    "List your sub-processors.", "Describe your business continuity plan.",
]


class FakeRfp(FakeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.workspace = {"id": "ws-acme", "name": "Acme Security", "kind": "company"}
        self.other_workspace = {"id": "ws-globex", "name": "Globex Bids", "kind": "company"}
        self._ws = ContextVar("fake_rfp_workspace", default=self.workspace["id"])
        self.company_set_up = True
        self.company_sheets: dict[str, dict] = {}  # workspace -> {"company", "facts"}
        self.locked_workspaces: set[str] = set()  # workspaces with the read-only sample fact sheet
        self.company_puts: list[dict] = []
        self.proposals: dict[int, dict] = {}
        self.proposal_bodies: list[dict] = []
        self.proposal_mode = "ok"  # ok | fail | empty
        self.proposal_pairs = [("Do you support SSO?", "Yes, SAML 2.0."), ("Where is data hosted?", "In the EU.")]
        self.kept_answers: list[dict] = []
        self.library_bodies: list[dict] = []
        self.library_empty = False
        self.library_answer: dict = {
            "found": True, "answer": "We hold SOC 2 Type II, renewed in 2026.", "sources": ["FACT-003", "ANS-0012"],
            "evidence": [{"id": "FACT-003", "label": "Certifications"}, {"id": "ANS-0012", "label": "Do you hold SOC 2? (Pinecrest)"}],
            "grounded": True, "findings": [], "degraded": False, "model": "fake"}
        self.extract_mode = "ok"  # ok | late_parse_fail | job_fail
        self.draft_mode = "ok"  # ok | partial | job_fail
        self.questions = list(QUESTIONS)
        self.mandatory = 4  # the first N questions are mandatory
        self.job_polls = 2
        self.projects: dict[int, dict] = {}
        self.jobs: dict[int, dict] = {}
        self.review_bodies: list[tuple[int, dict]] = []
        self.outcome_bodies: list[dict] = []
        self.create_bodies: list[dict] = []
        self._routes()

    # data ------------------------------------------------------------------------------------------------

    def _project(self, pid: int) -> dict:
        if pid not in self.projects:
            raise FakeError(404, "not_found", f"No project {pid}.")
        return self.projects[pid]

    def _job(self, kind: str, target: int, total: int = 0) -> dict:
        job_id = len(self.jobs) + 201
        job = {"id": job_id, "kind": kind, "target_id": target, "status": "running", "done": 0, "total": total,
               "error": None, "warning": None, "error_info": None, "polls": 0}
        self.jobs[job_id] = job
        return job

    @staticmethod
    def _job_view(job: dict | None) -> dict | None:
        return None if job is None else {k: v for k, v in job.items() if k != "polls"}

    def _tick(self, job: dict) -> None:
        if job["status"] != "running":
            return
        job["polls"] += 1
        if job["polls"] > self.job_polls:
            (self._finish_extract if job["kind"] == "extract_requirements" else self._finish_draft)(job)

    def _finish_extract(self, job: dict) -> None:
        project = self.projects[job["target_id"]]
        if self.extract_mode == "late_parse_fail":
            # The job itself looks fine; only the project carries the (plain text) parse error.
            project.update(state="failed", error="The PDF has no text layer (it looks scanned). Upload a text-based copy.")
            job["status"] = "completed"
        elif self.extract_mode == "job_fail":
            info = {"reason": "quota", "title": "The model quota is used up", "detail": "Extraction needs one model call.",
                    "action": "Wait for the quota to reset.", "switch_model": False}
            project.update(state="failed", error="429 quota", error_info=info)
            job.update(status="failed", error="429 quota", error_info=info)
        else:
            project["state"] = "requirements_extracted"
            project["requirements"] = [
                {"id": 500 + i, "code": f"R-{i + 1:03d}", "order": i, "section": "Security", "reference": None, "question": q,
                 "mandatory": i < self.mandatory, "word_limit": None, "final": False, "final_text": None, "draft": None, "draft_count": 0}
                for i, q in enumerate(self.questions)
            ]
            job["status"] = "completed"

    def _finish_draft(self, job: dict) -> None:
        project = self.projects[job["target_id"]]
        reqs = project["requirements"]
        if self.draft_mode == "job_fail":
            job.update(status="failed", error="403 key rejected", error_info={
                "reason": "auth", "title": "The API key was rejected", "detail": "No answer could be drafted.",
                "action": "Check the key in the RFP assistant.", "switch_model": False})
            project["state"] = "in_review"
            return
        for i, req in enumerate(reqs):
            if self.draft_mode == "partial" and i >= 5:
                continue  # the job stopped early: left undrafted
            req["draft_count"] = 1
            base = {"id": 900 + i, "version": 1, "status": "drafted", "answer": f"We do this ({i}).", "flags": [], "sme_question": None,
                    "error": None, "error_info": None, "sources": ["F-1"]}
            if self.draft_mode == "partial":
                if i == 2:
                    base["flags"] = ["evidence_partial"]
                elif i == 3:
                    base.update(status="needs_sme", answer="[SME input required: confirm the response time]", sme_question="What is it?")
                elif i == 4:
                    base.update(status="failed", answer="", error="Model returned an empty answer.")
            req["draft"] = base
        job.update(status="completed", done=len(reqs), warning="Stopped early: the model quota ran out after 5 requirements." if self.draft_mode == "partial" else None)
        project["state"] = "in_review"
        self._refresh_state(project)

    def _refresh_state(self, project: dict) -> None:
        if project["state"] in ("extracting", "requirements_extracted", "drafting", "failed"):
            return
        all_final = bool(project["requirements"]) and all(r["final"] for r in project["requirements"])
        if all_final and project["state"] != "exported":
            project["state"] = "approved"
        elif not all_final:
            project["state"] = "in_review"

    def view(self, project: dict) -> dict:
        reqs = project["requirements"]
        drafts = [r["draft"] for r in reqs if r["draft"]]
        stats = {"requirements": len(reqs), "drafted": len(drafts), "final": sum(r["final"] for r in reqs),
                 "needs_sme": sum(d["status"] == "needs_sme" for d in drafts), "flagged": sum(bool(d["flags"]) for d in drafts),
                 "failed": sum(d["status"] == "failed" for d in drafts)}
        job = self.jobs.get(project["_job_id"])
        return {**{k: v for k, v in project.items() if not k.startswith("_")}, "stats": stats, "facts": [], "job": self._job_view(job)}

    # routes ------------------------------------------------------------------------------------------------

    def workspace_ids(self) -> list[str]:
        return [self.workspace["id"], self.other_workspace["id"], *[w["id"] for w in self.extra]]

    def base_workspaces(self) -> list[dict]:
        return [self.workspace, self.other_workspace]

    def default_workspace_id(self) -> str:
        return self.workspace["id"]

    def select_workspace(self, workspace_id: str | None) -> None:
        self._ws.set(workspace_id or self.workspace["id"])

    def _routes(self) -> None:
        app = self.app

        @app.get("/v1/workspaces")
        async def workspaces() -> dict:
            return {"active": self.current_active(), "workspaces": self.workspace_list()}

        @app.post("/v1/library/ask")
        async def ask_library(request: Request) -> dict:
            body = await request.json()
            self.library_bodies.append(body)
            if self.library_empty:
                raise FakeError(409, "no_library", "No facts and no answers.")
            self.model_calls += 1
            return {"question": body["question"], **self.library_answer}

        # -- company facts and past proposals (what the chat can feed into the memory) --------------------------
        @app.get("/v1/company")
        async def get_company() -> dict:
            sheet = self.company_sheets.get(self._ws.get())
            return {"company": sheet["company"] if sheet else None, "facts": sheet["facts"] if sheet else [],
                    "locked": self._ws.get() in self.locked_workspaces, "set_up": sheet is not None}

        @app.put("/v1/company")
        async def put_company(request: Request) -> dict:
            if self._ws.get() in self.locked_workspaces:
                raise FakeError(409, "fact_sheet_locked", "This workspace uses the read-only sample fact sheet.")
            body = await request.json()
            self.company_puts.append(body)
            used = [int(f["id"].split("-")[1]) for f in body["facts"] if f.get("id")]
            n = max(used, default=0)
            facts = []
            for f in body["facts"]:
                if not f.get("id"):
                    n += 1
                facts.append({**f, "id": f.get("id") or f"FACT-{n:03d}"})
            self.company_sheets[self._ws.get()] = {"company": body["company"], "facts": facts}
            return {"company": body["company"], "facts": facts, "locked": False, "set_up": True}

        @app.post("/v1/library", status_code=202)
        async def import_proposal(file: UploadFile = File(...), client: str | None = Form(None), industry: str | None = Form(None),
                                  submitted_on: str | None = Form(None), result: str = Form("unknown"),
                                  loss_reason: str | None = Form(None)) -> dict:
            self.model_calls += 1
            pid = len(self.proposals) + 1
            fields = {"client": client, "industry": industry, "submitted_on": submitted_on, "result": result, "loss_reason": loss_reason}
            self.proposal_bodies.append({"filename": file.filename, **fields})
            pairs = [] if self.proposal_mode == "empty" else [
                {"id": pid * 100 + i, "order": i, "section": None, "reference": None, "question": q, "answer": a, "decision": "pending"}
                for i, (q, a) in enumerate(self.proposal_pairs, start=1)]
            self.proposals[pid] = {"id": pid, "filename": file.filename, **fields, "status": "extracting", "error": None,
                                   "pairs": pairs, "polls": 0, "workspace": self._ws.get()}
            return {k: v for k, v in self.proposals[pid].items() if k not in ("pairs", "polls")} | {"pairs": [], "job": {"id": 900 + pid}}

        @app.get("/v1/library/proposals/{pid}")
        async def get_proposal(pid: int) -> dict:
            p = self.proposals.get(pid)
            if p is None:
                raise FakeError(404, "not_found", f"No past proposal {pid}.")
            p["polls"] += 1
            if p["status"] == "extracting" and p["polls"] > 1:
                p["status"], p["error"] = ("failed", "The document has no readable text.") if self.proposal_mode == "fail" else ("extracted", None)
            return {k: v for k, v in p.items() if k != "polls"} | {"error_info": {"message": p["error"]} if p["error"] else None}

        @app.post("/v1/library/proposals/{pid}/confirm")
        async def confirm_pairs(pid: int, request: Request) -> dict:
            body = await request.json()
            p = self.proposals[pid]
            if p["status"] != "extracted":
                raise FakeError(409, "invalid_state", "only extracted pairs can be confirmed")
            p["status"] = "confirmed"
            kept = [d for d in body["pairs"] if d["decision"] == "kept"]
            self.kept_answers += kept
            return {"created": [f"ANS-{d['id']:04d}" for d in kept]}

        @app.post("/v1/library/proposals/{pid}/discard")
        async def discard_proposal(pid: int) -> dict:
            self.proposals[pid]["status"] = "discarded"
            return {"id": pid, "status": "discarded"}

        @app.get("/v1/workspace")
        async def workspace() -> dict:
            current = self.workspace if self._ws.get() == self.workspace["id"] else self.other_workspace
            return {**current, "company_set_up": self.company_set_up, "fact_sheet_locked": False,
                    "projects": len(self.projects), "past_proposals": 0, "answers": 0}

        @app.get("/v1/projects")
        async def list_projects() -> list:
            return [{k: p[k] for k in ("id", "name", "client", "state", "created_at")} for p in self.projects.values()]

        @app.post("/v1/projects", status_code=202)
        async def create_project(
            file: UploadFile = File(...), name: str | None = Form(None), client: str | None = Form(None),
            industry: str | None = Form(None),
        ) -> dict:
            data = await file.read()
            self.create_bodies.append({"filename": file.filename, "size": len(data), "name": name, "client": client, "industry": industry})
            self.model_calls += 1
            pid = len(self.projects) + 1
            fname = file.filename or "upload"
            job = self._job("extract_requirements", pid)
            self.projects[pid] = {
                "id": pid, "name": (name or "").strip() or fname.rsplit(".", 1)[0], "client": client or None, "industry": industry or None,
                "state": "extracting", "error": None, "error_info": None, "filename": fname,
                "file_kind": fname.rsplit(".", 1)[-1].lower(), "created_at": "2026-03-01T00:00:00", "requirements": [],
                "outcome": None, "_job_id": job["id"],
            }
            return {"id": pid, "name": self.projects[pid]["name"], "client": client, "state": "extracting", "filename": fname,
                    "created_at": "2026-03-01T00:00:00", "requirement_count": 0, "job": self._job_view(job)}

        @app.get("/v1/projects/{pid}")
        async def get_project(pid: int) -> dict:
            project = self._project(pid)
            if project["state"] == "extracting":
                self._tick(self.jobs[project["_job_id"]])
            return self.view(project)

        @app.post("/v1/projects/{pid}/draft", status_code=202)
        async def draft(pid: int) -> dict:
            project = self._project(pid)
            if project["state"] not in ("requirements_extracted", "in_review", "approved", "exported"):
                raise FakeError(409, "invalid_state", f"A '{project['state']}' project can't be drafted.")
            if not self.company_set_up:
                raise FakeError(409, "no_company_facts", "Company facts are required before drafting.")
            self.model_calls += len(project["requirements"])
            project["state"] = "drafting"
            job = self._job("draft_all", pid, total=len(project["requirements"]))
            project["_job_id"] = job["id"]
            return self._job_view(job)

        @app.get("/v1/jobs/{job_id}")
        async def get_job(job_id: int) -> dict:
            if job_id not in self.jobs:
                raise FakeError(404, "not_found", f"No job {job_id}.")
            job = self.jobs[job_id]
            self._tick(job)
            if job["status"] == "running" and job["total"]:
                job["done"] = min(job["total"] - 1, job["polls"])
            return self._job_view(job)

        @app.post("/v1/requirements/{rid}/review")
        async def review(rid: int, request: Request) -> dict:
            body = await request.json()
            self.review_bodies.append((rid, body))
            for project in self.projects.values():
                for req in project["requirements"]:
                    if req["id"] != rid:
                        continue
                    d = req["draft"]
                    if d is None:
                        raise FakeError(409, "invalid_state", "There is no draft to review yet.")
                    tags = body.get("reason_tags") or []
                    if set(tags) - {"outdated", "wrong_product", "too_long", "too_short", "too_vague", "incorrect", "client_specific", "tone", "other"}:
                        raise FakeError(422, "invalid_request", "Unknown review reason tags.")
                    if body["action"] == "accepted":
                        if not d["answer"].strip():
                            raise FakeError(422, "invalid_request", "This draft has no answer to accept; rewrite it instead.")
                        if "[SME input required:" in d["answer"]:
                            raise FakeError(422, "invalid_request", "Replace every SME placeholder before accepting.")
                        req.update(final=True, final_text=d["answer"])
                    elif body["action"] in ("edited", "rewritten"):
                        final = (body.get("final_text") or "").strip()
                        if not final:
                            raise FakeError(422, "invalid_request", "Provide the final text for an edit or rewrite.")
                        if "[SME input required:" in final:
                            raise FakeError(422, "invalid_request", "Replace every SME placeholder before accepting, editing, or rewriting this answer.")
                        req.update(final=True, final_text=final)
                    elif body["action"] == "rejected":
                        req.update(final=False, final_text=None)
                    else:
                        raise FakeError(422, "invalid_request", "action must be one of accepted, edited, rewritten, rejected.")
                    self._refresh_state(project)
                    return req
            raise FakeError(404, "not_found", f"No requirement {rid}.")

        @app.get("/v1/projects/{pid}/export")
        async def export(pid: int, format: str = "docx") -> Response:
            project = self._project(pid)
            pending = [r for r in project["requirements"] if not r["final"]]
            if not project["requirements"] or pending:
                raise FakeError(409, "not_final", f"Every requirement must be accepted ({len(pending)} still open).")
            if format == "xlsx" and project["file_kind"] != "xlsx":
                raise FakeError(422, "invalid_request", "Excel export needs an Excel RFP; export this one as Word.")
            project["state"] = "exported"
            return Response(
                content=f"FAKE-{format.upper()}-FILE".encode(), media_type=f"application/fake-{format}",
                headers={"Content-Disposition": f'attachment; filename="{project["name"]} - response.{format}"'},
            )

        @app.put("/v1/projects/{pid}/outcome")
        async def outcome(pid: int, request: Request) -> dict:
            project = self._project(pid)
            body = await request.json()
            if body["result"] not in ("won", "lost", "no_decision", "unknown"):
                raise FakeError(422, "invalid_request", "result must be one of won, lost, no_decision, unknown.")
            self.outcome_bodies.append({"project_id": pid, **body})
            project["outcome"] = {"result": body["result"], "loss_reason": body.get("loss_reason"), "decided_at": None}
            return {"project_id": pid, "result": body["result"], "loss_reason": body.get("loss_reason"), "decided_at": None}


class Fakes:
    def __init__(self) -> None:
        self.deal = FakeDeal()
        self.rfp = FakeRfp()

    def all_forbidden(self) -> list[tuple[str, str]]:
        return self.deal.forbidden_calls() + self.rfp.forbidden_calls()

    def all_spending(self) -> list[tuple[str, str]]:
        return self.deal.spending_calls() + self.rfp.spending_calls()

