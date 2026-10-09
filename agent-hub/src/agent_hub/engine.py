"""The conversation engine: turns chat messages and button clicks into work done in the agents.

One conversation = one state machine saved in `conversations.state` plus an append-only event log.
The rules every flow follows (they are enforced here, not left to each flow):

* A step that spends model calls first posts a confirmation (text with the cost, a one-shot token).
  The calls the person approved are recorded as a budget, and `spend()` refuses to go past it.
* A token fires at most once. Clicking it again, or after the request moved on, says so.
* Long steps run as background tasks that poll the agent's job and append progress events. Job ids
  are stored so `resume()` can re-attach after the hub restarts.
* The hub never edits requirements, writes SME answers, redrafts on its own, or changes workspaces.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from collections.abc import Awaitable, Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .auth import AGENT_IDS, Auth
from .clients import OPEN_LINKS, TENANT, WORKSPACES, AgentDown, DealClient, RfpClient, UpstreamError
from .intents import CANCEL, Intent, detect, parse_reason
from .llm import LLMError
from . import filepeek
from .planner import Planner, option_plan, plan_to_intent
from .router import route
from .store import ALLOWED_EXTENSIONS, Store, Upload, UploadRejected

log = logging.getLogger("agent_hub.engine")

TERMINAL_JOB = {"completed", "failed", "cancelled", "interrupted"}
# Button types that start a background run; only one run per conversation at a time.
RUN_ACTIONS = {"deal_run", "rfp_extract", "rfp_draft", "mem_proposal"}
EXAMPLE_PROMPTS = (
    "How many deals do I have?", "Create a new deal from these call notes", "Why do we lose deals?",
    "What have we said about SSO before?", "Answer this RFP or security questionnaire",
)


OPERATOR_PROMPTS = ("How many companies are using this?", "Show usage", "How many model calls were used this week?")


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def cost_line(calls: int, parts: str | None = None) -> str:
    extra = f" ({parts})" if parts else ""
    return f"This will use {plural(calls, 'model call')}{extra}."


CATALOGUE_DEALS, CATALOGUE_PROJECTS = 40, 20  # the most recent; labels only


@dataclass
class Prepared:
    accepted: list[Upload] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    from_button: bool = False  # the words of one of our own buttons: the keyword reader is built for them, no model needed


class Engine:
    def __init__(
        self, store: Store, deal_client: DealClient, rfp_client: RfpClient, registry: list[dict],
        poll_interval: float = 1.5, max_down_polls: int = 40, ask_workspace: bool = False, planner: Planner | None = None,
        auth: Auth | None = None,
    ) -> None:
        self.auth = auth  # who may use the hub; None = no sign-in
        self.planner = planner  # reads each message with the hub's language model; None = keyword matching only
        self.ask_workspace = ask_workspace  # ask which workspace to use the first time a chat needs an agent that has several
        self.store, self.deal, self.rfp, self.registry = store, deal_client, rfp_client, registry
        self.agents = {a["id"]: a for a in registry}
        self.poll_interval, self.max_down_polls = poll_interval, max_down_polls
        self._tasks: dict[str, set[asyncio.Task]] = {}
        self._runs: set[str] = set()  # conversations with a confirmed run in flight
        self._locks: dict[str, asyncio.Lock] = {}

    # -- events -----------------------------------------------------------------------------------

    def emit(self, conv: str, kind: str = "text", *, role: str = "assistant", **fields: Any) -> dict[str, Any]:
        return self.store.add_event(conv, role, kind, **fields)

    def say(self, conv: str, text: str, *, cards: list[dict] | None = None, actions: list[dict] | None = None) -> dict:
        if cards and not OPEN_LINKS.get():  # a client account is never sent to an agent's own, unauthenticated screen
            cards = [c for c in cards if not c.get("links")]
        return self.emit(conv, "cards" if cards else "text", text=text, cards=cards or None, actions=actions)

    def error(self, conv: str, text: str, *, actions: list[dict] | None = None) -> dict:
        return self.emit(conv, "error", text=text, actions=actions)

    def fail(self, conv: str, exc: Exception) -> None:
        """A friendly error event for an agent that is down or refused."""
        if isinstance(exc, AgentDown):
            self.error(conv, exc.message)
        elif isinstance(exc, UpstreamError):
            self.error(conv, f"{exc.message} {exc.action or ''}".strip())
        else:
            self.error(conv, f"Something unexpected went wrong on my side ({type(exc).__name__}). Nothing was changed by that step.")

    def report(self, conv: str, exc: Exception) -> None:
        if not isinstance(exc, (AgentDown, UpstreamError)):
            log.exception("flow failed")
        self.fail(conv, exc)

    def link_card(self, agent_id: str, label: str, suffix: str = "", *, title: str | None = None, text: str | None = None) -> dict:
        client = self.deal if agent_id == "deals" else self.rfp
        card: dict[str, Any] = {"title": title or label, "tone": "info", "links": [{"label": label, "url": client.open_url + suffix}]}
        if text:
            card["sections"] = [{"text": text}]
        return card

    # -- state ------------------------------------------------------------------------------------

    @contextmanager
    def scope(self, st: dict):  # noqa: ANN201
        """Work for this conversation inside the workspaces it chose: the clients send them as X-Workspace, so
        nothing is switched in the agents. Tasks started inside keep the workspace they started in."""
        token = WORKSPACES.set(dict(st.get("workspaces") or {}))
        tenant = TENANT.set((st.get("tenant") or {}).get("allowed") if st.get("tenant") is not None else None)
        links = OPEN_LINKS.set(st.get("tenant") is None or bool(st["tenant"].get("admin")))
        try:
            yield
        finally:
            OPEN_LINKS.reset(links)
            TENANT.reset(tenant)
            WORKSPACES.reset(token)

    @staticmethod
    def use_workspaces(st: dict) -> None:
        """After a flow changed st["workspaces"], make the rest of this step use the new choice."""
        WORKSPACES.set(dict(st.get("workspaces") or {}))

    def _tenant(self, conv: str, st: dict) -> None:
        """Which company this chat works for, from who owns it (never from the chat's own words). With sign-in on, the
        company's workspaces are all the chat may ever use; a company with exactly one workspace per agent is placed
        in it. A chat with no valid owner gets no workspaces at all."""
        if self.auth is None or self.auth.mode() != "on":
            st.pop("tenant", None)
            return
        user = self.auth.user_by_id(self.store.conversation_owner(conv))
        if user is None:
            st["tenant"] = {"company": None, "allowed": {a: [] for a in AGENT_IDS}}
            return
        allowed = {a: list(user.allowed(a)) for a in AGENT_IDS}
        if user.is_operator:  # the hub's own administrators: no company data at all, whatever their team record says
            allowed = {a: [] for a in AGENT_IDS}
        st["tenant"], st["company_name"] = ({"company": user.company_name, "allowed": allowed, "admin": user.is_admin,
                                             "operator": user.is_operator}, user.company_name)
        chosen, names = st.setdefault("workspaces", {}), st.setdefault("workspace_names", {})
        for agent_id in AGENT_IDS:
            if chosen.get(agent_id) and chosen[agent_id] not in allowed[agent_id]:
                chosen.pop(agent_id, None)
                names.pop(agent_id, None)
            if not chosen.get(agent_id) and len(allowed[agent_id]) == 1:
                chosen[agent_id] = allowed[agent_id][0]

    def load(self, conv: str) -> dict[str, Any]:
        st = self.store.get_state(conv)
        self._tenant(conv, st)
        for key, default in (("uploads", []), ("actions", {}), ("downloads", {})):
            st.setdefault(key, default)
        return st

    def save(self, conv: str, st: dict[str, Any]) -> None:
        self.store.set_state(conv, st)

    def lock(self, conv: str) -> asyncio.Lock:
        return self._locks.setdefault(conv, asyncio.Lock())

    # -- action tokens ----------------------------------------------------------------------------

    def offer(
        self, st: dict, type_: str, label: str, *, style: str = "primary", cost: str | None = None,
        data: dict | None = None, confirm: bool = False, group: str | None = None,
    ) -> dict[str, Any]:
        """Register a one-shot button and return it in the event shape."""
        token = secrets.token_hex(6)
        st["actions"][token] = {
            "type": type_, "data": data or {}, "status": "open", "confirm": confirm, "group": group, "label": label,
        }
        action: dict[str, Any] = {"id": token, "label": label, "style": style}
        if cost:
            action["cost"] = cost
        return action

    def confirmation(
        self, conv: str, st: dict, text: str, type_: str, data: dict, *, calls: int, cost_note: str | None = None,
        label: str = "Yes, go ahead", cost_label: str | None = None, also: tuple[str, dict] | None = None,
        extra: list[dict] | None = None,
    ) -> None:
        """Ask before doing something that spends model calls or changes memory."""
        self.expire_confirms(st, keep=data.get("upload_id"))
        group = secrets.token_hex(3) if also else None
        cost = cost_label or plural(calls, "model call")
        yes = self.offer(st, type_, label, cost=cost, data={**data, "calls": calls}, confirm=True, group=group)
        token = yes["id"]
        others = [self.offer(st, type_, also[0], style="ghost", cost=cost, data={**data, **also[1], "calls": calls},
                             confirm=True, group=group)] if also else []
        st["actions"][f"cancel:{token}"] = {"type": "cancel", "data": {"of": token}, "status": "open", "confirm": True, "label": "Not now"}
        st["pending"] = {"token": token, "step": type_, "cost": yes.get("cost")}
        self.say(conv, f"{text} {cost_note}".strip() if cost_note else text,
                 actions=[yes, *others, *(extra or []), {"id": f"cancel:{token}", "label": "Not now", "style": "ghost"}])

    def expire_confirms(self, st: dict, keep: str | None = None) -> None:
        for rec in st["actions"].values():
            if rec.get("confirm") and rec["status"] == "open":
                rec["status"] = "expired"
                self._release_upload(st, rec, keep)
        st["pending"] = None

    @staticmethod
    def _release_upload(st: dict, rec: dict, keep: str | None = None) -> None:
        """A file offered for a step that was declined or left behind stops waiting, so a later request never picks it up."""
        upload_id = (rec.get("data") or {}).get("upload_id")
        if upload_id and upload_id != keep:
            st["uploads"] = [u for u in st.get("uploads", []) if u != upload_id]

    def approve(self, st: dict, calls: int) -> None:
        st["budget"] = {"approved": calls, "spent": 0}

    def spend(self, st: dict, calls: int = 1) -> bool:
        """Count a model call against what the person approved; False means 'do not make it'."""
        budget = st.get("budget") or {"approved": 0, "spent": 0}
        if budget["spent"] + calls > budget["approved"]:
            return False
        budget["spent"] += calls
        st["budget"] = budget
        return True

    # -- background tasks -------------------------------------------------------------------------

    def spawn(self, conv: str, coro: Awaitable[None]) -> asyncio.Task:
        task = asyncio.ensure_future(self._safe(conv, coro))
        tasks = self._tasks.setdefault(conv, set())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return task

    async def _safe(self, conv: str, coro: Awaitable[None]) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a flow bug must end as a chat message, never a dead task
            self.report(conv, exc)

    def spawn_run(self, conv: str, runner: Callable[[Engine, str], Awaitable[None]]) -> asyncio.Task:
        async def guarded() -> None:
            try:
                await runner(self, conv)
            except asyncio.CancelledError:
                raise
            except BaseException:
                async with self.lock(conv):
                    st = self.load(conv)
                    st["run"] = None
                    self.save(conv, st)
                raise

        self._runs.add(conv)
        task = self.spawn(conv, guarded())
        task.add_done_callback(lambda _t: self._runs.discard(conv))
        return task

    def run_active(self, conv: str) -> bool:
        return conv in self._runs

    def busy(self, conv: str) -> bool:
        return any(not t.done() for t in self._tasks.get(conv, ()))

    async def wait_idle(self, conv: str | None = None) -> None:
        while True:
            pending = [t for c, ts in self._tasks.items() if conv in (None, c) for t in ts if not t.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def shutdown(self) -> None:
        """Stop background tasks without touching job rows, so a restart can resume them."""
        tasks = [t for ts in self._tasks.values() for t in ts]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def poll(
        self, conv: str, *, app: str, step: str, job_id: Any, fetch: Callable[[], Awaitable[dict]],
        done: Callable[[dict], bool], progress: Callable[[dict], tuple[int, int]], label: str,
        final_status: Callable[[dict], str] = lambda d: "completed",
    ) -> dict:
        """Poll until `done`, appending a progress event whenever the numbers change."""
        last: tuple[int, int] | None = None
        failures = 0
        while True:
            try:
                data = await fetch()
                failures = 0
            except AgentDown:
                failures += 1
                if failures >= self.max_down_polls:
                    self.store.finish_job(conv, app, job_id, "lost")
                    raise
                await asyncio.sleep(self.poll_interval)
                continue
            current = progress(data)
            if current != last:
                self.emit(conv, "progress", progress={"label": label, "done": current[0], "total": current[1]})
                last = current
            if done(data):
                self.store.finish_job(conv, app, job_id, final_status(data))
                return data
            await asyncio.sleep(self.poll_interval)

    # -- conversation entry points ---------------------------------------------------------------

    def new_conversation(self, user: Any = None) -> str:
        conv = self.store.create_conversation(user.id if user is not None else None)
        default = {} if user is not None else (self.store.get_setting("company") or {})
        if default.get("workspaces"):  # a saved default: every new chat starts in the company's own workspaces
            st = self.load(conv)
            st.update(workspaces=dict(default["workspaces"]), workspace_names=dict(default.get("workspace_names") or {}),
                      company_name=default.get("name"))
            self.save(conv, st)
        if user is not None and getattr(user, "is_operator", False):
            self.say(conv, "Hi, I'm the hub. You're signed in as an administrator: this chat holds no company data. Ask me how the "
                           "hub is used (companies, people, model calls, tokens, data stored), or open the admin page to add companies "
                           "and people. Try one of these:",
                     actions=[{"id": f"say:{p}", "label": p, "style": "ghost"} for p in OPERATOR_PROMPTS])
            return conv
        actions = [{"id": f"say:{p}", "label": p, "style": "ghost"} for p in EXAMPLE_PROMPTS]
        self.say(
            conv,
            "Hi, I'm the hub. Tell me what you need and I'll pick the right agent and do the work here: "
            "attach a file and I'll set it up, run the steps and show you the result. I always tell you "
            "before anything uses model calls. Try one of these:",
            actions=actions,
        )
        return conv

    # -- understanding ----------------------------------------------------------------------------

    def _history(self, conv: str, limit: int = 8) -> list[tuple[str, str]]:
        """The last few turns as plain text (card titles stand in for cards), without the message being understood."""
        events = self.store.events(conv, max(0, self.store.last_n(conv) - limit - 1))
        turns: list[tuple[str, str]] = []
        for e in events:
            if e["kind"] == "progress":
                continue
            parts = [e.get("text") or ""] + [c.get("title", "") for c in e.get("cards") or []]
            text = " ".join(p for p in parts if p).strip()
            if text:
                turns.append(("user" if e["role"] == "user" else "assistant", text[:240]))
        return turns[:-1] if turns and turns[-1][0] == "user" else turns

    def _basic_mode(self, conv: str, st: dict, why: str) -> None:
        if st.get("basic_mode_noted"):
            return
        st["basic_mode_noted"] = True
        self.say(conv, f"{why} Until that works I understand you with simple phrase matching: common requests like "
                       "\"brief me on Cedarline\" still work, but loose wording may not. To turn the chatbot on, put a Gemini key "
                       "(GEMINI_API_KEY) in agent-hub/.env.")

    async def _understand(self, conv: str, st: dict, text: str, prep: Prepared, keyword: Intent) -> Intent | None:
        """What the person wants, from the language model. None means \"use the keyword reading\" (no model, a failure,
        or the per-chat cap), and the chat is told once why."""
        planner = self.planner
        if planner is None or not planner.available():
            self._basic_mode(conv, st, "I don't have a language model set up.")
            return None
        used = int(st.get("router_calls", 0))
        if used >= planner.cap:
            self._basic_mode(conv, st, f"This chat has used its {planner.cap} understanding calls.")
            return None
        st["router_calls"] = used + 1
        used_tokens: list[tuple[str | None, int, int]] = []
        files = [filepeek.describe(u) for u in self.pending_uploads(st)]
        try:
            plan = await planner.plan(text, st, self._history(conv), files, usage=used_tokens, catalogue=await self._catalogue())
        except LLMError as exc:
            self._record_understanding(conv, used_tokens, ok=False)
            self._basic_mode(conv, st, exc.message)
            return None
        except Exception:  # noqa: BLE001 - never lose the message because the understanding step broke
            log.exception("planner failed")
            self._record_understanding(conv, used_tokens, ok=False)
            self._basic_mode(conv, st, "The language model had a problem.")
            return None
        self._record_understanding(conv, used_tokens, ok=True)
        intent = plan_to_intent(plan, text, (st.get("awaiting") or {}).get("type"))
        if intent.kind == "slot" and not st.get("awaiting"):
            return None  # "answer to my question" when nothing was asked: keep the keyword reading
        return intent

    async def _catalogue(self) -> list[str] | None:
        """The company's own deals and RFP projects as labels (code, name, account, status), so the model can resolve loose
        names and typos. Only this company's data (the clients are scoped to it), capped, and off with HUB_PLANNER_DATA=off."""
        if os.environ.get("HUB_PLANNER_DATA", "on").strip().lower() in ("off", "0", "false", "no"):
            return None
        lines: list[str] = []
        try:
            deals = sorted(await self.deal.list_deals(), key=lambda d: str(d.get("last_activity") or d.get("opened_on") or ""), reverse=True)
            lines += [f"deal {d.get('code')}: {d.get('name')} | account: {d.get('account')} | {d.get('result')}" for d in deals[:CATALOGUE_DEALS]]
        except Exception:  # noqa: BLE001 - an agent that is down or not set up for this company just isn't listed
            pass
        try:
            projects = sorted(await self.rfp.list_projects(), key=lambda p: str(p.get("created_at") or ""), reverse=True)
            lines += [f"rfp project {p.get('id')}: {p.get('name')} | {p.get('state')}" for p in projects[:CATALOGUE_PROJECTS]]
        except Exception:  # noqa: BLE001
            pass
        return lines or None

    async def _operator_reply(self, conv: str, st: dict, intent: Intent, has_files: bool) -> None:
        """Administrators run the hub; they hold no company data. Their chat answers usage questions from records and points
        everything else to the admin page or to a company's own account."""
        st["awaiting"] = None
        if intent.kind == "admin_usage":
            from .flows import usage as usage_flow

            return await usage_flow.show(self, conv, st)
        ask = [{"id": "say:How many companies are using this?", "label": "How many companies are using this?", "style": "ghost"},
               {"id": "say:Show usage", "label": "Show usage", "style": "ghost"}]
        prefix = "I don't keep files for administrators. " if has_files else ""
        self.say(conv, prefix + "You're signed in as a hub administrator, so this chat has no company data: deals, RFPs and their "
                 "memory belong to each company. I can tell you how the hub is used (companies, people, model calls, tokens, data "
                 "stored). To add companies or people, open the admin page. To work with a company's data, sign in with an account "
                 "of that company.", cards=[{"title": "Admin page", "links": [{"label": "Open the admin page", "url": "/admin"}]}],
                 actions=ask)

    def _record_understanding(self, conv: str, used: list[tuple[str | None, int, int]], *, ok: bool) -> None:
        """Count a call the hub made to read a message, against the company that owns the chat."""
        owner = self.store.conversation_owner(conv)
        user = self.auth.user_by_id(owner) if self.auth is not None and owner else None
        model, tokens_in, tokens_out = used[-1] if used else (None, 0, 0)
        self.store.record_usage(user.company_id if user else None, owner, "understand", model, tokens_in, tokens_out, ok)

    def _prepare(self, conv: str, text: str, files: Iterable[tuple[str, bytes]]) -> tuple[int, Prepared]:
        prep = Prepared()
        for name, data in files:
            try:
                prep.accepted.append(self.store.save_upload(conv, name, data))
            except UploadRejected as exc:
                prep.rejected.append((name, exc.message))
        shown = [{"name": u.filename, "size": u.size} for u in prep.accepted] + [
            {"name": n, "size": 0} for n, _ in prep.rejected
        ]
        event = self.emit(conv, "text", role="user", text=text or None, files=shown or None)
        return event["n"], prep

    def accept_message(self, conv: str, text: str, files: Iterable[tuple[str, bytes]] = ()) -> int:
        """Store the user's message now and process it in the background. Returns the event number."""
        n, prep = self._prepare(conv, text, files)
        self.spawn(conv, self._process(conv, text, prep))
        return n

    async def handle_message(self, conv: str, text: str, files: Iterable[tuple[str, bytes]] = (), *, from_button: bool = False) -> None:
        """Store and process a message; returns when this message is handled (background polls may continue)."""
        _, prep = self._prepare(conv, text, files)
        prep.from_button = from_button
        await self._process(conv, text, prep)

    def accept_action(self, conv: str, action_id: str) -> None:
        self.spawn(conv, self.handle_action(conv, action_id))

    async def _process(self, conv: str, text: str, prep: Prepared) -> None:
        async with self.lock(conv):
            st = self.load(conv)
            try:
                with self.scope(st):
                    await self._dispatch(conv, st, text, prep)
            except Exception as exc:  # noqa: BLE001 - always end as a chat message; the finally saves what was done
                self.report(conv, exc)
            finally:
                self.save(conv, st)

    # -- dispatch ---------------------------------------------------------------------------------

    async def _dispatch(self, conv: str, st: dict, text: str, prep: Prepared) -> None:
        from .flows import deal, rfp

        if prep.rejected:
            lines = "; ".join(f"{name}: {why}" for name, why in prep.rejected)
            self.say(conv, f"I couldn't use {'that file' if len(prep.rejected) == 1 else 'those files'}. {lines}")
        st["uploads"] = st["uploads"] + [u.id for u in prep.accepted]
        if not text and not prep.accepted:
            return
        awaiting = st.get("awaiting") or {}
        if awaiting.get("type") == "rfp_final_text" and text.strip() and not prep.accepted:
            # the final answer for a draft under review: taken word for word, never read by a model or as a command
            from .flows import review

            if CANCEL.match(" ".join(text.split())):
                st["awaiting"] = None
                return self.say(conv, f"Okay, {awaiting.get('code') or 'that answer'} is left as it was.")
            return await review.handle_text(self, conv, st, text)
        intent = detect(text, bool(prep.accepted), st, [u.filename for u in prep.accepted])
        kind = intent.kind
        if (st.get("tenant") or {}).get("operator"):  # an administrator's chat: admin questions only, never a model, never company data
            return await self._operator_reply(conv, st, intent, bool(prep.accepted))

        if kind == "confirm":
            return await self._on_confirm(conv, st)
        if kind == "cancel":
            return self._on_cancel(conv, st)
        if kind in ("mem_facts", "mem_plays", "mem_proposal"):  # feeding the memory: shown first, saved only on a click
            from .flows import memory_feed

            st["awaiting"] = None
            self.expire_confirms(st)
            return await memory_feed.start(self, conv, st, intent, text)
        if kind == "admin_usage":  # answered from records, never by a model
            from .flows import usage as usage_flow

            st["awaiting"] = None
            return await usage_flow.show(self, conv, st)
        clear_answer = kind == "slot" and awaiting.get("type") == "loss_reason" and parse_reason(text) is not None
        if self.planner is not None and (text or prep.accepted) and not prep.from_button and not clear_answer:
            # files sent with no words are understood from their labels; a button's own words and a clear answer to the
            # question just asked are read as they are, so the model can't turn them into a different request
            understood = await self._understand(conv, st, text, prep, intent)
            if understood is not None:
                if intent is not None and understood.kind == intent.kind:
                    # the keyword reading can hold details the model left out (an industry, a segment, a loss reason): keep them
                    understood.slots = {**intent.slots, **{k: v for k, v in understood.slots.items() if v not in (None, "")}}
                    for name in ("subject", "result", "loss_reason", "target", "fmt"):
                        if getattr(understood, name) in (None, "") and getattr(intent, name) not in (None, ""):
                            setattr(understood, name, getattr(intent, name))
                intent, kind = understood, understood.kind
        if kind == "slot":
            awaiting_type = str((st.get("awaiting") or {}).get("type", ""))
            if awaiting_type.startswith("mem_"):
                from .flows import memory_feed

                return await memory_feed.handle_slot(self, conv, st, intent, prep.accepted, text)
            module = rfp if awaiting_type.startswith("rfp_") else deal
            return await module.handle_slot(self, conv, st, intent, prep.accepted)

        return await self._act(conv, st, intent, prep, text)

    async def _act(self, conv: str, st: dict, intent: Intent, prep: Prepared, text: str) -> None:
        """Carry out a request: the reading of a message, or an option the person clicked."""
        from .flows import deal, rfp

        kind = intent.kind
        st["awaiting"] = None  # a new request replaces whatever was being asked
        self.expire_confirms(st)
        if kind == "choose":
            return self._offer_choices(conv, st, intent)
        if kind in ("mem_facts", "mem_plays", "mem_proposal"):
            from .flows import memory_feed

            return await memory_feed.start(self, conv, st, intent, text)
        if kind == "admin_usage":
            from .flows import usage as usage_flow

            return await usage_flow.show(self, conv, st)
        if kind == "llm_reply":
            return self.say(conv, intent.slots.get("message") or "Could you say that another way?")
        if kind == "greeting":
            return self._help(conv)
        if kind == "none":
            return self._help(conv, "I'm not sure which agent fits that. ")
        if kind == "route":
            return self._route_reply(conv, intent)
        if kind in ("workspace_list", "workspace_use", "workspace_save", "workspace_forget"):
            from .flows import workspace

            handler = {"workspace_list": workspace.show, "workspace_use": workspace.use,
                       "workspace_save": workspace.remember, "workspace_forget": workspace.forget}[kind]
            return await handler(self, conv, st, intent)
        if self.ask_workspace:
            from .flows import workspace as workspace_flow

            if await workspace_flow.gate(self, conv, st, intent):
                return
        if kind == "files_unclear":
            return self._ask_file_kind(conv, st)
        if kind == "deal_files":
            return await deal.files_for_deal(self, conv, st, intent)
        if kind == "deal_ask":
            return await deal.deal_ask(self, conv, st, intent)
        if kind == "company_ask":
            from .flows import company

            await company.company_question(self, conv, st, intent, targets=intent.slots["targets"])
            return
        if kind in ("deal_count", "deal_list"):
            from .flows import company

            if await company.deal_count(self, conv, st, intent):
                return
            intent = Intent("question", agent="deals", text=intent.text)
            kind = "question"
        if kind == "question":
            if await deal.deal_question(self, conv, st, intent):
                return
            from .flows import company

            if await company.company_question(self, conv, st, intent):
                return
            routed = route(text, self.registry)
            if routed.kind == "greeting":
                return self._help(conv)
            if routed.kind == "none":
                return self._help(conv, "I'm not sure which agent fits that. ")
            return self._route_reply(conv, Intent("route", text=text))
        if kind in ("deal_brief", "deal_read", "deal_update", "deal_new", "deal_add", "deal_note", "deal_followup"):
            return await getattr(deal, kind)(self, conv, st, intent, prep.accepted)
        if kind == "outcome":
            return await self._outcome(conv, st, intent)
        if kind in ("rfp_start", "rfp_accept_all", "rfp_export", "rfp_review"):
            return await getattr(rfp, kind)(self, conv, st, intent)

    async def _outcome(self, conv: str, st: dict, intent: Intent) -> None:
        from .flows import deal, rfp

        target = intent.target
        if target is None:
            has_deal, has_project = bool(st.get("deal_id")), bool(st.get("project_id"))
            target = "rfp" if has_project and (not has_deal or st.get("flow") == "rfp") else "deal"
        await (rfp.outcome if target == "rfp" else deal.outcome)(self, conv, st, intent)

    def _help(self, conv: str, prefix: str = "") -> None:
        actions = [{"id": f"say:{p}", "label": p, "style": "ghost"} for p in EXAMPLE_PROMPTS]
        self.say(
            conv,
            prefix + "Here is what I can do right now. Company-wide questions: ask how you do across all your deals (\"why do we lose fintech deals?\") "
            "or what you have told clients before (\"what have we said about SOC 2?\"). Workspaces: say \"show workspaces\" or \"use the <name> workspace\" to choose which "
            "workspace this chat works in. Deals: brief you on a deal, answer questions about it, draft a follow-up "
            "email or call agenda, create one from emails and notes, add files to it, and record how it ended. RFPs: attach an RFP or questionnaire and I'll create the project, "
            "draft the answers, and give you the finished document. Try one of these:",
            actions=actions,
        )

    def _route_reply(self, conv: str, intent: Intent) -> None:
        routing = route(intent.text, self.registry)
        agent = routing.best.agent
        if agent["status"] != "live":
            closest = next((m for m in routing.alternatives if m.agent["status"] == "live"), None)
            text = f"That sounds like {agent['name']} (agent {agent['number']}), which isn't built yet."
            if closest:
                text += f" The closest agent available now is {closest.agent['name']}."
            return self._help(conv, text + " ")
        if agent["id"] == "deals":
            return self.say(
                conv,
                "That's a Deal Intelligence question. Tell me which deal and what you need, for example \"Brief me on the "
                "<deal name> deal\", \"We lost <deal name> because of price\", or attach emails or call notes and say \"New deal "
                "<deal name> at <company>\".",
                cards=[self.link_card("deals", "Open Deal Intelligence")],
            )
        self.say(
            conv,
            "That's an RFP question. Attach the RFP or questionnaire (.docx, .xlsx, .pdf, .txt or .md) and I'll set it up "
            "and draft the answers, asking before anything uses model calls.",
            cards=[self.link_card("rfp", "Open the RFP assistant")],
        )

    def _offer_choices(self, conv: str, st: dict, intent: Intent) -> None:
        """The model wasn't sure what the person meant: each reading is a button, and nothing happens until one is clicked."""
        group = secrets.token_hex(3)
        actions = [self.offer(st, "plan_choice", " ".join(str(o.get("label") or "This one").split())[:80],
                              style="primary" if i == 0 else "ghost", data={"option": o, "text": intent.text}, group=group)
                   for i, o in enumerate(intent.slots.get("options") or [])]
        self.say(conv, intent.slots.get("message") or "I can do a few things with that. Which do you mean?", actions=actions)

    def _ask_file_kind(self, conv: str, st: dict) -> None:
        group = secrets.token_hex(3)
        actions = [
            self.offer(st, "route_files", "It's an RFP or questionnaire", style="ghost", data={"kind": "rfp"}, group=group),
            self.offer(st, "route_files", "It's for a deal", style="ghost", data={"kind": "deal"}, group=group),
        ]
        self.say(conv, "Got the file. What is it?", actions=actions)

    # -- confirm / cancel / actions --------------------------------------------------------------

    async def _on_confirm(self, conv: str, st: dict) -> None:
        pending = st.get("pending")
        if pending:
            return await self._fire(conv, st, pending["token"])
        awaiting = st.get("awaiting") or {}
        if awaiting.get("type") == "deal_choice" and len(awaiting.get("candidates", [])) == 1:
            from .flows import deal

            return await deal.pick(self, conv, st, awaiting["candidates"][0]["id"], awaiting.get("then") or {})
        self.say(conv, "Nothing is waiting for a yes. Tell me what you'd like to do.")

    def _on_cancel(self, conv: str, st: dict) -> None:
        pending = st.get("pending")
        if pending:
            return self._cancel_token(conv, st, pending["token"])
        if st.get("awaiting"):
            st["awaiting"] = None
            return self.say(conv, "Okay, dropped that. Nothing was changed.")
        self.say(conv, "There's nothing to cancel right now.")

    def _cancel_token(self, conv: str, st: dict, token: str) -> None:
        rec = st["actions"].get(token)
        if rec and rec["status"] == "open":
            rec["status"] = "cancelled"
            self._release_upload(st, rec)
            if (st.get("pending") or {}).get("token") == token:
                st["pending"] = None
            self.say(conv, "No problem. I haven't done anything.")
        elif rec and rec["status"] == "cancelled":
            self.say(conv, "That was already cancelled.")
        else:
            self.say(conv, "That was already done.")

    async def handle_action(self, conv: str, action_id: str) -> None:
        if action_id.startswith("say:"):
            return await self.handle_message(conv, action_id[4:], from_button=True)
        async with self.lock(conv):
            st = self.load(conv)
            try:
                with self.scope(st):
                    if action_id.startswith("cancel:"):
                        token = action_id[7:]
                        self.emit(conv, "text", role="user", text="Not now")
                        self._cancel_token(conv, st, token)
                    else:
                        rec = st["actions"].get(action_id)
                        if rec:
                            self.emit(conv, "text", role="user", text=rec.get("label"))
                        await self._fire(conv, st, action_id)
            except Exception as exc:  # noqa: BLE001
                self.report(conv, exc)
            finally:
                self.save(conv, st)

    async def _fire(self, conv: str, st: dict, token: str) -> None:
        from .flows import deal, rfp

        rec = st["actions"].get(token)
        if rec is None:
            return self.say(conv, "That option isn't available any more. Ask me again and I'll set it up.")
        status = rec["status"]
        if status == "used":
            return self.say(conv, "That was already done.")
        if status == "cancelled":
            return self.say(conv, "That was cancelled. Ask me again if you want to go ahead.")
        if status == "expired":
            return self.say(conv, "That option has expired because we moved on. Ask me again and I'll set it up.")
        if rec["type"] in RUN_ACTIONS and self.run_active(conv):
            # Not consumed: the person can press it again once the running job is done.
            return self.say(conv, "I'm still working on the previous request. Ask again when it has finished.")
        rec["status"] = "used"  # before running, so a failure or a double click can never repeat it
        if rec.get("group"):
            for other in st["actions"].values():
                if other.get("group") == rec["group"] and other["status"] == "open":
                    other["status"] = "expired"
        if (st.get("pending") or {}).get("token") == token:
            st["pending"] = None
        kind = rec["type"]
        if kind == "plan_choice":
            said = rec["data"].get("text") or ""
            return await self._act(conv, st, plan_to_intent(option_plan(rec["data"]["option"]), said), Prepared(), said)
        if kind == "route_files":
            return await self._route_files(conv, st, rec["data"]["kind"])
        if kind.startswith("ws_"):
            from .flows import workspace

            return await workspace.on_action(self, conv, st, kind, rec["data"])
        if kind.startswith("company_"):
            from .flows import company

            return await company.on_action(self, conv, st, kind, rec["data"])
        if kind.startswith("mem_"):
            from .flows import memory_feed

            return await memory_feed.on_action(self, conv, st, kind, rec["data"])
        module = rfp if kind.startswith("rfp_") else deal
        await module.on_action(self, conv, st, kind, rec["data"])

    async def _route_files(self, conv: str, st: dict, kind: str) -> None:
        from .flows import deal, rfp

        if kind == "rfp":
            return await rfp.rfp_start(self, conv, st, Intent("rfp_start", agent="rfp"))
        await deal.files_for_deal(self, conv, st, Intent("deal_files", agent="deals"))

    # -- uploads ----------------------------------------------------------------------------------

    def pending_uploads(self, st: dict, extensions: Iterable[str] | None = None) -> list[Upload]:
        found = [u for u in (self.store.get_upload(i) for i in st["uploads"]) if u]
        if extensions is not None:
            wanted = tuple(extensions)
            found = [u for u in found if u.ext in wanted]
        return found

    @staticmethod
    def file_types() -> str:
        return ", ".join(ALLOWED_EXTENSIONS)

    # -- downloads --------------------------------------------------------------------------------

    def new_download(self, st: dict, project_id: int, fmt: str) -> str:
        token = secrets.token_urlsafe(12)
        st["downloads"][token] = {"project_id": project_id, "format": fmt, "workspace": (st.get("workspaces") or {}).get("rfp")}
        return token

    async def download(self, conv: str, token: str) -> tuple[bytes, str, str]:
        """Proxy the finished response from the RFP assistant. KeyError when the token is unknown."""
        st = self.load(conv)
        record = st["downloads"][token]
        # Project ids belong to a workspace: fetch from the one the project was made in, not the chat's current choice.
        with self.scope(st):
            mark = WORKSPACES.set({"rfp": record["workspace"]} if record.get("workspace") else {})
            try:
                return await self.rfp.export(record["project_id"], record["format"])
            finally:
                WORKSPACES.reset(mark)

    # -- restart ----------------------------------------------------------------------------------

    async def resume(self) -> int:
        """Re-attach to jobs that were running when the hub stopped. Returns how many conversations resumed."""
        from .flows import deal, rfp

        resumed = 0
        for conv in sorted({row["conversation_id"] for row in self.store.running_jobs()}):
            async with self.lock(conv):
                st = self.load(conv)
                run = st.get("run")
                if not run:
                    for row in self.store.running_jobs():
                        if row["conversation_id"] == conv:
                            self.store.finish_job(conv, row["app"], row["upstream_job_id"], "abandoned")
                    continue
            # A *_start step may or may not have reached the agent before the stop, so it must not be repeated.
            step = str(run.get("step", ""))
            run["resumed"] = step.endswith("_start") and step != "start" and run.get("job_id") is None
            async with self.lock(conv):
                st = self.load(conv)
                if st.get("run"):
                    st["run"]["resumed"] = run["resumed"]
                    self.save(conv, st)
            from .flows import memory_feed

            runner = {"rfp": rfp.run, "mem_proposal": memory_feed.run}.get(run.get("flow"), deal.run)
            self.emit(conv, "text", text="I'm back after a restart. Reconnecting to the job that was running.")
            with self.scope(self.load(conv)):  # the task keeps the workspace the job was started in
                self.spawn_run(conv, runner)
            resumed += 1
        return resumed
