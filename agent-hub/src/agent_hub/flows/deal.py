"""Deal Intelligence flows: find or create a deal, read it, brief it, add files, record how it ended."""

from __future__ import annotations

import re
import secrets
from typing import TYPE_CHECKING, Any

from ..clients import WorkspaceInfo, UpstreamError, describe_error_info
from ..engine import TERMINAL_JOB, cost_line, plural
from ..intents import (
    LOSS_REASON_WORDS, Candidate, Intent, company_targets, match_deal, mentioned_deals, parse_reason, pick_candidate, reason_matches,
    reply_slots, resolve_candidates,
)
from .. import filepeek
from ..router import route
from ..store import Upload
from . import workspace

if TYPE_CHECKING:
    from ..engine import Engine


def _label(deal: dict) -> str:
    return f"{deal.get('code')} {deal.get('name')} ({deal.get('account')})"


def _count(value: Any) -> int:
    return len(value) if isinstance(value, list) else int(value or 0)


def _ws_text(st: dict) -> str:
    name = (st.get("workspace") or {}).get("name")
    return f" (workspace: {name})" if name else ""


async def _workspace(eng: Engine, st: dict) -> WorkspaceInfo:
    try:
        ws = await eng.deal.status()
    except UpstreamError:
        ws = WorkspaceInfo(None, None)
    st["workspace"] = {"agent": "deals", "id": ws.id, "name": ws.name}
    return ws


async def _workspace_unchanged(eng: Engine, conv: str, expected_id: str | None) -> bool:
    """Re-check before spending or changing memory: the active workspace is global and could have been switched."""
    now = await eng.deal.status()
    if expected_id is not None and now.id != expected_id:
        eng.say(conv, f"The active Deal Intelligence workspace changed to \"{now.name or now.id}\" since I asked, so I stopped "
                      "without doing anything. Ask me again and I'll start from the current one.")
        return False
    return True


def _deal_link(eng: Engine, deal_id: int, title: str = "Open in Deal Intelligence") -> dict:
    return eng.link_card("deals", title, f"/{deal_id}")


# -- finding the deal ------------------------------------------------------------------------------


async def _with_deal(eng: Engine, conv: str, st: dict, subject: str | None, then: dict) -> None:
    await _workspace(eng, st)
    if subject is None:
        if st.get("deal_id"):
            return await _continue(eng, conv, st, await eng.deal.get_deal(st["deal_id"]), then)
        st["awaiting"] = {"type": "deal_ref", "then": then}
        return eng.say(conv, "Which deal do you mean? Give me its name, account or code (like D-004).")
    deals = await eng.deal.list_deals()
    verdict, picks = resolve_candidates(match_deal(subject, deals))
    if verdict == "one":
        return await _continue(eng, conv, st, await eng.deal.get_deal(picks[0].deal["id"]), then)
    if verdict == "ask":
        return _ask_which(eng, conv, st, picks, then)
    routed = route(then.get("text", ""), eng.registry)
    if then.get("kind") == "brief" and routed.kind == "match" and routed.best.agent["id"] != "deals":
        return eng._route_reply(conv, Intent("route", text=then["text"]))
    if await workspace.elsewhere(eng, conv, st, subject, match_deal, again=then.get("text") or subject):
        return
    known = ", ".join(f"{d['code']} {d['name']}" for d in deals[:6]) or "none yet"
    st["awaiting"] = {"type": "deal_ref", "then": then}
    eng.say(
        conv,
        f"I couldn't find a deal matching \"{subject}\" in Deal Intelligence{_ws_text(st)}. Deals I can see: {known}. "
        "Tell me which one, or say \"New deal <name> at <account>\" and attach the emails to create it.",
    )


def _ask_which(eng: Engine, conv: str, st: dict, picks: list[Candidate], then: dict) -> None:
    group = secrets.token_hex(3)
    st["awaiting"] = {
        "type": "deal_choice", "then": then,
        "candidates": [{"id": c.deal["id"], "label": _label(c.deal)} for c in picks],
    }
    actions = [
        eng.offer(st, "deal_pick", c.deal.get("code", ""), style="ghost", data={"deal_id": c.deal["id"], "then": then}, group=group)
        | {"label": _label(c.deal)}
        for c in picks
    ]
    names = " or ".join(_label(c.deal) for c in picks)
    eng.say(conv, f"Did you mean {names}?", actions=actions)


async def pick(eng: Engine, conv: str, st: dict, deal_id: int, then: dict) -> None:
    st["awaiting"] = None
    await _continue(eng, conv, st, await eng.deal.get_deal(deal_id), then)


async def _continue(eng: Engine, conv: str, st: dict, detail: dict, then: dict) -> None:
    st.update({"flow": "deal", "deal_id": detail["id"], "deal_label": _label(detail)})
    kind = then.get("kind")
    if kind == "brief":
        await _offer_brief(eng, conv, st, detail)
    elif kind == "read":
        await _offer_read(eng, conv, st, detail)
    elif kind == "update":
        await _update(eng, conv, st, detail, then)
    elif kind == "outcome":
        await _offer_outcome(eng, conv, st, detail, then)
    elif kind == "add":
        await _add_files(eng, conv, st, detail)
    elif kind == "note":
        await _add_note(eng, conv, st, detail, then)
    elif kind == "ask":
        await _offer_ask(eng, conv, st, detail, then)
    elif kind == "followup":
        await _offer_followup(eng, conv, st, detail, then)


# -- brief ------------------------------------------------------------------------------------------


async def deal_brief(eng: Engine, conv: str, st: dict, intent: Intent, uploads: list[Upload]) -> None:
    await _with_deal(eng, conv, st, intent.subject, {"kind": "brief", "text": intent.text})


async def _offer_brief(eng: Engine, conv: str, st: dict, detail: dict) -> None:
    need_signals = detail.get("signals_status") != "ready"
    label = _label(detail)
    if need_signals and not _count(detail.get("interactions")):
        return eng.say(
            conv,
            f"I found {label}{_ws_text(st)}, but it has no emails or notes yet, so there is nothing to brief from. "
            "Attach them and say \"add these to " + detail["name"] + "\", then ask me again.",
            cards=[_deal_link(eng, detail["id"])],
        )
    calls = 2 if need_signals else 1
    intro = f"I found {label} in Deal Intelligence{_ws_text(st)}. "
    if need_signals:
        intro += "Its emails and notes haven't been read yet, so I'd read them first and then write the brief."
        parts = "1 to read the emails and notes, 1 to write the brief"
    else:
        intro += "Its signals were already read, so I'll skip that step and just write the brief."
        parts = None
    eng.confirmation(
        conv, st, intro, "deal_run",
        {"deal_ids": [detail["id"]], "workspace_id": (st.get("workspace") or {}).get("id")},
        calls=calls, cost_note=cost_line(calls, parts),
    )


async def deal_update(eng: Engine, conv: str, st: dict, intent: Intent, uploads: list[Upload]) -> None:
    fields = {k: v for k, v in (intent.slots or {}).items() if k in ("industry", "segment", "name", "account") and v}
    _fill(fields, details_from_words(intent.text), "your message")
    fields.pop("from", None)
    await _with_deal(eng, conv, st, intent.subject, {"kind": "update", "text": intent.text, "fields": fields})


async def _update(eng: Engine, conv: str, st: dict, detail: dict, then: dict) -> None:
    fields = dict(then.get("fields") or {})
    if fields.get("segment"):
        fields["segment"] = normal_segment(fields["segment"])
    fields = {k: v for k, v in fields.items() if v}
    if not fields:
        return eng.say(conv, f"What should I change on {_label(detail)}? For example \"it's a small software company\" sets its "
                             "industry and size.")
    updated = await eng.deal.update_deal(detail["id"], **fields)
    said = ", ".join(f"{k} {str(v).replace('_', ' ')}" for k, v in fields.items())
    _audit(eng, conv, "deal.update", f"{detail['code']}: {said}")
    eng.say(conv, f"Updated {_label(updated)}: {said}. Similar-deal matching uses these from now on. This used no model calls.",
            cards=[_deal_link(eng, detail["id"])])


async def deal_read(eng: Engine, conv: str, st: dict, intent: Intent, uploads: list[Upload]) -> None:
    await _with_deal(eng, conv, st, intent.subject, {"kind": "read", "text": intent.text})


async def _offer_read(eng: Engine, conv: str, st: dict, detail: dict) -> None:
    """Read a deal's emails and notes (1 call) without writing a brief: what a closed deal needs before its outcome."""
    label = _label(detail)
    if not _count(detail.get("interactions")):
        return eng.say(conv, f"{label} has no emails or notes to read yet. Attach them and say \"add these to {detail['name']}\".",
                       cards=[_deal_link(eng, detail["id"])])
    if detail.get("signals_status") == "ready":
        return eng.say(conv, f"{label} has already been read. Here is what it found.", cards=[signals_card(detail), _deal_link(eng, detail["id"])])
    eng.confirmation(
        conv, st, f"I'll read {label}'s emails and notes{_ws_text(st)} for its objections, competitors, promises, people and the plays "
                  "used. No brief is written.", "deal_run",
        {"deal_ids": [detail["id"]], "workspace_id": (st.get("workspace") or {}).get("id"), "read_only": True},
        calls=1, label="Yes, read it",
    )


def signals_card(detail: dict) -> dict:
    """What reading a deal found: checked against the files before an outcome is recorded."""
    s = detail.get("signals") or {}
    sections: list[dict] = []
    objections = s.get("objections") or []
    sections.append({"heading": "Objections", "bullets": [f"{o.get('type')}: {o.get('text')} ({o.get('status')})" for o in objections]}
                    if objections else {"heading": "Objections", "text": "None found."})
    sections.append({"heading": "Competitors", "text": ", ".join(s.get("competitors") or []) or "None found."})
    plays = s.get("plays_used") or []
    sections.append({"heading": "Plays used", "text": ", ".join(f"{p.get('code')} {p.get('name')}" for p in plays) or "None found."})
    people = detail.get("stakeholders") or []
    if people:
        sections.append({"heading": "People", "bullets": [
            f"{p.get('name')}" + (f", {p['title']}" if p.get("title") else "") + f": {p.get('stance')}"
            + (", economic buyer" if p.get("economic_buyer") else "") + ("" if p.get("engaged", True) else ", not engaged")
            for p in people]})
    promises = s.get("promises") or []
    if promises:
        sections.append({"heading": "Promises", "bullets": [
            f"{p.get('text')}" + (f" (due {p['due_on']})" if p.get("due_on") else "") + f": {p.get('status')}" for p in promises]})
    flags = detail.get("flags") or []
    if flags:
        sections.append({"heading": "Flags (checked by code, no model)", "bullets": [f"{f.get('severity', 'info')}: {f.get('text')}" for f in flags]})
    return {"title": f"Read: {_label(detail)}", "subtitle": "Check this against the files before recording the outcome.",
            "tone": "info", "sections": sections}


async def _start_run(eng: Engine, conv: str, st: dict, data: dict) -> None:
    if not await _workspace_unchanged(eng, conv, data.get("workspace_id")):
        return
    eng.approve(st, data["calls"])
    st["run"] = {"flow": "deal", "queue": data["deal_ids"], "idx": 0, "step": "start", "job_id": None,
                 "read_only": bool(data.get("read_only"))}
    eng.spawn_run(conv, run)


async def run(eng: Engine, conv: str) -> None:
    while await _advance(eng, conv):
        pass


async def _advance(eng: Engine, conv: str) -> bool:
    """Do one step of the confirmed brief run. False when the run is finished."""
    async with eng.lock(conv):
        st = eng.load(conv)
        state = st.get("run")
        if not state:
            return False
        if state["idx"] >= len(state["queue"]):
            st["run"] = None
            eng.save(conv, st)
            return False
        deal_id, step = state["queue"][state["idx"]], state["step"]
        if step in ("signals_start", "brief_start") and state.get("job_id") is None and state.get("resumed"):
            st["run"] = None
            eng.save(conv, st)
            eng.say(conv, "I was interrupted just before starting that job, and I can't tell whether it began. "
                          "To be safe I stopped without spending anything more. Ask me again if you still want it.")
            return False
        if step not in ("signals", "brief"):
            try:
                return await _start_step(eng, conv, st, state, deal_id, step)
            finally:
                eng.save(conv, st)
        job_id, label = state["job_id"], state.get("label", "")
    # Poll without holding the conversation lock, so the person can keep chatting.
    heading = f"Reading {label}" if step == "signals" else f"Writing the brief for {label}"
    view = await eng.poll(
        conv, app="deals", step=step, job_id=job_id, fetch=lambda: eng.deal.get_job(job_id),
        done=lambda v: v.get("status") in TERMINAL_JOB, progress=lambda v: (v.get("done") or 0, v.get("total") or 1),
        label=heading, final_status=lambda v: v.get("status", "completed"),
    )
    async with eng.lock(conv):
        st = eng.load(conv)
        try:
            return await _finish_step(eng, conv, st, st["run"], deal_id, step, view)
        finally:
            eng.save(conv, st)


def _stop_run(st: dict) -> bool:
    st["run"] = None
    return False


async def _start_step(eng: Engine, conv: str, st: dict, state: dict, deal_id: int, step: str) -> bool:
    if step == "start":
        detail = await eng.deal.get_deal(deal_id)
        state["label"] = _label(detail)
        need = detail.get("signals_status") != "ready"
        if need and not _count(detail.get("interactions")):
            eng.say(conv, f"Skipping {state['label']}: it has no emails or notes to read.")
            state.update(idx=state["idx"] + 1, step="start", job_id=None)
            return True
        if not need and state.get("read_only"):
            eng.say(conv, f"{state['label']} has already been read.", cards=[signals_card(detail)])
            state.update(idx=state["idx"] + 1, step="start", job_id=None)
            return True
        if not need:
            eng.say(conv, f"{state['label']}: signals were already read, so I'm skipping that step and going straight to the brief.")
        state["step"] = "signals_start" if need else "brief_start"
        return True
    if not eng.spend(st, 1):
        eng.say(conv, "I stopped before the next model call because it would go past the number you approved.")
        return _stop_run(st)
    state["resumed"] = False
    eng.save(conv, st)  # the step is on disk before the call, so a restart never repeats a spend
    job_id = await (eng.deal.start_signals if step == "signals_start" else eng.deal.start_brief)(deal_id)
    eng.store.add_job(conv, "deals", job_id, "signals" if step == "signals_start" else "brief")
    state.update(step="signals" if step == "signals_start" else "brief", job_id=job_id)
    return True


async def _finish_step(eng: Engine, conv: str, st: dict, state: dict | None, deal_id: int, step: str, view: dict) -> bool:
    if not state:
        return False
    label = state.get("label", f"deal {deal_id}")
    if view.get("status") != "completed":
        if view.get("status") == "interrupted":
            text = f"The job for {label} was interrupted (Deal Intelligence probably restarted). Nothing more was spent."
        else:
            text = describe_error_info(view.get("error_info"), view.get("error") or "The job failed.", eng.deal.name)
            text = f"That didn't work for {label}. {text}"
        eng.error(conv, text)
        eng.say(conv, "I stopped there. You can open the deal to see its state.", cards=[_deal_link(eng, deal_id)])
        return _stop_run(st)
    if step == "signals":
        detail = await eng.deal.get_deal(deal_id)
        if detail.get("signals_status") != "ready":
            eng.error(conv, f"Reading {label} didn't finish cleanly" + (f": {detail['signals_error']}" if detail.get("signals_error") else "."))
            return _stop_run(st)
        _audit(eng, conv, "deal.read", f"{detail.get('code')} {detail.get('name')}: 1 model call")
        if state.get("read_only"):
            nxt = [] if detail.get("result") != "open" else [
                {"id": f"say:Brief me on {detail['code']}", "label": f"Brief me on {detail['code']}", "style": "ghost"}]
            eng.say(conv, f"Here is what I read in {label}. This used 1 model call. If the deal has ended, tell me how "
                          f"(for example \"{detail.get('name')} was won\").", cards=[signals_card(detail), _deal_link(eng, deal_id)], actions=nxt)
            state.update(idx=state["idx"] + 1, step="start", job_id=None)
            return True
        state.update(step="brief_start", job_id=None)
        return True
    brief = await eng.deal.get_brief(deal_id)
    if not brief or brief.get("status") not in (None, "ready"):
        eng.error(conv, f"The brief for {label} didn't come out" + (f": {brief.get('error')}" if brief and brief.get("error") else "."))
        return _stop_run(st)
    detail = await eng.deal.get_deal(deal_id)
    eng.say(conv, f"Here is the brief for {label}.", cards=[brief_card(detail, brief), _deal_link(eng, deal_id)])
    state.update(idx=state["idx"] + 1, step="start", job_id=None)
    return True


def _union(*lists: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for items in lists:
        for item in items or []:
            seen.setdefault(str(item), None)
    return list(seen)


def brief_card(detail: dict, brief: dict) -> dict:
    """The BriefView as a Card: summary, flags, what similar deals say, what to avoid, next steps, gaps."""
    c = brief.get("content") or {}
    sections: list[dict] = []
    if c.get("summary"):
        sections.append({"heading": "Summary", "text": c["summary"], "chips": c.get("summary_sources") or []})
    if c.get("flags"):
        sections.append({
            "heading": "Flags (checked by code, no model)",
            "bullets": [f"{f.get('severity', 'info')}: {f.get('text')}" for f in c["flags"]],
            "chips": _union(*[f.get("evidence") or [] for f in c["flags"]]),
        })
    if c.get("this_deal"):
        sections.append({
            "heading": "What this deal's own record says",
            "bullets": [i.get("text", "") for i in c["this_deal"]],
            "chips": _union(*[i.get("source_ids") or [] for i in c["this_deal"]]),
        })
    if c.get("warnings"):
        sections.append({
            "heading": "What similar deals say",
            "bullets": [
                w.get("text", "") + (f" ({w['similar_lost']} of {w['similar_total']} similar deals were lost)"
                                     if w.get("similar_total") is not None and w.get("similar_lost") is not None else "")
                for w in c["warnings"]
            ],
            "chips": _union(*[w.get("source_deals") or [] for w in c["warnings"]]),
        })
    if c.get("memory"):
        sections.append({
            "heading": "From memory",
            "bullets": [i.get("text", "") for i in c["memory"]],
            "chips": _union(*[i.get("source_ids") or [] for i in c["memory"]]),
        })
    if c.get("avoid"):
        sections.append({
            "heading": "Likely to backfire",
            "bullets": [a.get("text") or a.get("name", "") for a in c["avoid"]],
            "chips": _union(*[a.get("source_deals") or [] for a in c["avoid"]]),
        })
    if c.get("next_steps"):
        sections.append({
            "heading": "Recommended next steps",
            "bullets": [
                f"{s.get('name')}: {s.get('rationale', '')}" + (f" (why: {'; '.join(s['reasons'])})" if s.get("reasons") else "")
                for s in c["next_steps"]
            ],
            "chips": _union(*[s.get("source_ids") or [] for s in c["next_steps"]]),
        })
    if c.get("missing_info"):
        sections.append({"heading": "Missing information", "bullets": list(c["missing_info"])})
    if c.get("degraded"):
        sections.append({"heading": "Note", "text": "Memory was thin or unavailable for this brief, so it leans on this deal's own "
                                                      "record and the similar-deal evidence may be incomplete."})
    n_closed = c.get("n_closed")
    return {
        "title": f"Brief: {_label(detail)}",
        "subtitle": f"Based on this deal's record and {n_closed} closed deals in memory." if n_closed is not None else None,
        "tone": "warn" if c.get("degraded") else "info",
        "sections": sections,
    }


# -- questions and follow-up drafts (one model call each, asked first) -----------------------------


def _unverified(card: dict) -> None:
    card["tone"] = "warn"
    card["sections"].append({"heading": "Unverified", "text": "I could not tie this to a note or a past deal in the record, so "
                                                              "treat it as a suggestion and check the deal before you rely on it."})


async def deal_question(eng: Engine, conv: str, st: dict, intent: Intent) -> bool:
    """A question about a deal: the one it names, or the deal this chat is already about.
    False when it is not clearly about a deal, so the engine can treat it as an ordinary message."""
    then = {"kind": "ask", "text": intent.text}
    ws = await _workspace(eng, st)
    found = mentioned_deals(intent.text, await eng.deal.list_deals())
    if len(found) > 1:
        _ask_which(eng, conv, st, found[:4], then)
        return True
    if found:
        await _continue(eng, conv, st, await eng.deal.get_deal(found[0].deal["id"]), then)
        return True
    if st.get("flow") == "deal" and st.get("deal_id") and not company_targets(intent.text):  # not about the whole company
        routed = route(intent.text, eng.registry)
        if routed.kind != "match" or routed.best.agent["id"] == "deals":
            await _continue(eng, conv, st, await eng.deal.get_deal(st["deal_id"]), then)
            return True
    return await workspace.elsewhere(eng, conv, st, intent.text, mentioned_deals, again=intent.text)


async def _offer_ask(eng: Engine, conv: str, st: dict, detail: dict, then: dict) -> None:
    label = _label(detail)
    if not _count(detail.get("interactions")):
        return eng.say(conv, f"{label} has no emails or notes yet, so there is nothing to answer from. Attach them first.",
                       cards=[_deal_link(eng, detail["id"])])
    data = {"deal_id": detail["id"], "label": label, "question": then["text"], "workspace_id": (st.get("workspace") or {}).get("id")}
    if st.get("auto_ask"):
        return await _run_ask(eng, conv, st, {**data, "calls": 1})
    eng.confirmation(
        conv, st, f"I'll answer from {label}'s emails and notes{_ws_text(st)} and the similar closed deals in memory.",
        "deal_ask", data, calls=1, cost_note=cost_line(1),
        also=("Yes, and don't ask again for questions in this chat", {"auto_ask": True}),
    )


async def _run_ask(eng: Engine, conv: str, st: dict, data: dict) -> None:
    if not await _workspace_unchanged(eng, conv, data.get("workspace_id")):
        return
    if data.get("auto_ask"):
        st["auto_ask"] = True
    eng.approve(st, 1)
    eng.spend(st, 1)
    result = await eng.deal.ask(data["deal_id"], data["question"])
    st.update({"flow": "deal", "deal_id": data["deal_id"], "deal_label": data["label"]})
    section: dict = {"text": result.get("answer", "")}
    if result.get("sources"):
        section["chips"] = list(result["sources"])
    card: dict = {"title": f"Answer: {data['label']}", "subtitle": data["question"], "tone": "info", "sections": [section]}
    if result.get("found") is False:
        card["sections"].append({"text": "The record doesn't answer this, so I'm not guessing."})
    elif not result.get("grounded", True):
        _unverified(card)
    if result.get("degraded"):
        card["sections"].append({"heading": "Note", "text": "Memory was unavailable, so this leans on the deal's own record."})
    code = data["label"].split(" ")[0]
    follow = {"id": f"say:Draft a follow-up for {code}", "label": "Draft a follow-up", "style": "ghost"}
    eng.say(conv, "Here is what the record says. This used 1 model call.", cards=[card, _deal_link(eng, data["deal_id"])], actions=[follow])


async def deal_ask(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    """A question about one deal that the language model has already pointed at (or about this chat's own deal)."""
    await _with_deal(eng, conv, st, intent.subject, {"kind": "ask", "text": intent.text})


async def deal_followup(eng: Engine, conv: str, st: dict, intent: Intent, uploads: list[Upload]) -> None:
    then = {"kind": "followup", "text": intent.text, "draft_kind": intent.slots.get("kind", "email")}
    await _with_deal(eng, conv, st, intent.subject, then)


async def _offer_followup(eng: Engine, conv: str, st: dict, detail: dict, then: dict) -> None:
    label, kind = _label(detail), then.get("draft_kind", "email")
    if detail.get("result") not in (None, "open"):
        return eng.say(conv, f"{label} is already closed ({detail['result']}), so there is no next step to follow up on.",
                       cards=[_deal_link(eng, detail["id"])])
    brief = await eng.deal.get_brief(detail["id"])
    if not brief or not (brief.get("content") or {}).get("next_steps"):
        nxt = {"id": f"say:Brief me on {detail['code']}", "label": f"Brief me on {detail['code']}", "style": "primary"}
        return eng.say(conv, f"I draft the follow-up from the brief's recommended next step, and {label} doesn't have a brief yet. "
                             "Write the brief first.", actions=[nxt])
    step = brief["content"]["next_steps"][0]
    what, article = ("call agenda", "a") if kind == "call_agenda" else ("email", "an")
    eng.confirmation(
        conv, st, f"I'll draft {article} {what} for {label}{_ws_text(st)} around the brief's top next step: {step.get('name')}.",
        "deal_followup",
        {"deal_id": detail["id"], "label": label, "kind": kind, "play_code": step.get("play_code"),
         "workspace_id": (st.get("workspace") or {}).get("id")},
        calls=1, cost_note=cost_line(1),
    )


async def _run_followup(eng: Engine, conv: str, st: dict, data: dict) -> None:
    if not await _workspace_unchanged(eng, conv, data.get("workspace_id")):
        return
    eng.approve(st, 1)
    eng.spend(st, 1)
    result = await eng.deal.followup(data["deal_id"], data["kind"], data.get("play_code"))
    what = "Call agenda" if data["kind"] == "call_agenda" else "Email"
    sections: list[dict] = [{"heading": result.get("subject") or what, "text": result.get("body", ""), "chips": list(result.get("sources") or [])}]
    card: dict = {"title": f"{what} draft: {data['label']}", "subtitle": f"For the next step: {result.get('step_name')}",
                  "tone": "info", "sections": sections}
    if "[" in (result.get("body") or ""):
        sections.append({"text": "Fill in the [bracketed] parts yourself; I left them where the record doesn't say."})
    if not result.get("grounded", True):
        _unverified(card)
    other = "email" if data["kind"] == "call_agenda" else "call agenda"
    code = data["label"].split(" ")[0]
    follow = {"id": f"say:Draft a {other} for {code}", "label": f"Make it a {other}", "style": "ghost"}
    eng.say(conv, "Here is the draft. Nothing is sent or saved, so copy it where you need it. This used 1 model call.",
            cards=[card, _deal_link(eng, data["deal_id"])], actions=[follow])


# -- outcome ----------------------------------------------------------------------------------------


async def outcome(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    then = {"kind": "outcome", "result": intent.result, "loss_reason": intent.loss_reason, "text": intent.text}
    await _with_deal(eng, conv, st, intent.subject, then)


async def _offer_outcome(eng: Engine, conv: str, st: dict, detail: dict, then: dict) -> None:
    label = _label(detail)
    if detail.get("result") not in (None, "open"):
        why = f" ({detail['loss_reason']})" if detail.get("loss_reason") else ""
        return eng.say(
            conv,
            f"{label} is already recorded as {detail['result']}{why}. I won't overwrite a recorded outcome from here; "
            "change it in Deal Intelligence if it needs correcting.",
            cards=[_deal_link(eng, detail["id"])],
        )
    result, reason = then["result"], then.get("loss_reason")
    if result == "lost" and not reason:
        st["awaiting"] = {"type": "loss_reason", "deal_id": detail["id"], "then": then}
        vocabulary = "; ".join(words for words, _ in LOSS_REASON_WORDS.values())
        actions = [{"id": f"say:{short}", "label": short, "style": "ghost"} for _, short in LOSS_REASON_WORDS.values()]
        return eng.say(
            conv,
            f"Why was {label} lost? Tell me in your own words. I can record: {vocabulary}.",
            actions=actions,
        )
    plays = [p["code"] for p in ((detail.get("signals") or {}).get("plays_used") or [])]
    names = [f"{p['code']} {p.get('name', '')}".strip() for p in ((detail.get("signals") or {}).get("plays_used") or [])]
    text = (
        f"Ready to record {label} as {result.upper()}" + (f" because: {LOSS_REASON_WORDS[reason][0]}" if reason else "") + ". "
        "This uses 0 model calls, but it changes memory: "
    )
    if plays:
        text += f"the plays used on this deal ({', '.join(names)}) get credit or blame, and a lesson is stored for future briefs."
    else:
        text += ("this deal hasn't been read yet, so no plays are recorded and no play credit will change, but the outcome and a "
                 "lesson are still stored for future briefs.")
    eng.confirmation(
        conv, st, text, "deal_outcome",
        {"deal_id": detail["id"], "label": label, "result": result, "loss_reason": reason, "plays_used": plays,
         "workspace_id": (st.get("workspace") or {}).get("id")},
        calls=0, label="Yes, record it", cost_label="0 model calls, updates memory",
    )


async def _record_outcome(eng: Engine, conv: str, st: dict, data: dict) -> None:
    if not await _workspace_unchanged(eng, conv, data.get("workspace_id")):
        return
    # plays_used is sent explicitly from the deal's own signals so the server never guesses what was used.
    result = await eng.deal.record_outcome(data["deal_id"], data["result"], data.get("loss_reason"), data["plays_used"])
    deal = result.get("deal") or {}
    _audit(eng, conv, "deal.outcome", f"{data['label']}: {data['result']}" + (f" ({data['loss_reason']})" if data.get("loss_reason") else ""))
    rows = [["Result", data["result"]]]
    if data.get("loss_reason"):
        rows.append(["Reason", LOSS_REASON_WORDS[data["loss_reason"]][0]])
    rows.append(["Lessons stored", str(result.get("lessons_added", 0))])
    sections: list[dict] = [{"rows": rows}]
    credit = result.get("credit") or []
    if credit:
        sections.append({"heading": "Play credit", "bullets": [
            f"{c['play_code']}: {c['delta']:+g} (used {c.get('times_used')}, won {c.get('won')}, lost on quality {c.get('lost_quality')})"
            for c in credit
        ]})
    else:
        sections.append({"heading": "Play credit", "text": "No play credit changed."})
    if result.get("memory_note"):
        sections.append({"text": result["memory_note"]})
    if result.get("events"):
        sections.append({"heading": "What was recorded", "bullets": list(result["events"])})
    card = {"title": f"Outcome recorded: {data['label']}", "tone": "ok", "sections": sections}
    follow = eng.offer(st, "brief_open", "Brief the open deals again", style="ghost")
    eng.say(
        conv, f"Done. {deal.get('code', '')} is now {data['result']}. This used no model calls.".replace("  ", " "),
        cards=[card, _deal_link(eng, data["deal_id"])], actions=[follow],
    )


async def _offer_brief_open(eng: Engine, conv: str, st: dict) -> None:
    ws = await _workspace(eng, st)
    deals = [d for d in await eng.deal.list_deals() if d.get("result") == "open"]
    if not deals:
        return eng.say(conv, "There are no open deals to brief right now.")
    ready = [d for d in deals if _count(d.get("interactions")) > 0]
    skipped = [d for d in deals if d not in ready]
    reads = sum(1 for d in ready if d.get("signals_status") != "ready")
    calls = reads + len(ready)
    if not ready:
        return eng.say(conv, "None of the open deals has emails or notes to brief from yet.")
    text = f"I can brief {plural(len(ready), 'open deal')}{_ws_text(st)}: " + ", ".join(f"{d['code']} {d['name']}" for d in ready) + "."
    if skipped:
        text += " Skipping (no emails or notes): " + ", ".join(d["code"] for d in skipped) + "."
    parts = f"{plural(reads, 'read')} of deals not yet read + {plural(len(ready), 'brief')}" if reads else f"{plural(len(ready), 'brief')}"
    eng.confirmation(
        conv, st, text, "deal_run", {"deal_ids": [d["id"] for d in ready], "workspace_id": ws.id},
        calls=calls, cost_note=cost_line(calls, parts),
    )


# -- new deal, add files, notes ---------------------------------------------------------------------


def _ask_slots(eng: Engine, conv: str, st: dict, slots: dict) -> None:
    st["awaiting"] = {"type": "deal_slots", "slots": slots}
    if not slots.get("name") and not slots.get("account"):
        question = "What is the deal called, and which account is it with? For example: Tidewater at Tidewater Freight."
    elif not slots.get("name"):
        question = f"What should I call the deal with {slots['account']}?"
    else:
        question = f"Which account (company) is \"{slots['name']}\" with?"
    eng.say(conv, question)


_SEGMENTS = {"smb": "smb", "small": "smb", "small_business": "smb", "startup": "smb", "mid_market": "mid_market",
             "midmarket": "mid_market", "mid": "mid_market", "mid_size": "mid_market", "midsize": "mid_market",
             "mid_sized": "mid_market", "medium": "mid_market", "enterprise": "enterprise", "large": "enterprise",
             "big": "enterprise", "global": "enterprise"}
_SEGMENT_WORDS = re.compile(r"\b(small(?:[\s-]business)?|smb|start-?up|tiny|mid[\s-]?(?:size[d]?|market)|medium(?:[\s-]sized)?|"
                            r"enterprise|large|big|huge|global|multinational)\b", re.I)
_INDUSTRIES = {  # what people say -> how the deal records it
    "logistics": "logistics", "freight": "logistics", "shipping": "logistics", "banking": "banking", "bank": "banking",
    "credit union": "banking", "fintech": "fintech", "financial services": "financial services", "insurance": "insurance",
    "healthcare": "healthcare", "health care": "healthcare", "hospital": "healthcare", "pharma": "pharma",
    "pharmaceutical": "pharma", "biotech": "biotech", "software": "software", "saas": "software", "manufacturing": "manufacturing",
    "manufacturer": "manufacturing", "industrial": "manufacturing", "retail": "retail", "ecommerce": "ecommerce",
    "e-commerce": "ecommerce", "education": "education", "university": "education", "government": "government",
    "public sector": "government", "telecom": "telecom", "telecommunications": "telecom", "energy": "energy",
    "utilities": "utilities", "media": "media", "hospitality": "hospitality", "real estate": "real estate",
    "construction": "construction", "automotive": "automotive", "legal": "legal", "law firm": "legal",
    "nonprofit": "nonprofit", "non-profit": "nonprofit", "agriculture": "agriculture", "consulting": "consulting",
}
_INDUSTRY_WORDS = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in _INDUSTRIES), key=len, reverse=True)) + r")\b", re.I)


def normal_segment(value: Any) -> str | None:
    """smb, mid_market or enterprise (the only values the Deal agent takes), or None."""
    raw = str(value or "").strip().lower()
    key = raw.replace("-", "_").replace(" ", "_")
    if key in _SEGMENTS:
        return _SEGMENTS[key]
    found = _SEGMENT_WORDS.search(raw)
    if not found:
        return None
    word = found.group(1).lower()
    if word.startswith(("small", "smb", "start", "tiny")):
        return "smb"
    if word.startswith(("mid", "medium")):
        return "mid_market"
    return "enterprise"


def details_from_words(text: str) -> dict[str, str]:
    """An industry and a size said in plain words ("a big manufacturing company"), read here without a model."""
    out: dict[str, str] = {}
    industry = _INDUSTRY_WORDS.search(text or "")
    if industry:
        out["industry"] = _INDUSTRIES[industry.group(1).lower()]
    size = _SEGMENT_WORDS.search(text or "")
    segment = normal_segment(size.group(1)) if size else None
    if segment:
        out["segment"] = segment
    return out


def _fill(slots: dict, found: dict, source: str) -> None:
    """Fill the details still missing, and remember where each came from (shown on the card)."""
    for key in ("industry", "segment"):
        value = found.get(key)
        if key == "segment":
            value = normal_segment(value)
        elif value:
            value = _INDUSTRIES.get(str(value).strip().lower(), str(value).strip().lower())
        if value and not slots.get(key):
            slots[key] = value
            slots.setdefault("from", {})[key] = source


def _audit(eng: Engine, conv: str, action: str, detail: str) -> None:
    if eng.auth is None:
        return
    user = eng.auth.user_by_id(eng.store.conversation_owner(conv))
    eng.auth.audit(user.email if user else "chat", action, detail)


async def deal_new(eng: Engine, conv: str, st: dict, intent: Intent, uploads: list[Upload]) -> None:
    await _workspace(eng, st)
    slots = dict(intent.slots)
    if slots.get("segment"):
        segment = normal_segment(slots["segment"])
        if segment:
            slots["segment"] = segment
        else:
            slots.pop("segment")  # the Deal agent accepts only smb, mid_market or enterprise
    _fill(slots, details_from_words(intent.text), "your message")
    if not (slots.get("name") and slots.get("account")):
        return _ask_slots(eng, conv, st, slots)
    await _create_or_confirm(eng, conv, st, slots)


async def _create_or_confirm(eng: Engine, conv: str, st: dict, slots: dict) -> None:
    st["awaiting"] = None
    deals = await eng.deal.list_deals()
    same = [d for d in deals if d.get("name", "").lower() == slots["name"].lower() and d.get("account", "").lower() == slots["account"].lower()]
    if same:
        group = secrets.token_hex(3)
        actions = [
            eng.offer(st, "deal_add_existing", f"Add to {same[0]['code']}", style="primary", data={"deal_id": same[0]["id"]}, group=group),
            eng.offer(st, "deal_create", "Create a separate deal", style="ghost", data={"slots": slots}, group=group),
        ]
        return eng.say(conv, f"{_label(same[0])} already exists with that name and account. Add the files to it, or create a separate deal?", actions=actions)
    await _create(eng, conv, st, slots)


async def _create(eng: Engine, conv: str, st: dict, slots: dict) -> None:
    files = eng.pending_uploads(st)
    slots = dict(slots)
    for upload in files:
        _fill(slots, filepeek.deal_details(filepeek.read_text(upload)), "the notes")
    detail = await eng.deal.create_deal(
        slots["name"], slots["account"], [(u.filename, u.read()) for u in files],
        amount=slots.get("amount"), segment=slots.get("segment"), industry=slots.get("industry"),
    )
    st["uploads"] = []
    st.update({"flow": "deal", "deal_id": detail["id"], "deal_label": _label(detail), "awaiting": None})
    n = _count(detail.get("interactions"))
    source = slots.get("from") or {}

    def shown(key: str) -> str:
        value = detail.get(key)
        if not value:
            return "not given"
        text = str(value).replace("_", " ")
        return f"{text} (from {source[key]})" if source.get(key) else text

    rows = [["Deal", _label(detail)], ["Industry", shown("industry")], ["Segment", shown("segment")],
            ["Files added", str(len(files))], ["Emails and notes read in", str(n)], ["Signals", detail.get("signals_status", "none")]]
    missing = [k for k in ("industry", "segment") if not detail.get(k)]
    card = {"title": f"Created {_label(detail)}", "tone": "warn" if missing else "ok", "sections": [{"rows": rows}]}
    _audit(eng, conv, "deal.create", f"{detail['code']} {detail.get('name')} ({detail.get('account')}) with {plural(len(files), 'file')}")
    nxt = [{"id": f"say:Read {detail['code']}", "label": f"Read {detail['code']}", "style": "primary"},
           {"id": f"say:Brief me on {detail['code']}", "label": f"Brief me on {detail['code']}", "style": "ghost"}] if n else []
    eng.say(
        conv,
        f"Created {_label(detail)} in Deal Intelligence{_ws_text(st)} with {plural(len(files), 'file')}. This used no model calls. "
        "Read it to see its objections and plays (1 model call); for a deal that has already ended, read it before recording "
        "the outcome so the right plays get the credit."
        + (f" Its {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not given: similar deals are matched on them, so tell "
           f"me if you know, for example \"{detail['code']} is a small software company\"." if missing else ""),
        cards=[card, _deal_link(eng, detail["id"])], actions=nxt,
    )


async def deal_add(eng: Engine, conv: str, st: dict, intent: Intent, uploads: list[Upload]) -> None:
    await _with_deal(eng, conv, st, intent.subject, {"kind": "add", "text": intent.text})


async def files_for_deal(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    await _workspace(eng, st)
    st["awaiting"] = {"type": "deal_ref", "then": {"kind": "add", "text": ""}}
    eng.say(conv, "Which deal are these files for? Name an existing deal, or say \"New deal <name> at <account>\" and I'll create it.")


async def _add_files(eng: Engine, conv: str, st: dict, detail: dict) -> None:
    files = eng.pending_uploads(st)
    if not files:
        return eng.say(conv, f"There are no attached files waiting. Attach the emails or notes and say \"add these to {detail['name']}\".")
    updated = await eng.deal.add_files(detail["id"], [(u.filename, u.read()) for u in files])
    st["uploads"] = []
    n = _count(updated.get("interactions"))
    eng.say(
        conv,
        f"Added {plural(len(files), 'file')} to {_label(detail)}. It now has {plural(n, 'email or note')} (signals: "
        f"{updated.get('signals_status', 'none')}). This used no model calls.",
        cards=[_deal_link(eng, detail["id"])],
        actions=[{"id": f"say:Brief me on {detail['code']}", "label": f"Brief me on {detail['code']}", "style": "primary"}],
    )


async def deal_note(eng: Engine, conv: str, st: dict, intent: Intent, uploads: list[Upload]) -> None:
    # The slots carry the note's own kind/text; keep them apart from the flow's `kind`/`text` keys.
    then = {"kind": "note", "text": intent.text, "note_kind": intent.slots.get("kind", "call_note"), "body": intent.slots.get("text", "")}
    await _with_deal(eng, conv, st, intent.subject, then)


async def _add_note(eng: Engine, conv: str, st: dict, detail: dict, then: dict) -> None:
    updated = await eng.deal.add_note(detail["id"], then.get("body", ""), kind=then.get("note_kind", "call_note"))
    eng.say(conv, f"Added the note to {_label(detail)}; it now has {plural(_count(updated.get('interactions')), 'email or note')}. "
                  "This used no model calls.", cards=[_deal_link(eng, detail["id"])])


# -- replies to my questions ------------------------------------------------------------------------


async def handle_slot(eng: Engine, conv: str, st: dict, intent: Intent, accepted: list[Upload]) -> None:
    awaiting = st.get("awaiting") or {}
    kind, text = awaiting.get("type"), intent.text
    if kind == "deal_slots":
        slots = reply_slots(text, awaiting.get("slots") or {})
        said = {k: v for k, v in (intent.slots or {}).items() if k in ("name", "account") and v}
        if said:  # the model read the reply ("lumen grid, a small software company"): trust its name and account
            slots = {**(awaiting.get("slots") or {}), **said}
            slots.setdefault("name", slots.get("account"))
        _fill(slots, {k: v for k, v in (intent.slots or {}).items() if k in ("industry", "segment")}, "your message")
        _fill(slots, details_from_words(text), "your message")
        if slots.get("name") and slots.get("account"):
            return await _create_or_confirm(eng, conv, st, slots)
        return _ask_slots(eng, conv, st, slots)
    if kind == "deal_choice":
        candidates = awaiting["candidates"]
        deals = [{"id": c["id"], "code": c["label"].split(" ")[0], "name": c["label"], "account": ""} for c in candidates]
        from ..intents import Candidate

        index = pick_candidate(text, [Candidate(d, 0.0, []) for d in deals])
        if index is None:
            return eng.say(conv, "Which one? Say 1" + (f" to {len(candidates)}" if len(candidates) > 1 else "") + ", or part of its name.")
        return await pick(eng, conv, st, candidates[index]["id"], awaiting.get("then") or {})
    if kind == "deal_ref":
        then = awaiting.get("then") or {}
        st["awaiting"] = None
        return await _with_deal(eng, conv, st, text.strip(" .?!") or None, then)
    if kind == "loss_reason":
        reason = parse_reason(text)
        if reason is None:
            options = reason_matches(text)
            if len(options) > 1:
                words = " or ".join(LOSS_REASON_WORDS[o][0] for o in options)
                return eng.say(conv, f"That could mean {words}. Which one is closest? I record one reason per deal.")
            return eng.say(conv, "I didn't catch a reason I can record. The choices are: "
                           + "; ".join(w for w, _ in LOSS_REASON_WORDS.values()) + ".")
        then = {**(awaiting.get("then") or {}), "loss_reason": reason}
        st["awaiting"] = None
        return await _offer_outcome(eng, conv, st, await eng.deal.get_deal(awaiting["deal_id"]), then)
    st["awaiting"] = None
    eng.say(conv, "Okay. Tell me what you'd like to do.")


# -- button clicks ----------------------------------------------------------------------------------


async def on_action(eng: Engine, conv: str, st: dict, kind: str, data: dict) -> None:
    if kind == "deal_run":
        return await _start_run(eng, conv, st, data)
    if kind == "deal_pick":
        return await pick(eng, conv, st, data["deal_id"], data.get("then") or {})
    if kind == "deal_ask":
        return await _run_ask(eng, conv, st, data)
    if kind == "deal_followup":
        return await _run_followup(eng, conv, st, data)
    if kind == "deal_outcome":
        return await _record_outcome(eng, conv, st, data)
    if kind == "deal_create":
        return await _create(eng, conv, st, data["slots"])
    if kind == "deal_add_existing":
        detail = await eng.deal.get_deal(data["deal_id"])
        st.update({"flow": "deal", "deal_id": detail["id"], "awaiting": None})
        return await _add_files(eng, conv, st, detail)
    if kind == "brief_open":
        return await _offer_brief_open(eng, conv, st)
    eng.say(conv, "I don't know how to do that any more.")
