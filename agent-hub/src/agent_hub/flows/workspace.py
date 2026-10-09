"""Which workspace a chat works in.

Every agent keeps its work in workspaces (one company's deals, one company's RFP library). A chat chooses
one per agent and the hub sends it with each request (X-Workspace), so the person never has to switch the
workspace an agent's own screens show, and two chats can work in two workspaces at once. Choosing costs no
model calls and changes nothing inside the agents: it is only this chat's setting."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ..clients import AgentDown, UpstreamError, workspace_override
from ..intents import _tokens

if TYPE_CHECKING:
    from ..engine import Engine
    from ..intents import Intent

AGENTS = (("deals", "Deal Intelligence", "deals"), ("rfp", "RFP Memory Assistant", "RFPs"))
DEFAULTS = {"default", "active", "current", "original", "usual", "normal"}
_FOR_AGENT = re.compile(r"\s+for\s+(?P<a>deals?|rfps?|proposals?|questionnaires?)\s*$", re.I)


def _client(eng: Engine, agent_id: str):  # noqa: ANN202
    return eng.deal if agent_id == "deals" else eng.rfp


def _chosen(st: dict, agent_id: str) -> str | None:
    return (st.get("workspaces") or {}).get(agent_id)


async def _lists(eng: Engine) -> dict[str, tuple[str | None, list[dict[str, Any]]] | str]:
    """Each live agent's (active workspace id, workspaces), or a sentence saying why it can't be read."""
    out: dict[str, Any] = {}
    for agent_id, name, _ in AGENTS:
        try:
            out[agent_id] = await _client(eng, agent_id).workspaces()
        except (AgentDown, UpstreamError) as exc:
            out[agent_id] = exc.message
    return out


def _set(st: dict, agent_id: str, space: dict | None) -> None:
    st.setdefault("workspaces", {})
    st.setdefault("workspace_names", {})
    if space is None:
        st["workspaces"].pop(agent_id, None)
        st["workspace_names"].pop(agent_id, None)
    else:
        st["workspaces"][agent_id] = space["id"]
        st["workspace_names"][agent_id] = space.get("name") or space["id"]
    # Ids (deals, projects) belong to a workspace, so what the chat was "about" no longer applies.
    if agent_id == "deals":
        st["deal_id"] = st["deal_label"] = None
    else:
        st["project_id"] = st["project"] = None
    if st.get("flow") == ("deal" if agent_id == "deals" else "rfp"):
        st["flow"] = None


async def show(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    lists = await _lists(eng)
    sections: list[dict] = []
    actions: list[dict] = []
    for agent_id, name, short in AGENTS:
        found = lists[agent_id]
        if isinstance(found, str):
            sections.append({"heading": name, "text": found})
            continue
        active, spaces = found
        using = _chosen(st, agent_id) or active
        bullets = []
        for space in spaces:
            marks = []
            if space["id"] == using:
                marks.append("this chat")
            if space["id"] == active and space["id"] != using:
                marks.append("open in the app")
            bullets.append(f"{space.get('name') or space['id']} ({space.get('kind', 'workspace')})" + (f" - {', '.join(marks)}" if marks else ""))
            if space["id"] != using:
                actions.append(eng.offer(st, "ws_use", f"Use {space.get('name') or space['id']} for {short}", style="ghost",
                                         data={"agent": agent_id, "workspace": space["id"]}))
        sections.append({"heading": name, "bullets": bullets or ["No workspaces yet."]})
    card = {"title": "Workspaces", "tone": "info", "sections": sections}
    eng.say(conv, "This chat works in the workspace marked \"this chat\" for each agent. Pick another and I'll use it from now on; "
                  "nothing is switched inside the agents.", cards=[card], actions=actions or None)


def _matches(subject: str, spaces: list[dict]) -> list[dict]:
    wanted = _tokens(_FOR_AGENT.sub("", subject))
    plain = re.sub(r"[^a-z0-9]+", "", subject.lower())
    exact = [s for s in spaces if plain and plain in (re.sub(r"[^a-z0-9]+", "", s["id"].lower()),
                                                        re.sub(r"[^a-z0-9]+", "", (s.get("name") or "").lower()))]
    if exact:
        return exact
    if not wanted:
        return []
    return [s for s in spaces if wanted <= _tokens(f"{s.get('name', '')} {s['id'].replace('-', ' ')}")]


async def use(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    subject = (intent.subject or "").strip()
    only = intent.slots.get("agent")
    hint = _FOR_AGENT.search(subject)
    if hint:
        only = "deals" if hint.group("a").lower().startswith("deal") else "rfp"
        subject = _FOR_AGENT.sub("", subject).strip()
    lists = await _lists(eng)
    targets = [a for a in AGENTS if only in (None, a[0])]

    if subject.lower() in DEFAULTS:
        for agent_id, _name, _short in targets:
            _set(st, agent_id, None)
        eng.use_workspaces(st)
        return eng.say(conv, "Okay. This chat uses each agent's own open workspace again, whichever that is.")

    picked: list[tuple[str, dict]] = []
    unclear: list[tuple[str, list[dict]]] = []
    for agent_id, _name, _short in targets:
        found = lists[agent_id]
        if isinstance(found, str):
            continue
        hits = _matches(subject, found[1])
        if len(hits) == 1:
            picked.append((agent_id, hits[0]))
        elif hits:
            unclear.append((agent_id, hits))
    if not picked and not unclear:
        return await _not_found(eng, conv, st, subject, lists)
    for agent_id, space in picked:
        _set(st, agent_id, space)
    eng.use_workspaces(st)
    lines = []
    for agent_id, space in picked:
        name = dict((a[0], a[1]) for a in AGENTS)[agent_id]
        lines.append(f"{name}: {space.get('name') or space['id']}")
    actions: list[dict] = []
    text = ""
    if picked:
        text = ("This chat now works in " + "; ".join(lines) + ". Nothing was switched inside the agents, so their own screens "
                "still show whichever workspace they have open.")
        detail = await _describe(eng, picked)
        if detail:
            text += " " + detail
    if unclear:
        for agent_id, hits in unclear:
            short = dict((a[0], a[2]) for a in AGENTS)[agent_id]
            for space in hits:
                actions.append(eng.offer(st, "ws_use", f"Use {space.get('name') or space['id']} for {short}", style="ghost",
                                         data={"agent": agent_id, "workspace": space["id"]}))
        text = (text + " " if text else "") + "More than one workspace fits for the others; which do you mean?"
    eng.say(conv, text, actions=actions or None)


async def _describe(eng: Engine, picked: list[tuple[str, dict]]) -> str:
    """How much is in the deal workspace just chosen, so the person can see it is the right one."""
    if not any(a == "deals" for a, _ in picked):
        return ""
    try:
        deals = await eng.deal.list_deals()
    except (AgentDown, UpstreamError):
        return ""
    open_deals = sum(1 for d in deals if d.get("result") == "open")
    return f"It has {len(deals)} deals ({open_deals} open)."


async def _not_found(eng: Engine, conv: str, st: dict, subject: str, lists: dict) -> None:
    names = []
    for agent_id, name, _short in AGENTS:
        found = lists[agent_id]
        if not isinstance(found, str):
            names += [f"{s.get('name') or s['id']} ({name})" for s in found[1]]
    shown = ", ".join(names) if names else "none that I can read"
    eng.say(conv, f"I couldn't find a workspace called \"{subject}\". The ones I can see: {shown}. "
                  "Say \"show workspaces\" to pick one with a button.")


async def on_action(eng: Engine, conv: str, st: dict, kind: str, data: dict) -> None:
    if kind != "ws_use":
        return eng.say(conv, "I don't know how to do that any more.")
    agent_id, space_id = data["agent"], data["workspace"]
    try:
        _active, spaces = await _client(eng, agent_id).workspaces()
    except (AgentDown, UpstreamError) as exc:
        return eng.fail(conv, exc)
    space = next((s for s in spaces if s["id"] == space_id), None)
    if space is None:
        return eng.say(conv, "That workspace isn't there any more. Say \"show workspaces\" to see what exists.")
    _set(st, agent_id, space)
    eng.use_workspaces(st)
    name = dict((a[0], a[1]) for a in AGENTS)[agent_id]
    if data.get("resume"):  # chosen in answer to "which one?": carry on with what was asked
        from ..engine import Prepared

        eng.say(conv, f"Using {space.get('name') or space['id']} for {name}.")
        return await eng._dispatch(conv, st, data["resume"], Prepared())
    text = f"This chat now works in {space.get('name') or space['id']} for {name}. Nothing was switched inside the agent."
    detail = await _describe(eng, [(agent_id, space)])
    follow = [{"id": f"say:{data['again']}", "label": "Ask it again", "style": "primary"}] if data.get("again") else None
    eng.say(conv, f"{text} {detail}".strip(), actions=follow)


async def elsewhere(
    eng: Engine, conv: str, st: dict, text: str, finder, *, again: str,  # noqa: ANN001
) -> bool:
    """A deal that is not in this chat's workspace may be in another of the deal agent's workspaces: say so and
    offer to use it. `finder(text, deals)` returns the candidates in one workspace. False when none has it."""
    try:
        active, spaces = await eng.deal.workspaces()
    except (AgentDown, UpstreamError):
        return False
    current = _chosen(st, "deals") or active
    hits: list[tuple[dict, dict]] = []
    for space in spaces:
        if space["id"] == current:
            continue
        try:
            with workspace_override("deals", space["id"]):
                deals = await eng.deal.list_deals()
        except (AgentDown, UpstreamError):
            continue
        hits += [(space, c.deal) for c in finder(text, deals)[:2]]
    if not hits:
        return False
    here = next((s.get("name") for s in spaces if s["id"] == current), current)
    actions = []
    lines = []
    for space, deal in hits[:3]:
        label = f"{deal.get('code')} {deal.get('name')}"
        lines.append(f"{label} is in {space.get('name') or space['id']}")
        actions.append(eng.offer(st, "ws_use", f"Use {space.get('name') or space['id']}", style="primary",
                                 data={"agent": "deals", "workspace": space["id"], "again": again}))
    eng.say(conv, f"I couldn't find that in \"{here}\", the workspace this chat is using. " + "; ".join(lines)
            + ". Want to use that workspace for this chat?", actions=actions)
    return True


async def remember(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    """Save this chat's workspaces (and a company name) as the start of every new chat."""
    if st.get("tenant") is not None:
        return eng.say(conv, "Your workspaces are set by your administrator, so there is nothing to save.")
    chosen = dict(st.get("workspaces") or {})
    if not chosen:
        return eng.say(conv, "This chat is using each agent's own open workspace, and that can change. Choose the workspaces first "
                             "(say \"show workspaces\"), then say \"remember these workspaces as <company name>\".")
    name = intent.subject or st.get("company_name")
    eng.store.set_setting("company", {"name": name, "workspaces": chosen, "workspace_names": dict(st.get("workspace_names") or {})})
    st["company_name"] = name
    parts = [f"{dict((a[0], a[1]) for a in AGENTS)[k]}: {(st.get('workspace_names') or {}).get(k, v)}" for k, v in chosen.items()]
    eng.say(conv, f"Saved{f' as {name}' if name else ''}. New chats now start in " + "; ".join(parts) + ". "
                  "Say \"forget the saved workspaces\" to undo that.")


async def forget(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    if st.get("tenant") is not None:
        return eng.say(conv, "Your workspaces are set by your administrator, so there is nothing to forget.")
    eng.store.set_setting("company", None)
    st.pop("company_name", None)
    eng.say(conv, "Forgotten. New chats start in each agent's own open workspace again. This chat keeps what it has now.")


async def busier(eng: Engine, st: dict) -> list[tuple[dict, int]]:
    """The deal agent's other workspaces that have deals in them (read only), most deals first."""
    try:
        active, spaces = await eng.deal.workspaces()
    except (AgentDown, UpstreamError):
        return []
    current = _chosen(st, "deals") or active
    found: list[tuple[dict, int]] = []
    for space in spaces:
        if space["id"] == current:
            continue
        try:
            with workspace_override("deals", space["id"]):
                count = len(await eng.deal.list_deals())
        except (AgentDown, UpstreamError):
            continue
        if count:
            found.append((space, count))
    return sorted(found, key=lambda item: -item[1])


async def nudge_empty(eng: Engine, conv: str, st: dict, here: str | None) -> None:
    """The chat's deal workspace has no deals: say so, and point at the workspaces that do."""
    others = await busier(eng, st)
    text = f"There are no deals in {here or 'this workspace'}."
    if not others:
        return eng.say(conv, text + " Create a deal or import some in Deal Intelligence first.")
    actions = [eng.offer(st, "ws_use", f"Use {space.get('name') or space['id']}", style="primary",
                         data={"agent": "deals", "workspace": space["id"]}) for space, _n in others[:3]]
    listed = "; ".join(f"{space.get('name') or space['id']} has {n}" for space, n in others[:3])
    eng.say(conv, f"{text} Other workspaces: {listed}. Want to use one of them for this chat?", actions=actions)


DEAL_KINDS = {"deal_brief", "deal_new", "deal_add", "deal_note", "deal_followup", "deal_count", "deal_list", "deal_files"}
RFP_KINDS = {"rfp_start"}


def _needed_agents(intent: Intent) -> list[str]:
    """Which agents' workspaces a request depends on (those that are not just replies or commands)."""
    from ..intents import company_targets

    if intent.kind in DEAL_KINDS or intent.kind == "deal_ask":
        return ["deals"]
    if intent.kind == "company_ask":
        return list(intent.slots.get("targets") or [])
    if intent.kind in RFP_KINDS:
        return ["rfp"]
    if intent.kind == "outcome":
        return ["rfp" if intent.target == "rfp" else "deals"]
    if intent.kind == "question":
        return company_targets(intent.text) or []
    return []


async def gate(eng: Engine, conv: str, st: dict, intent: Intent) -> bool:
    """A chat that has not chosen a workspace for an agent is asked once, when it first needs that agent and the agent
    has more than one workspace, instead of silently working in whichever one the agent has open. True when asked
    (the original request continues by itself after the person picks)."""
    asked = st.setdefault("ws_asked", {})
    for agent_id in _needed_agents(intent):
        client = _client(eng, agent_id)
        try:
            active, spaces = await client.workspaces()
        except (AgentDown, UpstreamError):
            continue  # the flow itself will say the agent is down
        chosen = _chosen(st, agent_id)
        names = st.setdefault("workspace_names", {})
        if chosen or asked.get(agent_id) or len(spaces) <= 1:
            effective = next((s for s in spaces if s["id"] == (chosen or active)), None)
            if effective and agent_id not in names:
                names[agent_id] = effective.get("name") or effective["id"]  # so the chip shows where this chat is working
            continue
        asked[agent_id] = True
        name, short = dict((a[0], a[1]) for a in AGENTS)[agent_id], dict((a[0], a[2]) for a in AGENTS)[agent_id]
        actions = []
        for space in spaces:
            label = space.get("name") or space["id"]
            extra = []
            if agent_id == "deals":
                try:
                    with workspace_override("deals", space["id"]):
                        extra.append(f"{len(await eng.deal.list_deals())} deals")
                except (AgentDown, UpstreamError):
                    pass
            if space["id"] == active:
                extra.append("open in the app")
            actions.append(eng.offer(st, "ws_use", label + (f" ({', '.join(extra)})" if extra else ""), style="ghost",
                                     data={"agent": agent_id, "workspace": space["id"], "resume": intent.text}))
        eng.say(conv, f"You have {len(spaces)} workspaces in {name}. Which one should this chat use for {short}? "
                      "(Say \"remember these workspaces as <company>\" afterwards to make your choice the start of every new chat.)",
                actions=actions)
        return True
    return False
