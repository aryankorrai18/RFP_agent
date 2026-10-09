"""Questions about the whole company's records, not one deal or one RFP.

The pipeline question ("why do we lose fintech deals?") goes to Deal Intelligence, which counts across every deal
in the chat's workspace and answers from those counts and the deal summaries. The library question ("what have we
told clients about SOC 2?") goes to the RFP assistant, which answers from the company facts and approved past
answers. A question that needs both asks both. Each agent answers only from its own data; the hub routes, asks
first (one model call per agent) and puts the answers side by side, with their citations."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from ..clients import AgentDown, UpstreamError
from ..engine import cost_line, plural
from ..intents import company_targets
from . import workspace

if TYPE_CHECKING:
    from ..engine import Engine
    from ..intents import Intent

NAMES = {"deals": "your deals", "rfp": "your RFP library and company facts"}


def _workspace_label(st: dict, agent_id: str) -> str:
    name = (st.get("workspace_names") or {}).get(agent_id)
    return f" ({name})" if name else ""


async def company_question(eng: Engine, conv: str, st: dict, intent: Intent, targets: list[str] | None = None) -> bool:
    """False when this is not a question about the company's records (so it is treated as an ordinary message).
    `targets` is given when the language model already chose which agents hold the answer."""
    targets = list(targets or company_targets(intent.text))
    if not targets:
        return False
    for agent_id in targets:  # resolve the workspace names for the confirmation text, and fail early if one is gone
        try:
            info = await (eng.deal.status() if agent_id == "deals" else eng.rfp.workspace_info())
        except (AgentDown, UpstreamError) as exc:
            eng.fail(conv, exc)
            return True
        st.setdefault("workspace_names", {}).setdefault(agent_id, info.name or info.id or "")
    if "deals" in targets and not await eng.deal.list_deals():
        await workspace.nudge_empty(eng, conv, st, (st.get("workspace_names") or {}).get("deals"))
        targets = [t for t in targets if t != "deals"]
        if not targets:
            return True
    data = {"question": intent.text, "targets": targets}
    if st.get("auto_ask"):
        await _run(eng, conv, st, {**data, "calls": len(targets)})
        return True
    where = " and ".join(f"{NAMES[a]}{_workspace_label(st, a)}" for a in targets)
    eng.confirmation(
        conv, st, f"I'll look across {where} to answer this.", "company_ask", data, calls=len(targets),
        cost_note=cost_line(len(targets), "one per agent" if len(targets) > 1 else None),
        also=("Yes, and don't ask again for questions in this chat", {"auto_ask": True}),
    )
    return True


def _stats_rows(stats: dict) -> list[list[str]]:
    rows = [["Deals", f"{stats['deals']} ({stats['won']} won, {stats['lost']} lost, {stats['open']} open)"]]
    if stats.get("win_rate") is not None:
        rows.append(["Win rate of closed deals", f"{round(stats['win_rate'] * 100)}%"])
    if stats.get("loss_reasons"):
        rows.append(["Loss reasons", ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in list(stats["loss_reasons"].items())[:4])])
    return rows


def _deal_card(question: str, result: dict) -> dict:
    section: dict[str, Any] = {"text": result.get("answer", "")}
    if result.get("sources"):
        section["chips"] = list(result["sources"])
    sections = [section]
    if result.get("found") is False and not result.get("sources"):
        sections.append({"text": "Your deals don't answer this, so I'm not guessing."})
    elif not result.get("grounded", True):
        sections.append({"heading": "Unverified", "text": "I could not tie this to specific deals, so treat it as a hint and check them."})
    if result.get("stats"):
        sections.append({"heading": "The counts it used (computed in code)", "rows": _stats_rows(result["stats"])})
    if result.get("truncated"):
        sections.append({"text": f"It looked at the {result['deals_considered']} most recent of {result['deals_total']} deals; "
                                 "the counts cover all of them."})
    tone = "warn" if result.get("found") and not result.get("grounded", True) else "info"
    return {"title": "From your deals", "subtitle": question, "tone": tone, "sections": sections}


def _library_card(question: str, result: dict) -> dict:
    section: dict[str, Any] = {"text": result.get("answer", "")}
    if result.get("sources"):
        section["chips"] = list(result["sources"])
    sections = [section]
    if result.get("found") is False and not result.get("sources"):
        sections.append({"text": "The facts and approved answers don't cover this, so I'm not guessing."})
    elif not result.get("grounded", True):
        sections.append({"heading": "Unverified", "text": "I could not tie this to a fact or an approved answer, so don't send it to a client as is."})
    if result.get("evidence"):
        sections.append({"heading": "Based on", "bullets": [f"{e['id']}: {e['label']}" for e in result["evidence"]]})
    if result.get("degraded"):
        why = f" ({result['warning']})" if result.get("warning") else ""
        sections.append({"heading": "Note", "text": "Library search was unavailable, so this leans on the company facts only." + why})
    tone = "warn" if result.get("found") and not result.get("grounded", True) else "info"
    return {"title": "From your RFP library and company facts", "subtitle": question, "tone": tone, "sections": sections}


async def _run(eng: Engine, conv: str, st: dict, data: dict) -> None:
    targets, question = data["targets"], data["question"]
    if data.get("auto_ask"):
        st["auto_ask"] = True
    eng.approve(st, len(targets))
    cards: list[dict] = []
    used = 0
    for agent_id in targets:
        eng.spend(st, 1)
        try:
            if agent_id == "deals":
                cards.append(_deal_card(question, await eng.deal.ask_portfolio(question)))
            else:
                cards.append(_library_card(question, await eng.rfp.ask_library(question)))
            used += 1
        except (AgentDown, UpstreamError) as exc:
            eng.fail(conv, exc)
    if not cards:
        return
    links = [eng.link_card(a, "Open Deal Intelligence" if a == "deals" else "Open the RFP assistant") for a in targets]
    eng.say(conv, f"Here is what {'your records say' if used > 1 else 'the record says'}. This used {plural(used, 'model call')}.",
            cards=[*cards, *links])


async def on_action(eng: Engine, conv: str, st: dict, kind: str, data: dict) -> None:
    if kind != "company_ask":
        return eng.say(conv, "I don't know how to do that any more.")
    await _run(eng, conv, st, data)


_COUNT_WORDS = {
    "many", "deals", "deal", "opportunities", "opportunity", "open", "closed", "won", "lost", "active", "live", "win", "lose",
    "close", "have", "are", "there", "do", "did", "we", "our", "how", "right", "now", "currently", "total", "pipeline", "the",
    "in", "a", "of", "to", "got", "has", "had", "is", "my", "us", "count", "number",
}


_LIST_WORDS = {"what", "which", "those", "these", "list", "show", "display", "give", "me", "all", "any", "you", "please", "of",
               "one", "ones", "or", "and", "both", "every", "everything", "each", "currently"}


async def deal_count(eng: Engine, conv: str, st: dict, intent: Intent) -> bool:
    """"How many deals ..." is a count, so it is answered from the deal list for free. False when the question carries
    a condition the list cannot test (stalled, overdue, by owner...), so it goes to the model instead."""
    from ..intents import LOSS_REASON_WORDS, _REASON_PATTERNS, _tokens, parse_reason

    text = intent.text
    info = await eng.deal.status()
    deals = await eng.deal.list_deals()
    if not deals:
        await workspace.nudge_empty(eng, conv, st, info.name)
        return True
    explicit = bool(intent.slots.get("explicit"))
    reason = intent.slots.get("reason") if explicit else parse_reason(text)
    allowed = set(_COUNT_WORDS) | _tokens(info.name or "")  # naming the team's own workspace is not a condition
    if reason:
        pattern = dict(_REASON_PATTERNS)[reason]
        allowed |= _tokens(" ".join(m.group(0) for m in pattern.finditer(text)))
    industries = {(d.get("industry") or "").lower(): d.get("industry") for d in deals if d.get("industry")}
    segments = {(d.get("segment") or "").lower(): d.get("segment") for d in deals if d.get("segment")}
    industry = next((v for k, v in industries.items() if k and k in text.lower()), None)
    segment = next((v for k, v in segments.items() if k and k.replace("_", " ") in text.lower().replace("_", " ")), None)
    if explicit:
        wanted_industry = (intent.slots.get("industry") or "").strip().lower()
        wanted_segment = (intent.slots.get("segment") or "").strip().lower().replace(" ", "_").replace("-", "_")
        industry = industries.get(wanted_industry) or (wanted_industry and "__no_such_industry__") or None
        segment = segments.get(wanted_segment) or (wanted_segment and "__no_such_segment__") or None
    for word in (industry, segment):
        allowed |= _tokens((word or "").replace("_", " "))
    if intent.kind == "deal_list":
        allowed |= _LIST_WORDS
    def known(token: str) -> bool:  # a typo of a known word still counts ("halycon" for "halcyon")
        return token in allowed or (len(token) >= 5 and any(
            len(w) >= 5 and SequenceMatcher(None, token, w).ratio() >= 0.8 for w in allowed))

    if not explicit and not all(known(t) for t in _tokens(text)):
        return False

    words = set(re.findall(r"[a-z]+", text.lower()))
    mentioned = {w for w in ("open", "active", "live", "closed", "won", "lost") if w in words}
    kind = (intent.slots.get("filter") or "").lower()
    if explicit:
        mentioned = set()  # the model already said which statuses it means
    if len(mentioned) > 1:
        kind = ""  # "the active ones and the ones we won or lost" asks for all of them
    elif not kind:
        kind = next(iter(mentioned), "")
    if reason and kind in ("", "closed"):
        kind = "lost"
    chosen = [d for d in deals if (not industry or d.get("industry") == industry) and (not segment or d.get("segment") == segment)]

    def of(result: str) -> int:
        return sum(1 for d in chosen if d.get("result") == result)

    counts = {"open": of("open"), "won": of("won"), "lost": of("lost")}
    if reason:
        counts["lost"] = sum(1 for d in chosen if d.get("result") == "lost" and d.get("loss_reason") == reason)
    total = {"": len(chosen), "active": counts["open"], "live": counts["open"], "open": counts["open"], "won": counts["won"],
             "lost": counts["lost"], "closed": counts["won"] + counts["lost"]}[kind]
    if intent.kind == "deal_list":
        return _list_deals(eng, conv, chosen, kind, reason, info.name)
    label = {"active": "open", "live": "open"}.get(kind, kind)
    where = " ".join(w for w in (industry, (segment or "").replace("_", " ")) if w)
    what = " ".join(w for w in (label, where, "deals" if total != 1 else "deal") if w)
    why = f" lost to {LOSS_REASON_WORDS[reason][0]}" if reason else ""
    ws = f" in {info.name}" if info.name else ""
    rows = [["Deals", str(len(chosen))], ["Open", str(of("open"))], ["Won", str(of("won"))], ["Lost", str(of("lost"))]]
    headline = f"{counts['lost']} {'deal' if counts['lost'] == 1 else 'deals'}{' ' + where if where else ''}{why}" if reason         else f"{total} {what}"
    eng.say(conv, f"{headline}{ws}. This used no model calls.",
            cards=[{"title": "Your deals" + (f" ({where})" if where else ""), "tone": "info", "sections": [{"rows": rows}]},
                   eng.link_card("deals", "Open Deal Intelligence")],
            actions=[{"id": "say:Why do we lose deals?", "label": "Why do we lose deals?", "style": "ghost"}])
    return True


LIST_CAP = 30


def _list_deals(eng: Engine, conv: str, deals: list[dict], kind: str, reason: str | None, workspace_name: str | None) -> bool:
    from ..intents import LOSS_REASON_WORDS

    def keep(d: dict) -> bool:
        result = d.get("result")
        if kind in ("open", "active", "live"):
            return result == "open"
        if kind == "won":
            return result == "won"
        if kind == "closed":
            return result in ("won", "lost")
        if kind == "lost":
            return result == "lost" and (not reason or d.get("loss_reason") == reason)
        return True

    kept = [d for d in deals if keep(d)]
    label = {"active": "open", "live": "open"}.get(kind, kind)
    ws = f" in {workspace_name}" if workspace_name else ""
    if not kept:
        eng.say(conv, f"No {label + ' ' if label else ''}deals{ws} match that. This used no model calls.")
        return True
    bullets = []
    for d in kept[:LIST_CAP]:
        state = f"open, {d.get('stage') or 'stage unknown'}" if d.get("result") == "open" else str(d.get("result"))
        if d.get("result") == "lost" and d.get("loss_reason"):
            state += f" ({d['loss_reason'].replace('_', ' ')})"
        amount = f", {d['amount']:,}" if isinstance(d.get("amount"), int) else ""
        bullets.append(f"{d.get('code')} {d.get('name')} ({d.get('account')}): {state}{amount}")
    sections: list[dict] = [{"bullets": bullets}]
    if len(kept) > LIST_CAP:
        sections.append({"text": f"Showing the first {LIST_CAP} of {len(kept)}. Open Deal Intelligence for the rest."})
    why = f" lost to {LOSS_REASON_WORDS[reason][0]}" if reason else ""
    n = len(kept)
    eng.say(conv, f"{n} {label + ' ' if label else ''}{'deal' if n == 1 else 'deals'}{why}{ws}. This used no model calls.",
            cards=[{"title": "Your deals", "tone": "info", "sections": sections}, eng.link_card("deals", "Open Deal Intelligence")],
            actions=[{"id": "say:Why do we lose deals?", "label": "Why do we lose deals?", "style": "ghost"}])
    return True
