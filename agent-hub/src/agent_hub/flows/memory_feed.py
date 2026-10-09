"""Feeding the agents' memory from the chat: the company's official facts and its past proposals (RFP Memory Assistant), and
the company's own sales plays (Deal Intelligence).

Nothing is saved without a click. Facts and plays are read by plain code (no model) and shown first; a past proposal costs one
model call to read, and its question-and-answer pairs are shown before they become approved answers. Everything goes to the
workspace the chat's company was given, and each save is written to the activity log when sign-in is on."""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import filepeek
from ..clients import WorkspaceInfo
from ..engine import cost_line, plural
from .rfp import RFP_EXTENSIONS, _ws_text

if TYPE_CHECKING:
    from ..engine import Engine
    from ..intents import Intent
    from ..store import Upload

LIST_EXTENSIONS = (".json", ".csv", ".txt", ".md")
MAX_ITEMS = 300
RESULT_WORDS = {"won": "won", "lost": "lost", "no_decision": "no decision", "unknown": "outcome not known"}
OBJECTION_WORDS = {
    "sso": "sso", "single sign on": "sso", "single sign-on": "sso", "saml": "sso",
    "security": "security_review", "security review": "security_review", "infosec": "security_review",
    "price": "pricing", "pricing": "pricing", "budget": "pricing", "cost": "pricing", "discount": "pricing",
    "integration": "integration", "integrations": "integration", "api": "integration",
    "timeline": "timeline", "timing": "timeline", "time": "timeline", "deadline": "timeline",
    "legal": "legal_terms", "legal terms": "legal_terms", "contract": "legal_terms", "terms": "legal_terms",
    "data residency": "data_residency", "residency": "data_residency", "hosting": "data_residency",
    "support": "support", "feature gap": "feature_gap", "features": "feature_gap", "feature": "feature_gap", "missing feature": "feature_gap",
}
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_VALID_TO = re.compile(r"\b(?:valid (?:until|to|through)|expires?(?: on)?|until|till)\s+(\d{4}-\d{2}-\d{2})\b", re.I)
_TAGGED = re.compile(r"[\[(]\s*(?P<k>objections?|answers|addresses|for|category|type)\s*:\s*(?P<v>[^\])]+)[\])]", re.I)


# -- reading what the person sent ----------------------------------------------------------------------------

def _lines(text: str) -> list[str]:
    if "\n" not in text.strip():  # one line from the chat box: bullets or semicolons separate the items
        text = re.sub(r"\s+(?:[-*•]|\d+[.)])\s+", "\n", " " + text)
        text = text.replace("; ", "\n")
    out = []
    for raw in text.splitlines():
        line = _BULLET.sub("", raw).strip()
        if not line or line.startswith("#") or (line.endswith(":") and len(line) < 60):
            continue  # blank lines, headings and "Security:" style labels are not items
        out.append(line)
    return out


def _date(value: Any) -> str | None:
    try:
        return date.fromisoformat(str(value).strip()[:10]).isoformat() if value else None
    except ValueError:
        return None


def parse_facts(text: str, filename: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
    """Facts from a fact sheet (.json), a table with a statement column (.csv) or one fact per line. (facts, company name)."""
    ext = Path(filename or "").suffix.lower()
    facts: list[dict[str, Any]] = []
    company = None
    if ext == ".json":
        data = json.loads(text)
        items = data.get("facts", []) if isinstance(data, dict) else data
        company = data.get("company") if isinstance(data, dict) else None
        for item in items if isinstance(items, list) else []:
            if isinstance(item, str):
                item = {"statement": item}
            statement = " ".join(str(item.get("statement") or item.get("fact") or item.get("text") or "").split())
            if statement:
                facts.append({"topic": str(item.get("topic") or "").strip()[:80], "statement": statement[:1000],
                              "valid_to": _date(item.get("valid_to") or item.get("valid_until") or item.get("expires"))})
    elif ext == ".csv":
        rows = list(csv.reader(io.StringIO(text)))
        header = [h.strip().lower() for h in rows[0]] if rows else []
        col = next((header.index(n) for n in ("statement", "fact", "text") if n in header), None)
        body = rows[1:] if col is not None else rows
        col = col or 0
        for row in body:
            if len(row) > col and row[col].strip():
                get = lambda name: row[header.index(name)].strip() if name in header and header.index(name) < len(row) else ""  # noqa: E731
                facts.append({"topic": get("topic")[:80], "statement": " ".join(row[col].split())[:1000],
                              "valid_to": _date(get("valid_to") or get("valid_until") or get("expires"))})
    else:
        for line in _lines(text):
            if len(line.split()) < 3:
                continue  # a title such as "Our facts" or "Security", not a fact
            topic, statement = "", line
            head, sep, rest = line.partition(":")
            if sep and rest.strip() and len(head) <= 40 and len(head.split()) <= 5 and "." not in head:
                topic, statement = head.strip(), rest.strip()
            match = _VALID_TO.search(statement)
            facts.append({"topic": topic[:80], "statement": statement[:1000], "valid_to": match.group(1) if match else None})
    return facts[:MAX_ITEMS], company


def _objections(value: Any) -> list[str]:
    parts = value if isinstance(value, list) else re.split(r"[,/;|]| and ", str(value or ""))
    out = []
    for part in parts:
        word = " ".join(str(part).lower().split())
        if word:
            out.append(OBJECTION_WORDS.get(word, word.replace(" ", "_").replace("-", "_")))
    return list(dict.fromkeys(out))


def parse_plays(text: str, filename: str | None = None) -> list[dict[str, Any]]:
    """Plays from JSON, a table with a name column (.csv) or one play per line: "Name: what we do [objections: pricing]"."""
    ext = Path(filename or "").suffix.lower()
    plays: list[dict[str, Any]] = []
    if ext == ".json":
        data = json.loads(text)
        items = data.get("plays", []) if isinstance(data, dict) else data
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and str(item.get("name") or "").strip():
                plays.append({"name": str(item["name"]).strip(), "description": str(item.get("description") or "").strip(),
                              "category": str(item.get("category") or "process").strip().lower(),
                              "addresses": _objections(item.get("addresses") or item.get("objections") or [])})
    elif ext == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        for row in rows:
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            if row.get("name"):
                plays.append({"name": row["name"], "description": row.get("description", ""), "category": (row.get("category") or "process").lower(),
                              "addresses": _objections(row.get("addresses") or row.get("objections") or "")})
    else:
        for line in _lines(text):
            category, addresses = "process", []
            for tag in _TAGGED.finditer(line):
                if tag.group("k").lower() in ("category", "type"):
                    category = tag.group("v").strip().lower()
                else:
                    addresses += _objections(tag.group("v"))
            line = _TAGGED.sub("", line).strip()
            name, description = line, ""
            for sep in (": ", " - ", " – ", " — "):
                if sep in line:
                    name, description = (part.strip() for part in line.split(sep, 1))
                    break
            if name:
                plays.append({"name": name[:120], "description": description, "category": category, "addresses": list(dict.fromkeys(addresses))})
    return plays[:100]


_INDUSTRIES = ("banking", "finance", "financial services", "insurance", "healthcare", "pharma", "retail", "government", "public sector",
               "telecom", "manufacturing", "energy", "utilities", "education", "logistics", "media", "technology", "software", "automotive")
_MONTHS = {m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august", "september",
                                        "october", "november", "december"), start=1)}


def proposal_slots(text: str) -> dict[str, Any]:
    """Client, industry, result, loss reason and date from a sentence like "past proposal we lost on technical fit for Acme
    Bank (banking), submitted 2025-03-10". Anything not said is left out (and asked for if it matters)."""
    slots: dict[str, Any] = {}
    low = text.lower()
    if re.search(r"\bno[ -]decision\b", low):
        slots["result"] = "no_decision"
    elif re.search(r"\b(?:won|win)\b", low):
        slots["result"] = "won"
    elif re.search(r"\blost\b", low):
        slots["result"] = "lost"
        m = re.search(r"\blost\b(?:\s+(?:it|this|that|the bid))?\s+(?:on|because of|because|due to|over|to)\s+(?P<r>[^.,;()]+)", text, re.I)
        if m:
            reason = " ".join(m.group("r").split()).strip()
            reason = re.split(r"\s+(?:for|in|submitted|from)\s+", reason)[0]
            slots["loss_reason"] = "technical fit" if reason.lower() in ("tech fit", "technical", "technical fit") else reason
    m = re.search(r"\b(?:for|to|with)\s+(?P<c>[A-Z][\w&.'\-]*(?:\s+[A-Z][\w&.'\-]*)*)", text)
    if m:
        slots["client"] = m.group("c").strip()
    industry = next((i for i in _INDUSTRIES if re.search(rf"\b{re.escape(i)}\b", low)), None)
    if industry:
        slots["industry"] = industry
    iso = re.search(r"\b(20\d\d-\d\d-\d\d)\b", text)
    month = re.search(r"\b(" + "|".join(_MONTHS) + r")\s+(20\d\d)\b", low)
    if iso and _date(iso.group(1)):
        slots["submitted_on"] = iso.group(1)
    elif month:
        slots["submitted_on"] = f"{month.group(2)}-{_MONTHS[month.group(1)]:02d}-01"
    return slots


def _describe(slots: dict[str, Any]) -> str:
    parts = []
    if slots.get("client"):
        parts.append("for " + slots["client"] + (f" ({slots['industry']})" if slots.get("industry") else ""))
    elif slots.get("industry"):
        parts.append(f"in {slots['industry']}")
    result = RESULT_WORDS.get(slots.get("result") or "unknown", "outcome not known")
    if slots.get("result") == "lost" and slots.get("loss_reason"):
        result += f" ({slots['loss_reason']})"
    parts.append(result)
    if slots.get("submitted_on"):
        parts.append(f"submitted {slots['submitted_on']}")
    return ", ".join(parts)


def _said(intent: Intent) -> dict[str, Any]:
    return {k: v for k, v in (intent.slots or {}).items() if k in ("client", "industry", "submitted_on", "result", "loss_reason") and v}


def _details_rows(slots: dict[str, Any]) -> list[list[str]]:
    """Each detail, where it came from, and what is missing: checked before the one model call is spent."""
    from_file = set(slots.get("from_file") or [])

    def show(key: str, value: str | None) -> str:
        if not value:
            return "not given"
        return f"{value} (read from the file)" if key in from_file else value

    result = RESULT_WORDS.get(slots.get("result") or "unknown", "outcome not known")
    if slots.get("result") == "lost" and slots.get("loss_reason"):
        result += f" ({slots['loss_reason']})"
    return [["Client", show("client", slots.get("client"))], ["Industry", show("industry", slots.get("industry"))],
            ["Submitted", show("submitted_on", slots.get("submitted_on"))], ["Outcome", result]]


def _norm(statement: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", statement.lower()))


def _audit(eng: Engine, conv: str, action: str, detail: str) -> None:
    if eng.auth is None:
        return
    user = eng.auth.user_by_id(eng.store.conversation_owner(conv))
    eng.auth.audit(user.email if user else "chat", action, detail)


def _upload_text(upload: Upload) -> str:
    return upload.read().decode("utf-8", errors="replace")


# -- starting from a message -------------------------------------------------------------------------------------

def inline_part(raw: str) -> str:
    """What follows the request itself: the lines after the first one, or what follows the first colon."""
    raw = (raw or "").strip()
    if "\n" in raw:
        return raw.split("\n", 1)[1]
    head, sep, rest = raw.partition(":")
    return rest if sep else ""


async def start(eng: Engine, conv: str, st: dict, intent: Intent, raw: str) -> None:
    if intent.kind == "mem_facts":
        return await _facts(eng, conv, st, inline_part(raw))
    if intent.kind == "mem_plays":
        return await _plays(eng, conv, st, inline_part(raw))
    return await _proposal(eng, conv, st, {**proposal_slots(raw), **{k: v for k, v in intent.slots.items() if v not in (None, "")}})


async def handle_slot(eng: Engine, conv: str, st: dict, intent: Intent, accepted: list[Upload], raw: str) -> None:
    awaiting = st.get("awaiting") or {}
    st["awaiting"] = None
    kind = awaiting.get("type")
    if kind == "mem_facts_input":
        return await _facts(eng, conv, st, raw)
    if kind == "mem_plays_input":
        return await _plays(eng, conv, st, raw)
    if kind == "mem_proposal_file":
        return await _proposal(eng, conv, st, {**(awaiting.get("slots") or {}), **proposal_slots(raw), **_said(intent)})
    if kind == "mem_proposal_details":
        upload = eng.store.get_upload(awaiting.get("upload_id") or "")
        if upload is None:
            return eng.say(conv, "I can't find that file any more. Attach it again and I'll start over.")
        slots = {**(awaiting.get("slots") or {}), **proposal_slots(raw), **_said(intent)}
        slots["from_file"] = [k for k in (awaiting.get("slots") or {}).get("from_file", []) if k not in proposal_slots(raw) and k not in _said(intent)]
        return await _confirm_proposal(eng, conv, st, upload, slots)
    eng.say(conv, "Okay. Tell me what you'd like to do.")


# -- company facts (RFP) --------------------------------------------------------------------------------------------

async def _rfp_company(eng: Engine, conv: str) -> tuple[WorkspaceInfo, dict] | None:
    ws = await eng.rfp.workspace_info()
    company = await eng.rfp.get_company()
    if company.get("locked"):
        eng.say(conv, f"The RFP workspace this chat uses{_ws_text(ws)} is the built-in sample, which can't hold your company's data. "
                      "An administrator can give your company its own data: open the admin page and click \"Set up its data\" on your company.")
        return None
    return ws, company


async def _facts(eng: Engine, conv: str, st: dict, typed: str) -> None:
    files = eng.pending_uploads(st, LIST_EXTENSIONS)
    try:
        if files:
            facts, named = parse_facts(_upload_text(files[-1]), files[-1].filename)
        else:
            facts, named = parse_facts(typed) if typed.strip() else ([], None)
    except (ValueError, json.JSONDecodeError) as exc:
        return eng.say(conv, f"I couldn't read that file as a fact sheet ({type(exc).__name__}). Send a .json fact sheet, a .csv with a "
                             "\"statement\" column, or one fact per line in a .txt or .md file or in the chat.")
    if not facts:
        st["awaiting"] = {"type": "mem_facts_input"}
        return eng.say(conv, "Send the facts: attach a fact sheet (.json, .csv, .txt or .md, one fact per line), or type them here, one "
                             "per line. Only things you're allowed to promise in a proposal. Optional: start a line with a short topic and "
                             "a colon (\"Certifications: ...\"), and end it with \"valid until 2027-03-31\" if it expires.")
    found = await _rfp_company(eng, conv)
    if found is None:
        return
    ws, company = found
    known = {_norm(f.get("statement", "")) for f in company.get("facts") or []}
    new = [f for f in facts if _norm(f["statement"]) not in known]
    if not new:
        return eng.say(conv, f"All {plural(len(facts), 'fact')} are already in the company facts{_ws_text(ws)}. Nothing to add.")
    bullets = [(f"{f['topic']}: " if f["topic"] else "") + f["statement"] + (f" (until {f['valid_to']})" if f.get("valid_to") else "")
               for f in new[:12]]
    card = {"title": f"{plural(len(new), 'company fact')} to add", "subtitle": (f"{len(facts) - len(new)} already there. " if len(facts) > len(new) else "")
            + "Drafted answers may cite only these facts and approved past answers.", "tone": "info",
            "sections": [{"bullets": bullets}] + ([{"text": f"and {len(new) - 12} more."}] if len(new) > 12 else [])}
    eng.say(conv, "Here is what I read. Check it before I save it:", cards=[card])
    name = company.get("company") or named or st.get("company_name") or ws.name or "Our company"
    data = {"facts": new, "company": name, "workspace_id": ws.id, "upload_id": files[-1].id if files else None}
    eng.confirmation(conv, st, f"Save {plural(len(new), 'fact')} as {name}'s official facts{_ws_text(ws)}?", "mem_facts_save", data,
                     calls=0, cost_label="free", label=f"Save {plural(len(new), 'fact')}")


async def _save_facts(eng: Engine, conv: str, st: dict, data: dict) -> None:
    found = await _rfp_company(eng, conv)
    if found is None:
        return
    ws, company = found
    if data.get("workspace_id") and ws.id != data["workspace_id"]:
        return eng.say(conv, "The RFP workspace changed since I read the facts, so I didn't save them. Send them again.")
    existing = [{k: f.get(k) for k in ("id", "topic", "statement", "valid_to")} for f in company.get("facts") or []]
    known = {_norm(f["statement"] or "") for f in existing}
    add = [f for f in data["facts"] if _norm(f["statement"]) not in known]
    saved = await eng.rfp.save_company(company.get("company") or data["company"], existing + add)
    if data.get("upload_id"):
        st["uploads"] = [u for u in st["uploads"] if u != data["upload_id"]]
    _audit(eng, conv, "rfp.facts", f"added {len(add)} company fact(s) in {ws.name or ws.id}")
    eng.say(conv, f"Saved {plural(len(add), 'new fact')}. The company facts{_ws_text(ws)} now hold {plural(len(saved.get('facts') or []), 'fact')}; "
                  "drafted answers can cite them from now on. Next you can add past proposals: attach one and say whether it was won or lost.")


# -- past proposals (RFP) -------------------------------------------------------------------------------------------

async def _proposal(eng: Engine, conv: str, st: dict, slots: dict[str, Any]) -> None:
    files = eng.pending_uploads(st, RFP_EXTENSIONS)
    if not files:
        st["awaiting"] = {"type": "mem_proposal_file", "slots": slots}
        return eng.say(conv, "Attach the past proposal (.docx, .pdf, .xlsx, .txt or .md) and tell me how it went, for example "
                             "\"we lost this on technical fit, for Acme Bank (banking), submitted 2025-03-10\".")
    if await _rfp_company(eng, conv) is None:
        return
    upload = files[-1]
    found = filepeek.proposal_details(filepeek.read_text(upload))  # what the document says about itself, read here
    slots = {**slots, "from_file": [k for k in found if not slots.get(k)]}
    for key in slots["from_file"]:
        slots[key] = found[key]
    if not slots.get("result"):
        choices = [eng.offer(st, "mem_proposal_result", label, data={"upload_id": upload.id, "slots": {**slots, "result": value}},
                             style="primary" if value in ("won", "lost") else "ghost", group="mem_result")
                   for value, label in (("won", "Won"), ("lost", "Lost"), ("no_decision", "No decision"), ("unknown", "Don't know"))]
        return eng.say(conv, f"Was {upload.filename} won or lost? The memory learns most from the answers in bids that were won, and "
                             "from losses where the answers themselves were the problem.", actions=choices)
    await _confirm_proposal(eng, conv, st, upload, slots)


async def _confirm_proposal(eng: Engine, conv: str, st: dict, upload: Upload, slots: dict[str, Any]) -> None:
    ws = await eng.rfp.workspace_info()
    hint = " If it was lost because of the answers, say \"lost on technical fit\": that is the loss the memory learns from." \
        if slots.get("result") == "lost" and not slots.get("loss_reason") else ""
    missing = [label.lower() for label, value in _details_rows(slots) if value == "not given"]
    gap = f" The {' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not given; the memory uses them to rank answers for " \
          "similar clients, so add them if you know them." if missing else ""
    eng.say(conv, "", cards=[{"title": f"Past proposal: {upload.filename}", "tone": "warn" if missing else "info",
                              "subtitle": "Check these details. They are saved with every answer from this proposal.",
                              "sections": [{"rows": _details_rows(slots)}]}])
    eng.confirmation(
        conv, st, f"Ready to add it to the RFP memory{_ws_text(ws)}. I'll read it and pull out its questions and answers, then show them "
                  f"to you before anything is saved.{gap}{hint}",
        "mem_proposal", {"upload_id": upload.id, "slots": slots, "workspace_id": ws.id},
        calls=1, cost_note=cost_line(1, "to read the proposal"), label="Yes, read it",
        extra=[eng.offer(st, "mem_proposal_edit", "Change the details", style="ghost", data={"upload_id": upload.id, "slots": slots})])


async def _start_proposal(eng: Engine, conv: str, st: dict, data: dict) -> None:
    ws = await eng.rfp.workspace_info()
    if data.get("workspace_id") and ws.id != data["workspace_id"]:
        return eng.say(conv, "The RFP workspace changed since you asked, so I stopped. Ask me again.")
    if eng.store.get_upload(data["upload_id"]) is None:
        return eng.say(conv, "I can't find that file any more. Attach it again and I'll start over.")
    eng.approve(st, 1)
    st["run"] = {"flow": "mem_proposal", "step": "upload_start", "upload_id": data["upload_id"], "slots": data.get("slots") or {},
                 "workspace_id": ws.id, "proposal_id": None, "job_id": None, "track": None}
    eng.spawn_run(conv, run)


async def run(eng: Engine, conv: str) -> None:
    async with eng.lock(conv):
        st = eng.load(conv)
        state = st.get("run")
        if not state or state.get("flow") != "mem_proposal":
            return
        try:
            if state["step"] == "upload_start":
                if state.get("resumed"):
                    st["run"] = None
                    return eng.say(conv, "I was interrupted just before reading the proposal and can't tell whether it began, so I "
                                         "stopped without spending anything more. Attach it again if it isn't in the library.")
                if not eng.spend(st, 1):
                    st["run"] = None
                    return eng.say(conv, "I stopped before the model call because it would go past what you approved.")
                state["resumed"] = False
                eng.save(conv, st)  # on disk before the call, so a restart never repeats the spend
                upload = eng.store.get_upload(state["upload_id"])
                slots = state.get("slots") or {}
                view = await eng.rfp.import_proposal(upload.filename, upload.read(), **{k: slots.get(k) for k in (
                    "client", "industry", "submitted_on", "result", "loss_reason")})
                pid, job_id = view["id"], (view.get("job") or {}).get("id")
                track = job_id if job_id is not None else f"proposal-{pid}"
                eng.store.add_job(conv, "rfp", track, "library")
                st["uploads"] = [u for u in st["uploads"] if u != state["upload_id"]]
                state.update(step="extract", proposal_id=pid, job_id=job_id, track=track, filename=upload.filename)
            pid, track = state["proposal_id"], state["track"]
        finally:
            eng.save(conv, st)
    view = await eng.poll(conv, app="rfp", step="library", job_id=track, fetch=lambda: eng.rfp.get_proposal(pid),
                          done=lambda p: p.get("status") != "extracting", progress=lambda p: (0, 0),
                          label="Reading the past proposal", final_status=lambda p: str(p.get("status")))
    async with eng.lock(conv):
        st = eng.load(conv)
        state = st.get("run") or {}
        st["run"] = None
        try:
            _after_reading(eng, conv, st, view, state.get("filename") or view.get("filename") or "the proposal", state.get("workspace_id"))
        finally:
            eng.save(conv, st)


def _after_reading(eng: Engine, conv: str, st: dict, view: dict, filename: str, workspace_id: str | None) -> None:
    pid = view["id"]
    if view.get("status") != "extracted":
        why = (view.get("error_info") or {}).get("message") or view.get("error") or "the RFP assistant could not read it"
        return eng.error(conv, f"I couldn't read {filename}: {why}. Nothing was added to the memory.")
    pairs = view.get("pairs") or []
    if not pairs:
        return eng.say(conv, f"I read {filename} but found no question-and-answer pairs in it, so there is nothing to add.",
                       actions=[eng.offer(st, "mem_proposal_discard", "Discard this import", style="ghost", data={"proposal_id": pid})])
    card = {"title": f"{plural(len(pairs), 'question-and-answer pair')} in {filename}", "tone": "info",
            "subtitle": "Kept pairs become approved answers the RFP assistant can reuse and cite.",
            "sections": [{"bullets": [" ".join(p["question"].split())[:160] for p in pairs[:8]]}]
            + ([{"text": f"and {len(pairs) - 8} more."}] if len(pairs) > 8 else [])}
    group = f"mem_pairs_{pid}"
    eng.say(conv, "Here is what I found. Keep them all, or discard the import if it looks wrong:", cards=[card], actions=[
        eng.offer(st, "mem_proposal_keep", f"Keep all {len(pairs)}", data={"proposal_id": pid, "pair_ids": [p["id"] for p in pairs],
                                                                        "workspace_id": workspace_id, "filename": filename}, group=group),
        eng.offer(st, "mem_proposal_discard", "Discard", style="ghost", data={"proposal_id": pid, "filename": filename}, group=group)])


async def _keep(eng: Engine, conv: str, st: dict, data: dict) -> None:
    ws = await eng.rfp.workspace_info()
    if data.get("workspace_id") and ws.id != data["workspace_id"]:
        return eng.say(conv, "The RFP workspace changed since I read the proposal, so I didn't save it. Switch back and ask again.")
    result = await eng.rfp.keep_pairs(data["proposal_id"], data["pair_ids"])
    added = len(result.get("created") or [])
    _audit(eng, conv, "rfp.library", f"added {added} approved answer(s) from {data.get('filename')} in {ws.name or ws.id}")
    eng.say(conv, f"Added {plural(added, 'approved answer')} from {data.get('filename')} to the RFP memory{_ws_text(ws)}. "
                  "Add more past proposals the same way, or attach a new RFP and say \"answer this\".")


# -- plays (Deal) ------------------------------------------------------------------------------------------------------

async def _plays(eng: Engine, conv: str, st: dict, typed: str) -> None:
    files = eng.pending_uploads(st, LIST_EXTENSIONS)
    try:
        plays = parse_plays(_upload_text(files[-1]), files[-1].filename) if files else (parse_plays(typed) if typed.strip() else [])
    except (ValueError, json.JSONDecodeError) as exc:
        return eng.say(conv, f"I couldn't read that file ({type(exc).__name__}). Send a .csv with a \"name\" column, a .json list, or one "
                             "play per line.")
    if not plays:
        st["awaiting"] = {"type": "mem_plays_input"}
        return eng.say(conv, "Send your plays, the moves your team uses to move a deal forward: attach a .csv (name, description, category, "
                             "objections), a .json list, or type one per line, for example \"Security review pack: send our SOC 2 report and "
                             "security answers [objections: security, sso]\". Objections it understands: sso, security review, pricing, "
                             "integration, timeline, legal terms, data residency, support, feature gap.")
    ws = await eng.deal.status()
    known = {p["name"].strip().lower() for p in await eng.deal.list_plays()}
    bullets = [p["name"] + (f": {p['description']}" if p["description"] else "") + (f" [{', '.join(p['addresses'])}]" if p["addresses"] else "")
               + (" (updates the existing one)" if p["name"].strip().lower() in known else "") for p in plays[:15]]
    card = {"title": f"{plural(len(plays), 'play')} to save", "tone": "info",
            "subtitle": "Briefs recommend these and learn which ones win in your deals.",
            "sections": [{"bullets": bullets}] + ([{"text": f"and {len(plays) - 15} more."}] if len(plays) > 15 else [])}
    eng.say(conv, "Here is what I read. Check it before I save it:", cards=[card])
    where = f" in {ws.name}" if ws.name else ""
    eng.confirmation(conv, st, f"Save {plural(len(plays), 'play')} to Deal Intelligence{where}?", "mem_plays_save",
                     {"plays": plays, "workspace_id": ws.id, "upload_id": files[-1].id if files else None},
                     calls=0, cost_label="free", label=f"Save {plural(len(plays), 'play')}")


async def _save_plays(eng: Engine, conv: str, st: dict, data: dict) -> None:
    ws = await eng.deal.status()
    if data.get("workspace_id") and ws.id != data["workspace_id"]:
        return eng.say(conv, "The Deal Intelligence workspace changed since I read the plays, so I didn't save them. Send them again.")
    result = await eng.deal.save_plays(data["plays"])
    if data.get("upload_id"):
        st["uploads"] = [u for u in st["uploads"] if u != data["upload_id"]]
    added, updated, ignored = result.get("added") or [], result.get("updated") or [], result.get("ignored_objections") or []
    _audit(eng, conv, "deal.plays", f"added {len(added)}, updated {len(updated)} play(s) in {ws.name or ws.id}")
    text = f"Saved: {plural(len(added), 'new play')}" + (f", {plural(len(updated), 'play')} updated" if updated else "") + \
        f". Deal Intelligence now has {plural(len(result.get('plays') or []), 'play')}."
    if ignored:
        text += (f" I left out objections it doesn't know: {', '.join(ignored)}. It understands sso, security review, pricing, integration, "
                 "timeline, legal terms, data residency, support and feature gap.")
    eng.say(conv, text + " Next, add deals: attach emails or call notes and say \"New deal <name> at <company>\".")


# -- buttons ---------------------------------------------------------------------------------------------------------------

async def on_action(eng: Engine, conv: str, st: dict, kind: str, data: dict) -> None:
    if kind == "mem_facts_save":
        return await _save_facts(eng, conv, st, data)
    if kind == "mem_plays_save":
        return await _save_plays(eng, conv, st, data)
    if kind == "mem_proposal_result":
        upload = eng.store.get_upload(data["upload_id"])
        if upload is None:
            return eng.say(conv, "I can't find that file any more. Attach it again.")
        return await _confirm_proposal(eng, conv, st, upload, data["slots"])
    if kind == "mem_proposal":
        return await _start_proposal(eng, conv, st, data)
    if kind == "mem_proposal_edit":
        eng.expire_confirms(st, keep=data["upload_id"])
        st["awaiting"] = {"type": "mem_proposal_details", "upload_id": data["upload_id"], "slots": data["slots"]}
        return eng.say(conv, "Tell me what to change, in your own words, for example \"it was for Acme Bank, banking, sent in March 2025, "
                             "and we lost on technical fit\". Anything you don't mention stays as it is.")
    if kind == "mem_proposal_keep":
        return await _keep(eng, conv, st, data)
    if kind == "mem_proposal_discard":
        await eng.rfp.discard_proposal(data["proposal_id"])
        return eng.say(conv, f"Discarded {data.get('filename') or 'the import'}. Nothing was added to the memory.")
    eng.say(conv, "I don't know how to do that any more.")
