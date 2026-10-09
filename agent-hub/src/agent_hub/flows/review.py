"""Reviewing drafted RFP answers one at a time in the chat: accept, edit, reject, or write the answer an expert gives.

The RFP assistant owns the drafts, the evidence and the learning. Every decision here is one call to its own review route
(`POST /v1/requirements/{id}/review`, the same one its own screens use), so how a review is scored (accepted, light or heavy
edit, rewritten, rejected), what it does to the library and which lessons it produces stay its decision. The hub only shows
the draft and its evidence, collects the person's decision and writes who made it to the activity log."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..clients import UpstreamError
from ..engine import plural

if TYPE_CHECKING:
    from ..engine import Engine

# The RFP assistant's own review reasons (rfp_assistant/api/v1/learning.py REASON_TAGS), in plain words.
REASONS = (("incorrect", "Incorrect"), ("outdated", "Outdated"), ("too_vague", "Too vague"), ("client_specific", "Needed client specifics"),
           ("too_long", "Too long"), ("too_short", "Too short"), ("tone", "Tone"), ("wrong_product", "Wrong product"), ("other", "Other"))
CHECKS = {
    "evidence_unsupported": "a claim isn't supported by the evidence it cites", "evidence_partial": "a claim is only partly supported",
    "invalid_citation": "it cites something it wasn't given", "unsupported_claims": "it says something no source supports",
    "over_word_limit": "it is over the word limit", "no_claims": "it cites no evidence at all", "empty_answer": "it has no answer",
}


def _reviewable(req: dict) -> bool:
    d = req.get("draft") or {}
    return bool(d) and not req.get("final") and d.get("status") in ("drafted", "needs_sme", "failed")


def _needs_writing(req: dict) -> bool:
    d = req.get("draft") or {}
    return d.get("status") != "drafted" or "[SME input required:" in (d.get("answer") or "") or not (d.get("answer") or "").strip()


def _audit(eng: Engine, conv: str, detail: str) -> None:
    if eng.auth is None:
        return
    user = eng.auth.user_by_id(eng.store.conversation_owner(conv))
    eng.auth.audit(user.email if user else "chat", "rfp.review", detail)


def _card(req: dict, left: int, project_name: str | None) -> dict:
    d = req.get("draft") or {}
    sections: list[dict[str, Any]] = [{"heading": "Question", "text": req.get("question") or ""}]
    if d.get("status") == "failed":
        sections.append({"heading": "No draft", "text": "The model couldn't draft this one. Write the answer yourself."})
    elif _needs_writing(req):
        text = d.get("sme_question") or "The sources didn't cover this."
        start = (d.get("answer") or "").split("Supported starting point:", 1)
        if len(start) == 2:
            text += "\n\nWhat the sources did support: " + start[1].split("Complete with:", 1)[0].strip()
        sections.append({"heading": "Sent to an expert", "text": text})
    else:
        sections.append({"heading": "Draft answer", "text": d.get("answer") or ""})
    claims = d.get("claims") or []
    if claims:
        sections.append({"heading": "Evidence it cites", "bullets": [
            f"{' '.join(str(c.get('text') or '').split())[:180]} ({', '.join(c.get('source_ids') or []) or 'no source'})" for c in claims[:6]]})
    elif d.get("status") == "drafted" and not _needs_writing(req):
        sections.append({"heading": "Evidence it cites", "text": "None."})
    checks = [CHECKS[f] for f in d.get("flags") or [] if f in CHECKS]
    if checks:
        sections.append({"heading": "Checks", "bullets": checks})
    tone = "warn" if checks or _needs_writing(req) else "ok"
    return {"title": f"{req.get('code', '')}: {plural(left, 'answer')} left to review", "subtitle": project_name, "tone": tone, "sections": sections}


# -- the loop ----------------------------------------------------------------------------------------------------

async def start(eng: Engine, conv: str, st: dict, project_id: int | None = None, workspace_id: str | None = None) -> None:
    from .rfp import _project_state, _same_workspace

    proj = _project_state(st)
    pid = project_id or proj.get("id")
    if not pid:
        return eng.say(conv, "There is no RFP in this chat to review yet. Attach one and say \"answer this\".")
    ws = await _same_workspace(eng, conv, workspace_id or proj.get("workspace_id"))
    if ws is None:
        return
    st["review"] = {"project_id": pid, "workspace_id": ws.id, "skipped": [], "reviewed": 0}
    await show_next(eng, conv, st)


async def show_next(eng: Engine, conv: str, st: dict) -> None:
    rv = st.get("review") or {}
    project = await eng.rfp.get_project(rv["project_id"])
    todo = [r for r in project.get("requirements") or [] if _reviewable(r) and r["id"] not in rv.get("skipped", [])]
    if not todo:
        return await _finish(eng, conv, st, project)
    req, left = todo[0], len(todo)
    group = f"rv_{req['id']}"
    data = {"rid": req["id"], "code": req.get("code"), "project_id": project["id"]}
    if _needs_writing(req):
        actions = [eng.offer(st, "rfp_rv_write", "Write the answer", data=data, group=group),
                   eng.offer(st, "rfp_rv_skip", "Skip for now", style="ghost", data=data, group=group)]
    else:
        actions = [eng.offer(st, "rfp_rv_accept", "Accept", data=data, group=group),
                   eng.offer(st, "rfp_rv_edit", "Edit", style="ghost", data=data, group=group),
                   eng.offer(st, "rfp_rv_reject", "Reject", style="danger", data=data, group=group),
                   eng.offer(st, "rfp_rv_skip", "Skip for now", style="ghost", data=data, group=group)]
    eng.say(conv, "", cards=[_card(req, left, project.get("name"))], actions=actions)


async def _finish(eng: Engine, conv: str, st: dict, project: dict) -> None:
    from .rfp import _breakdown, _offer_exports

    rv = st.get("review") or {}
    skipped = len(rv.get("skipped", []))
    st["review"] = None
    buckets = _breakdown(project)
    if project.get("state") in ("approved", "exported"):
        eng.say(conv, f"That's every answer reviewed ({plural(rv.get('reviewed', 0), 'decision')} in this round).")
        return _offer_exports(eng, conv, st, project)
    rows = [["Final answers", f"{buckets['final']} of {buckets['total']}"]]
    if skipped:
        rows.append(["Skipped for now", str(skipped)])
    if buckets["not_drafted"]:
        rows.append(["Never drafted", str(buckets["not_drafted"])])
    again = [eng.offer(st, "rfp_rv_again", "Review the skipped ones", style="primary", data={"project_id": project["id"]})] if skipped else []
    eng.say(conv, "Nothing else is waiting for you in this round." + (
        " Answers that were never drafted can't be reviewed here; they need drafting again." if buckets["not_drafted"] else ""),
        cards=[{"title": "Review so far", "subtitle": project.get("name"), "tone": "warn", "sections": [{"rows": rows}]}], actions=again)


# -- decisions -------------------------------------------------------------------------------------------------------

async def _submit(eng: Engine, conv: str, st: dict, data: dict, action: str, text: str | None = None, tags: list[str] | None = None) -> bool:
    try:
        await eng.rfp.review(data["rid"], action, text, reason_tags=tags or [])
    except UpstreamError as exc:
        eng.error(conv, f"The RFP assistant didn't take that: {exc.message}")
        return False
    rv = st.get("review") or {}
    rv["reviewed"] = rv.get("reviewed", 0) + 1
    _audit(eng, conv, f"{data.get('code')} in project {data['project_id']}: {action}" + (f" ({', '.join(tags)})" if tags else ""))
    return True


def _ask_reason(eng: Engine, conv: str, st: dict, data: dict, action: str, text: str | None, question: str, optional: bool) -> None:
    group = f"rvr_{data['rid']}"
    payload = {**data, "action": action, "text": text}
    actions = [eng.offer(st, "rfp_rv_reason", label, style="ghost", data={**payload, "tag": tag}, group=group) for tag, label in REASONS]
    if optional:
        actions.insert(0, eng.offer(st, "rfp_rv_reason", "Save without a reason", data={**payload, "tag": None}, group=group))
    eng.say(conv, question, actions=actions)


async def handle_text(eng: Engine, conv: str, st: dict, text: str) -> None:
    """The final answer the person typed after Edit or Write the answer."""
    awaiting = st.get("awaiting") or {}
    st["awaiting"] = None
    final = text.strip()
    if len(final) >= 2 and final[0] + final[-1] in ('""', "''", "\u201c\u201d", "\u2018\u2019") and final[0] not in final[1:-1]:
        final = final[1:-1].strip()  # pasted from a spreadsheet cell: the quotes wrap the whole answer
    if not final:
        return eng.say(conv, "That was empty, so nothing changed.")
    data = {k: awaiting.get(k) for k in ("rid", "code", "project_id")}
    if awaiting.get("write"):  # the answer an expert gives: there was no usable draft to critique
        if await _submit(eng, conv, st, data, "edited", final):
            eng.say(conv, f"Saved {data['code']} as final. It is now an approved answer the RFP assistant can reuse.")
            await show_next(eng, conv, st)
        else:
            st["awaiting"] = awaiting  # let them try again
        return
    _ask_reason(eng, conv, st, data, "edited", final, f"Saved your text for {data['code']}. Why did you change it? It helps the assistant learn.", optional=True)


async def on_action(eng: Engine, conv: str, st: dict, kind: str, data: dict) -> None:
    if kind == "rfp_rv_again":
        return await start(eng, conv, st, data["project_id"])
    if not st.get("review"):
        st["review"] = {"project_id": data["project_id"], "workspace_id": None, "skipped": [], "reviewed": 0}
    if kind == "rfp_rv_accept":
        if await _submit(eng, conv, st, data, "accepted"):
            eng.say(conv, f"Accepted {data['code']}.")
        return await show_next(eng, conv, st)
    if kind == "rfp_rv_skip":
        st["review"].setdefault("skipped", []).append(data["rid"])
        return await show_next(eng, conv, st)
    if kind in ("rfp_rv_edit", "rfp_rv_write"):
        st["awaiting"] = {"type": "rfp_final_text", **data, "write": kind == "rfp_rv_write"}
        what = "the answer as it should go to the buyer" if kind == "rfp_rv_edit" else "the answer your expert gives"
        return eng.say(conv, f"Type {what} for {data['code']} in your next message (Shift+Enter for a new line). Say \"cancel\" to leave it as it is.")
    if kind == "rfp_rv_reject":
        return _ask_reason(eng, conv, st, data, "rejected", None, f"Why is {data['code']} wrong? Pick the closest reason.", optional=False)
    if kind == "rfp_rv_reason":
        tags = [data["tag"]] if data.get("tag") else []
        if await _submit(eng, conv, st, data, data["action"], data.get("text"), tags):
            if data["action"] == "rejected":
                st["review"].setdefault("skipped", []).append(data["rid"])
                eng.say(conv, f"Rejected {data['code']}. It still needs an answer before the response can be exported.",
                        actions=[eng.offer(st, "rfp_rv_write", "Write it now", data={k: data[k] for k in ("rid", "code", "project_id")})])
            else:
                eng.say(conv, f"Saved {data['code']} as final.")
        return await show_next(eng, conv, st)
    eng.say(conv, "I don't know how to do that any more.")
