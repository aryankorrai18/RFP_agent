"""RFP flows: attach a document, extract its requirements, draft the answers, accept the grounded ones,
hand over the export, and record how the bid ended.

What the chat never does here: edit requirements, write SME answers, redraft (a new /draft replaces
every draft and un-finalises reviews), or touch workspaces and models. Those stay in the RFP app."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..clients import AgentDown, UpstreamError, WorkspaceInfo, describe_error_info
from ..engine import TERMINAL_JOB, cost_line, plural
from ..intents import Intent, free_reason
from ..store import Upload

if TYPE_CHECKING:
    from ..engine import Engine

RFP_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".txt", ".md")  # what the RFP app can parse
JOB_FAILED = {"failed", "cancelled", "interrupted"}
SAMPLE_QUESTIONS = 5
RESULT_WORDS = {"won": "WON", "lost": "LOST", "no_decision": "NO DECISION"}


def _short(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _ws_text(ws: WorkspaceInfo | None) -> str:
    return f" (workspace: {ws.name or ws.id})" if ws and (ws.name or ws.id) else ""


def _project_link(eng: Engine, pid: int, label: str = "Open in RFP Memory Assistant", text: str | None = None) -> dict:
    return eng.link_card("rfp", label, f"/{pid}", text=text)


def _no_facts(eng: Engine, conv: str, ws: WorkspaceInfo) -> None:
    eng.say(
        conv,
        f"Your active RFP workspace{_ws_text(ws)} has no company facts yet, so there is nothing to ground the answers in. "
        "Add your company facts in the RFP Memory Assistant first, then ask me again. I've kept your file.",
        cards=[eng.link_card("rfp", "Open the RFP Memory Assistant")],
    )


async def _same_workspace(eng: Engine, conv: str, expected_id: str | None, *, need_facts: bool = False) -> WorkspaceInfo | None:
    """Re-check before spending or changing anything: the active workspace is global and project ids are per workspace."""
    now = await eng.rfp.workspace_info()
    if expected_id is not None and now.id != expected_id:
        eng.say(conv, f"The active RFP workspace changed to \"{now.name or now.id}\" since I asked, so I stopped without "
                      "doing anything. Ask me again and I'll start from the current one.")
        return None
    if need_facts and not now.company_set_up:
        _no_facts(eng, conv, now)
        return None
    return now


def _project_state(st: dict) -> dict:
    return st.get("project") or {}


def _is_grounded(req: dict) -> bool:
    """Drafted from facts, no flags, nobody has finalised it yet: safe to accept in bulk."""
    d = req.get("draft") or {}
    return bool(d) and d.get("status") == "drafted" and not d.get("flags") and not req.get("final") and bool((d.get("answer") or "").strip())


def _breakdown(project: dict) -> dict[str, int]:
    """Disjoint buckets that add up to the requirement count."""
    reqs = project.get("requirements") or []
    out = {"total": len(reqs), "grounded": 0, "flagged": 0, "needs_sme": 0, "failed": 0, "not_drafted": 0, "final": 0}
    for r in reqs:
        d = r.get("draft")
        out["final"] += bool(r.get("final"))
        if not d:
            out["not_drafted"] += 1
        elif d.get("status") == "failed":
            out["failed"] += 1
        elif d.get("status") == "needs_sme":
            out["needs_sme"] += 1
        elif d.get("flags"):
            out["flagged"] += 1
        else:
            out["grounded"] += 1
    return out


# -- start: attach, preflight, confirm extraction ------------------------------------------------------


async def rfp_start(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    everything = eng.pending_uploads(st)
    usable = [u for u in everything if u.ext in RFP_EXTENSIONS]
    if not usable:
        if everything:
            return eng.say(conv, f"The RFP assistant reads {', '.join(RFP_EXTENSIONS)} files; the ones I have are "
                                 f"{', '.join(u.filename for u in everything)}. Attach the RFP in one of those formats.")
        st["awaiting"] = {"type": "rfp_file"}
        return eng.say(conv, "Attach the RFP or questionnaire (.docx, .xlsx, .pdf, .txt or .md) and I'll set it up, "
                             "asking before anything uses model calls.")
    upload = usable[-1]
    ws = await eng.rfp.workspace_info()
    st["awaiting"] = None
    if not ws.company_set_up:
        return _no_facts(eng, conv, ws)
    slots = {k: v for k, v in (intent.slots or {}).items() if k in ("name", "client", "industry") and v}
    name = slots.get("name") or Path(upload.filename).stem
    text = (
        f"Ready to set up {upload.filename} in the RFP Memory Assistant{_ws_text(ws)}; your company facts are in place. "
        f"I'll create a project called \"{name}\"" + (f" for {slots['client']}" if slots.get("client") else "")
        + " and read out its questions."
    )
    if len(usable) > 1:
        text += f" (I'm using the latest file; {plural(len(usable) - 1, 'other attachment')} will wait for a separate request.)"
    eng.confirmation(
        conv, st, text, "rfp_extract",
        {"upload_id": upload.id, "slots": slots, "workspace_id": ws.id, "workspace_name": ws.name},
        calls=1, cost_note=cost_line(1, "to read the document and pull out its questions"), label="Yes, read it",
    )


async def _start_extract(eng: Engine, conv: str, st: dict, data: dict) -> None:
    ws = await _same_workspace(eng, conv, data.get("workspace_id"), need_facts=True)
    if ws is None:
        return
    if eng.store.get_upload(data["upload_id"]) is None:
        return eng.say(conv, "I can't find that file any more. Attach it again and I'll start over.")
    eng.approve(st, 1)
    st["rfp_workspace"] = {"id": ws.id, "name": ws.name}
    st["run"] = {
        "flow": "rfp", "step": "extract_start", "upload_id": data["upload_id"], "slots": data.get("slots") or {},
        "workspace_id": ws.id, "project_id": None, "job_id": None, "track": None,
    }
    eng.spawn_run(conv, run)


# -- the confirmed run ---------------------------------------------------------------------------------


async def run(eng: Engine, conv: str) -> None:
    while await _advance(eng, conv):
        pass


async def _advance(eng: Engine, conv: str) -> bool:
    """One step of the confirmed run (extract, then draft). False when it is finished."""
    async with eng.lock(conv):
        st = eng.load(conv)
        state = st.get("run")
        if not state or state.get("flow") != "rfp":
            return False
        step = state["step"]
        if step.endswith("_start"):
            try:
                return await _start_step(eng, conv, st, state, step)
            finally:
                eng.save(conv, st)
        pid, track = state["project_id"], state["track"]
    # Poll without holding the conversation lock, so the person can keep chatting.
    if step == "extract":
        view = await eng.poll(
            conv, app="rfp", step="extract", job_id=track, fetch=lambda: eng.rfp.get_project(pid),
            done=lambda p: p.get("state") != "extracting" or (p.get("job") or {}).get("status") in JOB_FAILED,
            progress=lambda p: (0, 0), label="Reading the RFP", final_status=lambda p: str(p.get("state")),
        )
    else:
        job_id, total = state["job_id"], state["calls"]
        view = await eng.poll(
            conv, app="rfp", step="draft", job_id=track, fetch=lambda: eng.rfp.get_job(job_id),
            done=lambda j: j.get("status") in TERMINAL_JOB,
            progress=lambda j: (j.get("done") or 0, j.get("total") or total), label="Drafting answers",
            final_status=lambda j: str(j.get("status", "completed")),
        )
    async with eng.lock(conv):
        st = eng.load(conv)
        state = st.get("run")
        if not state or state.get("flow") != "rfp":
            return False
        try:
            if step == "extract":
                return await _finish_extract(eng, conv, st, state, view)
            return await _finish_draft(eng, conv, st, state, view)
        finally:
            eng.save(conv, st)


async def _start_step(eng: Engine, conv: str, st: dict, state: dict, step: str) -> bool:
    if state.get("job_id") is None and state.get("resumed"):
        st["run"] = None
        eng.say(conv, "I was interrupted just before starting that step, and I can't tell whether it began. To be safe I "
                      "stopped without spending anything more. Open the project in the RFP Memory Assistant to check, "
                      "or ask me again.", cards=[_project_link(eng, state["project_id"])] if state.get("project_id") else None)
        return False
    calls = 1 if step == "extract_start" else state["calls"]
    if not eng.spend(st, calls):
        st["run"] = None
        eng.say(conv, "I stopped before the next model call because it would go past the number you approved.")
        return False
    state["resumed"] = False
    eng.save(conv, st)  # the step is on disk before the call, so a restart never repeats a spend
    if step == "extract_start":
        upload = eng.store.get_upload(state["upload_id"])
        slots = state.get("slots") or {}
        project = await eng.rfp.create_project(
            upload.filename, upload.read(), name=slots.get("name"), client=slots.get("client"), industry=slots.get("industry"),
        )
        pid, job_id = project["id"], (project.get("job") or {}).get("id")
        track = job_id if job_id is not None else f"project-{pid}"
        eng.store.add_job(conv, "rfp", track, "extract")
        st["uploads"] = [u for u in st["uploads"] if u != state["upload_id"]]
        st["project_id"], st["flow"] = pid, "rfp"
        st["project"] = {
            "id": pid, "name": project.get("name") or Path(upload.filename).stem, "filename": upload.filename,
            "file_kind": upload.ext.lstrip("."), "workspace_id": state["workspace_id"],
        }
        state.update(step="extract", project_id=pid, job_id=job_id, track=track)
        return True
    pid = state["project_id"]
    job = await eng.rfp.start_draft(pid)
    eng.store.add_job(conv, "rfp", job["id"], "draft")
    state.update(step="draft", job_id=job["id"], track=job["id"])
    return True


# -- requirements card, then the drafting confirmation ---------------------------------------------------


def _requirements_card(project: dict) -> dict:
    reqs = project.get("requirements") or []
    mandatory = sum(1 for r in reqs if r.get("mandatory") is True)
    rows = [["Requirements", str(len(reqs))], ["Marked mandatory", str(mandatory)], ["Optional or unmarked", str(len(reqs) - mandatory)]]
    sections: list[dict] = [{"rows": rows}, {
        "heading": f"First {min(SAMPLE_QUESTIONS, len(reqs))} questions",
        "bullets": [f"{r.get('code', '')} {_short(r.get('question', ''))}".strip() for r in reqs[:SAMPLE_QUESTIONS]],
    }]
    return {"title": f"Requirements found in {project.get('filename', 'the document')}", "subtitle": project.get("name"), "tone": "ok", "sections": sections}


async def _finish_extract(eng: Engine, conv: str, st: dict, state: dict, project: dict) -> bool:
    st["run"] = None
    pid = state["project_id"]
    if project.get("state") != "requirements_extracted":
        job = project.get("job") or {}
        detail = describe_error_info(
            project.get("error_info") or job.get("error_info"),
            project.get("error") or job.get("error") or "The document could not be read.", eng.rfp.name,
        )
        eng.error(conv, f"The RFP assistant couldn't read {project.get('filename', 'the document')}. {detail}")
        eng.say(conv, "Nothing was drafted. You can fix the file and attach it again, or open the project to see its state.",
                cards=[_project_link(eng, pid)])
        return False
    n = len(project.get("requirements") or [])
    if n == 0:
        eng.say(conv, "The document was read, but I found no questions in it. Open the project to check what was extracted.",
                cards=[_project_link(eng, pid)])
        return False
    eng.say(conv, f"Done reading. I found {plural(n, 'requirement')}.", cards=[_requirements_card(project), _project_link(eng, pid)])
    _offer_draft(eng, conv, st, pid, n)
    return False


def _offer_draft(eng: Engine, conv: str, st: dict, pid: int, n: int) -> None:
    ws = st.get("rfp_workspace") or {}
    eng.confirmation(
        conv, st,
        f"Want me to draft answers for all {plural(n, 'requirement')}? I'll only draft; I won't edit the requirements, the "
        "company facts, or anything you've already approved. It can take a few minutes.",
        "rfp_draft", {"project_id": pid, "requirements": n, "workspace_id": _project_state(st).get("workspace_id") or ws.get("id")},
        calls=n, cost_note=cost_line(n, "one per requirement"), label=f"Yes, draft {plural(n, 'answer')}",
    )


async def _start_draft(eng: Engine, conv: str, st: dict, data: dict) -> None:
    ws = await _same_workspace(eng, conv, data.get("workspace_id"), need_facts=True)
    if ws is None:
        return
    pid = data["project_id"]
    project = await eng.rfp.get_project(pid)
    if project.get("state") != "requirements_extracted":
        return eng.say(
            conv,
            "That project has moved on since I asked (it has already been drafted or changed), and drafting again would replace "
            "the existing drafts and un-approve reviews. I won't do that from here; use the RFP Memory Assistant if you really "
            "want to redraft.", cards=[_project_link(eng, pid)],
        )
    n = len(project.get("requirements") or [])
    if n != data["requirements"]:
        eng.say(conv, f"The project now has {plural(n, 'requirement')} instead of {data['requirements']}, so I'm asking again with the new cost.")
        return _offer_draft(eng, conv, st, pid, n)
    eng.approve(st, n)
    st["run"] = {"flow": "rfp", "step": "draft_start", "project_id": pid, "calls": n, "job_id": None, "track": None,
                 "workspace_id": data.get("workspace_id")}
    eng.spawn_run(conv, run)


# -- drafts card, accept all -------------------------------------------------------------------------------


def _drafts_card(project: dict, job: dict, buckets: dict[str, int]) -> dict:
    n = buckets["total"]
    drafted = n - buckets["not_drafted"] - buckets["failed"]
    rows = [
        ["Requirements", str(n)],
        ["Drafted from company facts, ready to review", str(buckets["grounded"])],
        ["Flagged for a closer look", str(buckets["flagged"])],
        ["Need a subject-matter expert", str(buckets["needs_sme"])],
        ["Failed", str(buckets["failed"])],
        ["Not drafted", str(buckets["not_drafted"])],
    ]
    sections: list[dict] = [{"rows": rows}]
    if job.get("warning"):
        sections.append({"heading": "What the RFP assistant reported", "text": str(job["warning"])})
    failed = [r for r in project.get("requirements") or [] if (r.get("draft") or {}).get("status") == "failed"]
    if failed:
        bullets = [f"{r.get('code', '')}: {_short((r['draft'].get('error') or 'no answer was produced'), 140)}" for r in failed[:3]]
        if len(failed) > 3:
            bullets.append(f"and {len(failed) - 3} more")
        sections.append({"heading": "Why some failed", "bullets": bullets})
    ok = buckets["grounded"] == n
    return {
        "title": f"Drafts: {plural(drafted, 'answer')} for {plural(n, 'requirement')}",
        "subtitle": project.get("name"), "tone": "ok" if ok else "warn", "sections": sections,
    }


async def _finish_draft(eng: Engine, conv: str, st: dict, state: dict, job: dict) -> bool:
    st["run"] = None
    pid = state["project_id"]
    project = await eng.rfp.get_project(pid)
    buckets = _breakdown(project)
    status = job.get("status")
    if status == "failed" or (status in JOB_FAILED and not buckets["grounded"] + buckets["flagged"] + buckets["needs_sme"]):
        eng.error(conv, "Drafting didn't finish. " + describe_error_info(job.get("error_info"), job.get("error") or f"The job {status}.", eng.rfp.name))
    done_all = status == "completed" and not buckets["failed"] and not buckets["not_drafted"]
    if status == "completed":
        lead = ("The drafting job finished and every requirement has a draft." if done_all else
                "The drafting job finished, but that doesn't mean every answer worked. Here is what actually happened.")
    else:
        lead = f"Drafting {status or 'stopped'} part-way. Here is what was drafted before that."
    actions = []
    if buckets["grounded"]:
        actions.append(eng.offer(st, "rfp_accept", "Accept all grounded answers", style="primary", cost="0 model calls",
                                 data={"project_id": pid, "workspace_id": _project_state(st).get("workspace_id")}))
    reviewable = buckets["grounded"] + buckets["flagged"] + buckets["needs_sme"] + buckets["failed"]
    if reviewable:
        actions.append(eng.offer(st, "rfp_rv_again", "Review them one by one", style="ghost", cost="0 model calls",
                                 data={"project_id": pid}))
    notes = []
    if buckets["flagged"] or buckets["needs_sme"] or buckets["failed"]:
        notes.append("Accepting all only takes the grounded ones. Review one by one to check the flagged answers, write the "
                     "ones that need an expert, and say why when you change something: that is what the assistant learns from.")
    if buckets["not_drafted"]:
        notes.append("Answers that were never drafted need drafting again before they can be reviewed.")
    eng.say(conv, " ".join([lead, *notes]), cards=[_drafts_card(project, job, buckets), _project_link(eng, pid)], actions=actions)
    return False


async def _accept_all(eng: Engine, conv: str, st: dict, pid: int, workspace_id: str | None) -> None:
    if await _same_workspace(eng, conv, workspace_id) is None:
        return
    project = await eng.rfp.get_project(pid)
    eligible = [r for r in project.get("requirements") or [] if _is_grounded(r)]
    if not eligible:
        buckets = _breakdown(project)
        return eng.say(
            conv,
            "There are no grounded answers waiting to be accepted" + (f" ({buckets['final']} already final)" if buckets["final"] else "") + ". "
            "Flagged, SME and failed drafts need a person.", cards=[_project_link(eng, pid)],
        )
    accepted = skipped = 0
    try:
        for req in eligible:
            try:
                await eng.rfp.review(req["id"], "accepted")
                accepted += 1
            except UpstreamError:
                skipped += 1
    except AgentDown:
        eng.say(conv, f"The RFP assistant stopped answering after I had accepted {accepted} of {len(eligible)}. The rest are untouched.")
        raise
    project = await eng.rfp.get_project(pid)
    buckets = _breakdown(project)
    rows = [["Accepted now", str(accepted)]] + ([["Could not accept", str(skipped)]] if skipped else []) + [
        ["Final answers", f"{buckets['final']} of {buckets['total']}"],
    ]
    left = [(buckets[k], label) for k, label in (("flagged", "flagged"), ("needs_sme", "need an SME"), ("failed", "failed"), ("not_drafted", "not drafted")) if buckets[k]]
    sections: list[dict] = [{"rows": rows}]
    if left:
        sections.append({"heading": "Still needs you", "bullets": [f"{n} {label}" for n, label in left]})
    eng.say(
        conv, f"Accepted {plural(accepted, 'grounded answer')}. This used no model calls.",
        cards=[{"title": "Grounded answers accepted", "subtitle": project.get("name"), "tone": "ok" if not skipped else "warn", "sections": sections}],
    )
    await _after_review(eng, conv, st, project)


async def _after_review(eng: Engine, conv: str, st: dict, project: dict) -> None:
    pid = project["id"]
    if project.get("state") in ("approved", "exported"):
        return _offer_exports(eng, conv, st, project)
    eng.say(
        conv, "The response isn't complete yet, so it can't be exported. Review the rest here: check the flagged answers and "
              "write the ones that need an expert.",
        cards=[_project_link(eng, pid, "Open in RFP Memory Assistant to complete the SME answers")],
        actions=[eng.offer(st, "rfp_rv_again", "Review the rest here", style="primary", cost="0 model calls", data={"project_id": pid})],
    )


def _offer_exports(eng: Engine, conv: str, st: dict, project: dict, fmt: str | None = None) -> None:
    pid = project["id"]
    is_xlsx = str(project.get("file_kind") or "").lower() == "xlsx" or str(project.get("filename", "")).lower().endswith(".xlsx")
    formats = [("docx", "Download Word (.docx)")] + ([("xlsx", "Download Excel (.xlsx)")] if is_xlsx else [])
    if fmt:
        formats = [f for f in formats if f[0] == fmt]
        if not formats:
            return eng.say(conv, "That RFP wasn't an Excel file, so only a Word export is available.", cards=[_project_link(eng, pid)])
    links = [{"label": label, "url": f"/api/conversations/{conv}/downloads/{eng.new_download(st, pid, f)}"} for f, label in formats]
    eng.say(
        conv, "Every answer is final, so the response can be exported.",
        cards=[{"title": "Your response is ready", "subtitle": project.get("name"), "tone": "ok", "links": links,
                "sections": [{"text": "Exporting marks the project as exported in the RFP Memory Assistant."}]}],
    )


async def rfp_review(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    from . import review

    await review.start(eng, conv, st)


async def rfp_accept_all(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    pid = st.get("project_id")
    if not pid:
        return eng.say(conv, "I don't have an RFP project in this chat yet. Attach the RFP and I'll set one up.")
    await _accept_all(eng, conv, st, pid, _project_state(st).get("workspace_id"))


async def rfp_export(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    pid = st.get("project_id")
    if not pid:
        return eng.say(conv, "I don't have an RFP project in this chat yet. Attach the RFP and I'll set one up.")
    if await _same_workspace(eng, conv, _project_state(st).get("workspace_id")) is None:
        return
    project = await eng.rfp.get_project(pid)
    if project.get("state") in ("approved", "exported"):
        return _offer_exports(eng, conv, st, project, intent.fmt)
    buckets = _breakdown(project)
    open_n = buckets["total"] - buckets["final"]
    actions = []
    if buckets["grounded"]:
        actions.append(eng.offer(st, "rfp_accept", "Accept all grounded answers", style="primary", cost="0 model calls",
                                 data={"project_id": pid, "workspace_id": _project_state(st).get("workspace_id")}))
    eng.say(
        conv, f"It can't be exported yet: {open_n} of {buckets['total']} answers aren't final. The app only exports a response "
              "when every answer is accepted, edited or rewritten (SME answers need a person).",
        cards=[_project_link(eng, pid, "Open in RFP Memory Assistant to complete the SME answers")], actions=actions,
    )


# -- outcome ------------------------------------------------------------------------------------------------


async def outcome(eng: Engine, conv: str, st: dict, intent: Intent) -> None:
    pid = st.get("project_id")
    if not pid:
        return eng.say(
            conv, "I don't have an RFP project in this chat, so I don't know which response you mean. Record it in the "
                  "RFP Memory Assistant, or attach the RFP here first.", cards=[eng.link_card("rfp", "Open the RFP Memory Assistant")],
        )
    if await _same_workspace(eng, conv, _project_state(st).get("workspace_id")) is None:
        return
    project = await eng.rfp.get_project(pid)
    if project.get("outcome"):
        o = project["outcome"]
        why = f" ({o['loss_reason']})" if o.get("loss_reason") else ""
        return eng.say(conv, f"{project.get('name')} is already recorded as {o.get('result')}{why}. I won't overwrite a recorded "
                             "outcome from here; change it in the RFP Memory Assistant if it needs correcting.", cards=[_project_link(eng, pid)])
    result = intent.result or "lost"
    reason = (intent.slots or {}).get("free_reason")
    if result == "lost" and not reason:
        st["awaiting"] = {"type": "rfp_reason", "project_id": pid, "result": result}
        no_reason = eng.offer(st, "rfp_outcome_noreason", "Record without a reason", style="ghost",
                              data={"project_id": pid, "result": result})
        return eng.say(conv, f"Why was {project.get('name')} lost? Tell me in your own words (for example \"the price was higher "
                             "than the incumbent's\"), or record it without a reason.", actions=[no_reason])
    _offer_outcome(eng, conv, st, project, result, reason)


def _offer_outcome(eng: Engine, conv: str, st: dict, project: dict, result: str, reason: str | None) -> None:
    text = (
        f"Ready to record {project.get('name')} as {RESULT_WORDS.get(result, result.upper())}"
        + (f" because: {reason}" if reason else "") + ". This uses 0 model calls, but it changes memory: "
        "the answers used in this response get credit or blame, and the reason is kept for future drafts."
    )
    eng.confirmation(
        conv, st, text, "rfp_outcome",
        {"project_id": project["id"], "result": result, "reason": reason, "label": project.get("name"),
         "workspace_id": _project_state(st).get("workspace_id")},
        calls=0, label="Yes, record it", cost_label="0 model calls, updates memory",
    )


async def _record_outcome(eng: Engine, conv: str, st: dict, data: dict) -> None:
    if await _same_workspace(eng, conv, data.get("workspace_id")) is None:
        return
    row = await eng.rfp.record_outcome(data["project_id"], data["result"], data.get("reason"))
    rows = [["Result", str(row.get("result", data["result"]))]]
    if row.get("loss_reason") or data.get("reason"):
        rows.append(["Reason", str(row.get("loss_reason") or data["reason"])])
    eng.say(
        conv, f"Done. {data['label']} is now recorded as {data['result'].replace('_', ' ')}. This used no model calls.",
        cards=[{"title": f"Outcome recorded: {data['label']}", "tone": "ok", "sections": [{"rows": rows}]}, _project_link(eng, data["project_id"])],
    )


async def handle_slot(eng: Engine, conv: str, st: dict, intent: Intent, accepted: list[Upload]) -> None:
    awaiting = st.get("awaiting") or {}
    if awaiting.get("type") != "rfp_reason":
        st["awaiting"] = None
        return eng.say(conv, "Okay. Tell me what you'd like to do.")
    st["awaiting"] = None
    project = await eng.rfp.get_project(awaiting["project_id"])
    reason = free_reason(intent.text) or intent.text.strip(" .!")
    _offer_outcome(eng, conv, st, project, awaiting["result"], reason or None)


# -- button clicks ------------------------------------------------------------------------------------------


async def on_action(eng: Engine, conv: str, st: dict, kind: str, data: dict) -> None:
    if kind.startswith("rfp_rv_"):
        from . import review

        return await review.on_action(eng, conv, st, kind, data)
    if kind == "rfp_extract":
        return await _start_extract(eng, conv, st, data)
    if kind == "rfp_draft":
        return await _start_draft(eng, conv, st, data)
    if kind == "rfp_accept":
        return await _accept_all(eng, conv, st, data["project_id"], data.get("workspace_id"))
    if kind == "rfp_outcome":
        return await _record_outcome(eng, conv, st, data)
    if kind == "rfp_outcome_noreason":
        st["awaiting"] = None
        project = await eng.rfp.get_project(data["project_id"])
        return _offer_outcome(eng, conv, st, project, data["result"], None)
    eng.say(conv, "I don't know how to do that any more.")

