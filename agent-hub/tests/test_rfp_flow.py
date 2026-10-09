"""The RFP flow end to end against the fake RFP Memory Assistant."""

from __future__ import annotations

import pytest

from .helpers import RFP_DOC, action, actions_of, cards, click, links, say, texts

pytestmark = pytest.mark.anyio


async def extracted(engine, conv, files=(RFP_DOC,), text=""):
    """Attach, confirm extraction, return the events up to the drafting confirmation."""
    ask = await say(engine, conv, text, list(files))
    return ask, await click(engine, conv, action(ask, "Yes")["id"])


async def test_happy_path_from_attachment_to_export_and_outcome(engine, fakes):
    conv = engine.new_conversation()

    # 1. preflight and the extraction confirmation: nothing has been sent or spent yet
    ask = await say(engine, conv, "", [RFP_DOC])
    text = texts(ask)
    assert "Acme Security" in text and "company facts are in place" in text and "1 model call" in text
    assert actions_of(ask)[0]["cost"] == "1 model call"
    assert fakes.rfp.spending_calls() == [] and fakes.rfp.model_calls == 0

    # 2. confirm: the project is created with the file name as its name, then polled until extracted
    extract = await click(engine, conv, action(ask, "Yes")["id"])
    assert fakes.rfp.create_bodies == [{"filename": RFP_DOC[0], "size": len(RFP_DOC[1]), "name": None, "client": None, "industry": None}]
    assert fakes.rfp.projects[1]["name"] == "Acme Security Questionnaire"
    assert any(e["kind"] == "progress" for e in extract)
    req_card = cards(extract)[0]
    assert req_card["title"] == "Requirements found in Acme Security Questionnaire.docx"
    rows = req_card["sections"][0]["rows"]
    assert rows == [["Requirements", "6"], ["Marked mandatory", "4"], ["Optional or unmarked", "2"]]
    bullets = req_card["sections"][1]["bullets"]
    assert len(bullets) == 5 and bullets[0].startswith("R-001 Describe your information security program")
    assert links(extract)[0]["url"] == "http://127.0.0.1:8001/#/projects/1"
    draft_ask = actions_of(extract)
    assert "6 model calls" in texts(extract) and draft_ask[0]["cost"] == "6 model calls"
    assert fakes.rfp.model_calls == 1  # drafting has not started

    # 3. confirm drafting
    drafted = await click(engine, conv, draft_ask[0]["id"])
    assert fakes.rfp.model_calls == 7
    assert [c[1] for c in fakes.rfp.spending_calls()] == ["/v1/projects", "/v1/projects/1/draft"]
    assert fakes.rfp.called("POST", r"/draft")[0][0] == "POST"
    card = cards(drafted)[0]
    assert card["tone"] == "ok" and card["title"] == "Drafts: 6 answers for 6 requirements"
    assert ["Drafted from company facts, ready to review", "6"] in card["sections"][0]["rows"]
    accept = action(drafted, "Accept all grounded answers")
    assert accept["cost"] == "0 model calls"

    # 4. accept all grounded: one review call per draft, then the export links
    done = await click(engine, conv, accept["id"])
    assert len(fakes.rfp.review_bodies) == 6 and all(b == {"action": "accepted"} for _, b in fakes.rfp.review_bodies)
    assert "Accepted 6 grounded answers" in texts(done) and fakes.rfp.model_calls == 7
    export = [c for c in cards(done) if c["title"] == "Your response is ready"][0]
    assert [link["label"] for link in export["links"]] == ["Download Word (.docx)"]  # not an .xlsx upload
    url = export["links"][0]["url"]
    assert url.startswith(f"/api/conversations/{conv}/downloads/")

    # 5. the download proxy hands over the finished file with its Content-Disposition
    content, media, disposition = await engine.download(conv, url.rsplit("/", 1)[1])
    assert content == b"FAKE-DOCX-FILE" and media == "application/fake-docx"
    assert disposition == 'attachment; filename="Acme Security Questionnaire - response.docx"'

    # 6. outcome capture, confirmed first, sent with the free-text reason
    confirm = await say(engine, conv, "We won this RFP because our security answers were the most detailed")
    assert "WON" in texts(confirm) and "0 model calls" in texts(confirm) and fakes.rfp.outcome_bodies == []
    recorded = await click(engine, conv, action(confirm, "Yes")["id"])
    assert fakes.rfp.outcome_bodies == [
        {"project_id": 1, "result": "won", "loss_reason": "our security answers were the most detailed"}]
    assert "Outcome recorded" in cards(recorded)[0]["title"]
    assert fakes.rfp.model_calls == 7 and fakes.all_forbidden() == []


async def test_an_excel_rfp_also_offers_the_xlsx_export(engine, fakes):
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv, [("Vendor Questionnaire.xlsx", b"xlsx bytes")])
    drafted = await click(engine, conv, actions_of(extract)[0]["id"])
    done = await click(engine, conv, action(drafted, "Accept all")["id"])
    export = [c for c in cards(done) if c["title"] == "Your response is ready"][0]
    assert [link["label"] for link in export["links"]] == ["Download Word (.docx)", "Download Excel (.xlsx)"]
    token = export["links"][1]["url"].rsplit("/", 1)[1]
    content, _, disposition = await engine.download(conv, token)
    assert content == b"FAKE-XLSX-FILE" and "response.xlsx" in disposition


async def test_project_name_client_and_industry_are_used_only_when_the_user_gave_them(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Answer this RFP, called Q3 Bid, client Northwind, industry retail", [RFP_DOC])
    assert '"Q3 Bid"' in texts(ask) and "Northwind" in texts(ask)
    await click(engine, conv, action(ask, "Yes")["id"])
    assert fakes.rfp.create_bodies[0]["name"] == "Q3 Bid" and fakes.rfp.create_bodies[0]["client"] == "Northwind"
    assert fakes.rfp.create_bodies[0]["industry"] == "retail"


async def test_a_workspace_without_company_facts_stops_before_any_spend(engine, fakes):
    fakes.rfp.company_set_up = False
    conv = engine.new_conversation()
    stop = await say(engine, conv, "", [RFP_DOC])
    assert "Add your company facts in the RFP Memory Assistant first" in texts(stop)
    assert "Acme Security" in texts(stop) and not actions_of(stop)
    assert fakes.rfp.spending_calls() == [] and fakes.rfp.projects == {}
    # facts get added in the app; "start" again works with the file that was kept
    fakes.rfp.company_set_up = True
    again = await say(engine, conv, "Draft the RFP")
    assert "Ready to set up" in texts(again) and actions_of(again)


async def test_facts_removed_between_confirmation_and_extraction_still_stop(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "", [RFP_DOC])
    fakes.rfp.company_set_up = False
    stop = await click(engine, conv, action(ask, "Yes")["id"])
    assert "Add your company facts" in texts(stop) and fakes.rfp.spending_calls() == []


async def test_a_late_parse_failure_shows_the_project_error_plainly(engine, fakes):
    fakes.rfp.extract_mode = "late_parse_fail"
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv)
    error = next(e for e in extract if e["kind"] == "error")
    assert "The PDF has no text layer (it looks scanned). Upload a text-based copy." in error["text"]
    assert "Nothing was drafted" in texts(extract)
    assert not [a for a in actions_of(extract) if "draft" in a["label"].lower()]
    assert fakes.rfp.model_calls == 1  # only the extraction was spent; drafting was never offered


async def test_a_failed_extraction_job_uses_the_error_info(engine, fakes):
    fakes.rfp.extract_mode = "job_fail"
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv)
    error = next(e for e in extract if e["kind"] == "error")
    assert "The model quota is used up" in error["text"] and "What to do: Wait for the quota to reset" in error["text"]


async def test_partial_drafting_is_reported_honestly_and_accept_all_only_takes_the_grounded(engine, fakes):
    fakes.rfp.draft_mode = "partial"
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv)
    drafted = await click(engine, conv, actions_of(extract)[0]["id"])
    text = texts(drafted)
    assert "doesn't mean every answer worked" in text
    card = cards(drafted)[0]
    assert card["tone"] == "warn"
    rows = dict(map(tuple, card["sections"][0]["rows"]))
    assert rows == {
        "Requirements": "6", "Drafted from company facts, ready to review": "2", "Flagged for a closer look": "1",
        "Need a subject-matter expert": "1", "Failed": "1", "Not drafted": "1",
    }
    reported = [s for s in card["sections"] if s.get("heading") == "What the RFP assistant reported"][0]
    assert "Stopped early" in reported["text"]
    why = [s for s in card["sections"] if s.get("heading") == "Why some failed"][0]
    assert "R-005: Model returned an empty answer." in why["bullets"]
    assert "Accepting all only takes the grounded ones" in text and action(drafted, "Review them one by one")

    done = await click(engine, conv, action(drafted, "Accept all grounded")["id"])
    assert [rid for rid, _ in fakes.rfp.review_bodies] == [500, 501]  # only the two grounded ones
    assert "Accepted 2 grounded answers" in texts(done)
    left = [s for c in cards(done) for s in c.get("sections", []) if s.get("heading") == "Still needs you"][0]
    assert left["bullets"] == ["1 flagged", "1 need an SME", "1 failed", "1 not drafted"]
    assert fakes.rfp.projects[1]["state"] == "in_review"
    assert not [c for c in cards(done) if c["title"] == "Your response is ready"]
    assert any(
        link["label"] == "Open in RFP Memory Assistant to complete the SME answers" and link["url"].endswith("#/projects/1")
        for link in links(done)
    )
    export = await say(engine, conv, "export it as Word")
    assert "can't be exported yet: 4 of 6 answers aren't final" in texts(export)
    assert fakes.rfp.called("GET", "export") == []  # the hub never asks the app to export an unfinished response
    again = await click(engine, conv, action(drafted, "Accept all grounded")["id"])
    assert "already done" in texts(again) and len(fakes.rfp.review_bodies) == 2


async def test_a_failed_draft_job_is_reported_and_never_retried(engine, fakes):
    fakes.rfp.draft_mode = "job_fail"
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv)
    drafted = await click(engine, conv, actions_of(extract)[0]["id"])
    error = next(e for e in drafted if e["kind"] == "error")
    assert "The API key was rejected" in error["text"]
    assert fakes.rfp.model_calls == 1 + 6  # one extraction, one drafting attempt, no automatic redraft
    assert len(fakes.rfp.called("POST", r"/draft")) == 1
    assert not action_labels(drafted, "Accept")


def action_labels(events, part):
    return [a for a in actions_of(events) if part.lower() in a["label"].lower()]


async def test_no_company_facts_during_drafting_is_a_clear_409_and_nothing_else(engine, fakes):
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv)
    fakes.rfp.company_set_up = False
    stop = await click(engine, conv, actions_of(extract)[0]["id"])
    assert "Add your company facts" in texts(stop)
    assert fakes.rfp.called("POST", r"/draft") == [] and fakes.rfp.model_calls == 1


async def test_drafting_a_project_that_already_has_drafts_is_refused(engine, fakes):
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv)
    fakes.rfp.projects[1]["state"] = "in_review"  # someone drafted it in the app meanwhile
    refused = await click(engine, conv, actions_of(extract)[0]["id"])
    assert "would replace the existing drafts" in texts(refused)
    assert fakes.rfp.called("POST", r"/draft") == []


async def test_a_changed_requirement_count_asks_again_with_the_new_cost(engine, fakes):
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv)
    fakes.rfp.projects[1]["requirements"].pop()
    ask = await click(engine, conv, actions_of(extract)[0]["id"])
    assert "5 model calls" in texts(ask) and fakes.rfp.called("POST", r"/draft") == []


async def test_the_workspace_changing_before_drafting_stops_everything(engine, fakes):
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv)
    fakes.rfp.workspace = {"id": "ws-other", "name": "Other Co", "kind": "company"}
    stop = await click(engine, conv, actions_of(extract)[0]["id"])
    assert "workspace changed" in texts(stop) and fakes.rfp.called("POST", r"/draft") == []


async def test_a_loss_asks_for_a_reason_and_the_reason_is_sent_as_given(engine, fakes):
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv)
    ask = await say(engine, conv, "We lost this RFP")
    assert "Why was Acme Security Questionnaire lost?" in texts(ask) and action(ask, "without a reason")
    confirm = await say(engine, conv, "the incumbent was cheaper")
    assert "LOST because: the incumbent was cheaper" in texts(confirm)
    await click(engine, conv, action(confirm, "Yes")["id"])
    assert fakes.rfp.outcome_bodies == [{"project_id": 1, "result": "lost", "loss_reason": "the incumbent was cheaper"}]
    again = await say(engine, conv, "We lost this RFP because of price")
    assert "already recorded as lost" in texts(again) and len(fakes.rfp.outcome_bodies) == 1


async def test_a_loss_can_be_recorded_without_a_reason(engine, fakes):
    conv = engine.new_conversation()
    await extracted(engine, conv)
    ask = await say(engine, conv, "We lost this RFP")
    confirm = await click(engine, conv, action(ask, "without a reason")["id"])
    assert "LOST" in texts(confirm) and fakes.rfp.outcome_bodies == []
    await click(engine, conv, action(confirm, "Yes")["id"])
    assert fakes.rfp.outcome_bodies == [{"project_id": 1, "result": "lost"}]


async def test_an_outcome_without_a_project_in_the_chat_explains(engine, fakes):
    conv = engine.new_conversation()
    reply = await say(engine, conv, "We won this RFP")
    assert "don't have an RFP project in this chat" in texts(reply) and fakes.rfp.outcome_bodies == []


async def test_a_file_the_rfp_app_cannot_read_is_declined_before_any_call(engine, fakes):
    conv = engine.new_conversation()
    reply = await say(engine, conv, "Answer this RFP", [("emails.eml", b"From: a")])
    assert "reads .pdf, .docx, .xlsx, .txt, .md" in texts(reply)
    assert fakes.rfp.calls == []


async def test_asking_for_an_rfp_without_a_file_waits_for_one(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "I need to answer an RFP")
    assert "Attach the RFP" in texts(ask)
    ready = await say(engine, conv, "", [RFP_DOC])
    assert "Ready to set up" in texts(ready)


async def test_the_chat_never_touches_workspaces_models_or_requirements(engine, fakes):
    conv = engine.new_conversation()
    _, extract = await extracted(engine, conv)
    drafted = await click(engine, conv, actions_of(extract)[0]["id"])
    await click(engine, conv, action(drafted, "Accept all")["id"])
    await say(engine, conv, "export it as Word")
    await say(engine, conv, "We won this RFP")
    assert fakes.all_forbidden() == []
    assert all(c[0] == "GET" or c[1] in ("/v1/projects", "/v1/projects/1/draft", "/v1/projects/1/outcome") or "/review" in c[1]
               for c in fakes.rfp.calls)
