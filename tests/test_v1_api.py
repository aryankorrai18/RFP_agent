"""HTTP integration tests with the real application lifespan and offline fakes."""

from __future__ import annotations

import io
import json
import time

import docx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.llm import LLMError
from tests.conftest import docx_bytes
from tests.v1_fakes import FakeMemory, FakeV1LLM, make_context, pair, req

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture
def v1_client(tmp_path):
    llm = FakeV1LLM(
        pairs=[
            pair("Which SSO standards do you support?", "We support SAML 2.0 and OpenID Connect.", reference="3.1"),
            pair("Describe automatic user provisioning.", "We support SCIM 2.0 provisioning with Okta.", reference="3.2"),
        ],
        requirements=[
            req("Does your solution support SAML single sign-on?", section="Security", reference="3.1"),
            req("Is automatic user provisioning via SCIM supported?", section="Security", reference="3.2"),
        ],
    )
    memory = FakeMemory()
    ctx = make_context(tmp_path, llm, memory)
    app.state.v1_factory = lambda: ctx
    with TestClient(app) as client:
        yield client, ctx, llm, memory
    app.state.v1_factory = None


def wait_job(client: TestClient, job_id: int, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def upload(name="past.docx", data=None):
    return {"file": (name, data or docx_bytes("3.1 Q", "Response: A"), DOCX)}


def test_pages_are_served(v1_client):
    client, *_ = v1_client
    assert "RFP Memory Assistant" in client.get("/").text


def test_status_reports_model_hindsight_and_library(v1_client):
    client, *_ = v1_client
    status = client.get("/v1/status").json()
    assert status["hindsight"]["healthy"] is True
    assert status["hindsight"]["extraction_mode"] == "chunks" and status["hindsight"]["baseline_ok"] is True
    assert status["library"] == {"answers": 0, "pending_sync": 0}


def test_full_flow_library_project_review_export(v1_client):
    client, ctx, llm, memory = v1_client

    # 1. Import a past proposal and confirm its pairs.
    proposal = client.post("/v1/library", files=upload(), data={"client": "Northwind Bank", "industry": "finance", "result": "won"})
    assert proposal.status_code == 202, proposal.text
    wait_job(client, proposal.json()["job"]["id"])
    detail = client.get(f"/v1/library/proposals/{proposal.json()['id']}").json()
    assert detail["status"] == "extracted" and len(detail["pairs"]) == 2
    confirmed = client.post(
        f"/v1/library/proposals/{detail['id']}/confirm",
        json={"pairs": [{"id": p["id"], "decision": "kept"} for p in detail["pairs"]]},
    ).json()
    assert confirmed["created"] == ["ANS-0001", "ANS-0002"]
    assert client.post("/v1/sync").json()["pending"] == 0
    assert set(memory.docs) == {"ANS-0001", "ANS-0002"}

    # 2. Search the library (same plain retrieval drafting uses).
    found = client.get("/v1/library/answers", params={"query": "SCIM provisioning"}).json()
    assert found["answers"][0]["id"] == "ANS-0002" and found["answers"][0]["retrieval"]["position"] == 1
    # Both answers share one won proposal: each must report the win, not just the last one built
    # (regression: a dict keyed by proposal id silently dropped every answer but one).
    assert {a["id"]: a["outcome"] for a in found["answers"]} == {
        "ANS-0002": {"result": "won", "loss_reason": None}, "ANS-0001": {"result": "won", "loss_reason": None},
    }

    # 3. New project → requirements → draft all.
    project = client.post("/v1/projects", files=upload("rfp.docx"), data={"name": "Harborview", "client": "Harborview CU"}).json()
    wait_job(client, project["job"]["id"])
    view = client.get(f"/v1/projects/{project['id']}").json()
    assert view["state"] == "requirements_extracted" and len(view["requirements"]) == 2
    job = client.post(f"/v1/projects/{project['id']}/draft").json()
    assert wait_job(client, job["id"])["status"] == "completed"
    view = client.get(f"/v1/projects/{project['id']}").json()
    assert view["state"] == "in_review" and view["stats"]["drafted"] == 2
    scim = next(r for r in view["requirements"] if r["reference"] == "3.2")
    assert scim["draft"]["sources"] == ["ANS-0002"]
    assert scim["draft"]["retrieved"][0]["answer"].startswith("We support SCIM")
    assert scim["draft"]["prompt_version"] == "v1.0"

    # 4. Export is blocked until everything is final.
    blocked = client.get(f"/v1/projects/{project['id']}/export", params={"format": "docx"})
    assert blocked.status_code == 409 and blocked.json()["error"]["code"] == "not_final"

    # 5. Review both, then export.
    for r in view["requirements"]:
        reviewed = client.post(f"/v1/requirements/{r['id']}/review", json={"action": "accepted"}).json()
        assert reviewed["final"] is True
    assert client.get(f"/v1/projects/{project['id']}").json()["state"] == "approved"
    exported = client.get(f"/v1/projects/{project['id']}/export", params={"format": "docx"})
    assert exported.status_code == 200
    text = "\n".join(p.text for p in docx.Document(io.BytesIO(exported.content)).paragraphs)
    assert "We support SCIM 2.0 provisioning with Okta." in text
    assert client.get(f"/v1/projects/{project['id']}").json()["state"] == "exported"
    xlsx = client.get(f"/v1/projects/{project['id']}/export", params={"format": "xlsx"})
    assert xlsx.status_code == 422  # a Word RFP can't be exported as Excel

    # 6. Accepted answers joined the library and are findable next time.
    client.post("/v1/sync")
    assert client.get("/v1/status").json()["library"]["answers"] == 4


def test_regenerate_and_reject_over_http(v1_client):
    client, ctx, llm, memory = v1_client
    proposal = client.post("/v1/library", files=upload(), data={"result": "won"}).json()
    wait_job(client, proposal["job"]["id"])
    pairs = client.get(f"/v1/library/proposals/{proposal['id']}").json()["pairs"]
    client.post(f"/v1/library/proposals/{proposal['id']}/confirm", json={"pairs": [{"id": p["id"], "decision": "kept"} for p in pairs]})
    project = client.post("/v1/projects", files=upload("rfp.docx")).json()
    wait_job(client, project["job"]["id"])
    wait_job(client, client.post(f"/v1/projects/{project['id']}/draft").json()["id"])
    first = client.get(f"/v1/projects/{project['id']}").json()["requirements"][0]

    regenerated = client.post(f"/v1/requirements/{first['id']}/regenerate", json={"instructions": "Be brief"}).json()
    assert regenerated["draft_count"] == 2 and regenerated["draft"]["instructions"] == "Be brief"
    rejected = client.post(f"/v1/requirements/{first['id']}/review", json={"action": "rejected"}).json()
    assert rejected["final"] is False


def test_failed_project_can_attach_a_fact_sheet_and_retry(v1_client):
    client, _ctx, llm, _memory = v1_client
    llm.fail_requirements = LLMError("api_error", "requirement extraction: Gemini API error 503")
    created = client.post("/v1/projects", files=upload("retry.docx"), data={"name": "Retry me"}).json()
    assert wait_job(client, created["job"]["id"])["status"] == "failed"
    failed = client.get(f"/v1/projects/{created['id']}").json()
    assert failed["state"] == "failed" and "503" in failed["error"]

    custom = json.dumps({
        "company": "Meridian",
        "facts": [{"id": "FACT-900", "topic": "Identity", "statement": "Meridian supports SAML 2.0."}],
    }).encode()
    attached = client.put(
        f"/v1/projects/{created['id']}/fact-sheet",
        files={"file": ("meridian-facts.json", custom, "application/json")},
    )
    assert attached.status_code == 200, attached.text
    assert attached.json()["company"] == "Meridian"
    assert attached.json()["fact_sheet_filename"] == "meridian-facts.json"
    assert [fact["id"] for fact in attached.json()["facts"]] == ["FACT-900"]

    llm.fail_requirements = None
    retried = client.post(f"/v1/projects/{created['id']}/retry")
    assert retried.status_code == 202, retried.text
    assert wait_job(client, retried.json()["id"])["status"] == "completed"
    ready = client.get(f"/v1/projects/{created['id']}").json()
    assert ready["state"] == "requirements_extracted"

    # A successful but incomplete extraction can be replaced before drafting, without re-uploading.
    reextracted = client.post(f"/v1/projects/{created['id']}/retry")
    assert reextracted.status_code == 202, reextracted.text
    assert wait_job(client, reextracted.json()["id"])["status"] == "completed"
    assert client.get(f"/v1/projects/{created['id']}").json()["state"] == "requirements_extracted"

    drafted = client.post(f"/v1/projects/{created['id']}/draft").json()
    assert wait_job(client, drafted["id"])["status"] == "completed"
    assert llm.fact_sets[-1] == ["FACT-900"]
    refused = client.post(f"/v1/projects/{created['id']}/retry")
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "invalid_state"


def test_project_creation_validates_the_optional_fact_sheet(v1_client):
    client, *_ = v1_client
    response = client.post(
        "/v1/projects",
        files=upload("rfp.docx") | {"fact_sheet": ("facts.json", b"{broken", "application/json")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_fact_sheet"
    assert client.get("/v1/projects").json() == []


def test_errors_use_the_standard_shape(v1_client):
    client, *_ = v1_client
    assert client.get("/v1/projects/999").json()["error"]["code"] == "not_found"
    bad = client.post("/v1/library", files=upload("deck.pptx", b"x"))
    assert bad.status_code == 415 and bad.json()["error"]["code"] == "unsupported_file_type"
    assert client.delete("/v1/library/answers/ANS-0404").status_code == 404
    bad_date = client.post("/v1/library", files=upload(), data={"submitted_on": "yesterday"})
    assert bad_date.status_code == 422
    bad_result = client.post("/v1/library", files=upload(), data={"result": "maybe"})
    assert bad_result.status_code == 422


def test_interrupted_jobs_resume_on_startup(tmp_path):
    """A job left 'running' by a dead process is resumed when the app starts."""
    from backend.v1 import projects
    from backend.v1.db import Job, Project

    llm = FakeV1LLM(requirements=[req("Q1?"), req("Q2?")])
    ctx = make_context(tmp_path, llm)
    import asyncio

    async def prepare():
        project, job = projects.create_project(ctx, filename="rfp.docx", data=docx_bytes("x"), name="P", client=None, industry=None)
        await ctx.jobs.wait(job.id)
        with ctx.db.session() as s:
            s.get(Project, project.id).state = "drafting"
            stuck = Job(kind="draft_all", target_id=project.id, status="running", payload={})
            s.add(stuck)
            s.commit()
            return project.id, stuck.id

    project_id, job_id = asyncio.run(prepare())
    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as client:
            assert wait_job(client, job_id)["status"] == "completed"
            assert client.get(f"/v1/projects/{project_id}").json()["stats"]["drafted"] == 2
    finally:
        app.state.v1_factory = None


def test_status_without_a_cloud_key_does_not_read_the_bank(tmp_path):
    ctx = make_context(tmp_path, FakeV1LLM(), FakeMemory(), hindsight_url="https://api.hindsight.vectorize.io")
    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as client:
            hindsight = client.get("/v1/status").json()["hindsight"]
    finally:
        app.state.v1_factory = None
    assert hindsight["cloud"] is True and hindsight["api_key_set"] is False
    assert hindsight["extraction_mode"] is None and hindsight["baseline_ok"] is False


def test_the_same_file_cannot_be_imported_twice_until_discarded(v1_client):
    client, *_ = v1_client
    data = docx_bytes("3.1 Q", "Response: A")
    first = client.post("/v1/library", files=upload(data=data)).json()
    wait_job(client, first["job"]["id"])
    again = client.post("/v1/library", files=upload("renamed.docx", data))
    assert again.status_code == 409 and again.json()["error"]["code"] == "already_imported"
    assert f"#{first['id']}" in again.json()["error"]["message"]

    discarded = client.post(f"/v1/library/proposals/{first['id']}/discard")
    assert discarded.status_code == 200 and discarded.json()["status"] == "discarded"
    assert client.get("/v1/library/proposals").json() == []  # hidden from the list
    assert client.post(f"/v1/library/proposals/{first['id']}/confirm", json={"pairs": []}).status_code == 409
    assert client.post(f"/v1/library/proposals/{first['id']}/discard").status_code == 404

    redo = client.post("/v1/library", files=upload(data=data))
    assert redo.status_code == 202  # a discarded import no longer blocks the file


def test_a_confirmed_proposal_cannot_be_discarded(v1_client):
    client, *_ = v1_client
    proposal = client.post("/v1/library", files=upload()).json()
    wait_job(client, proposal["job"]["id"])
    pairs = client.get(f"/v1/library/proposals/{proposal['id']}").json()["pairs"]
    client.post(f"/v1/library/proposals/{proposal['id']}/confirm", json={"pairs": [{"id": p["id"], "decision": "kept"} for p in pairs]})
    refused = client.post(f"/v1/library/proposals/{proposal['id']}/discard")
    assert refused.status_code == 409 and "Delete its answers" in refused.json()["error"]["message"]
