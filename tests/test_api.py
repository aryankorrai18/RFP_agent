from __future__ import annotations

import json

from backend.llm import LLMError
from backend.schemas import DraftResponse, ExtractionResult
from tests.conftest import FakeLLM, docx_bytes, grounded, needs_sme, requirement

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def extraction() -> ExtractionResult:
    return ExtractionResult(
        requirements=[
            requirement("Describe SSO support.", section="Security", mandatory=True, word_limit=100, reference="3.1"),
            requirement("Do you support SCIM?", section="Security", reference="3.2"),
        ]
    )


def upload(name="rfp.docx", data=None, content_type=DOCX):
    return {"rfp_file": (name, data if data is not None else docx_bytes("3.1 Describe SSO.", "3.2 SCIM?"), content_type)}


def test_index_page_is_served(api):
    response = api(FakeLLM(extraction())).get("/")
    assert response.status_code == 200
    assert "RFP Memory Assistant" in response.text


def test_health(api):
    body = api(FakeLLM(extraction())).get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_credentials"] in {"env", "not_found_in_env"}


def test_browser_security_headers_and_sensitive_api_no_store(api):
    client = api(FakeLLM(extraction()))
    page = client.get("/")
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["x-frame-options"] == "DENY"
    assert page.headers["referrer-policy"] == "no-referrer"
    assert page.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"

    response = client.get("/v0/fact-sheet")
    assert response.headers["cache-control"] == "no-store"


def test_default_fact_sheet_endpoint(api):
    body = api(FakeLLM(extraction())).get("/v0/fact-sheet").json()
    assert body["company"] == "Larkspur Data"
    assert len(body["facts"]) >= 20


def test_draft_returns_the_documented_schema(api):
    llm = FakeLLM(
        extraction(),
        drafter=lambda r: grounded("We support SAML 2.0.", "FACT-003") if "SSO" in r.question else needs_sme("SCIM?"),
    )
    response = api(llm).post("/v0/draft", files=upload())
    assert response.status_code == 200, response.text
    body = DraftResponse.model_validate(response.json())
    assert body.company == "Larkspur Data"
    assert [d.status for d in body.drafts] == ["drafted", "needs_sme"]
    assert body.drafts[0].sources == ["FACT-003"]
    assert body.stats.grounded == 1
    assert llm.documents[0].filename == "rfp.docx"


def test_uploaded_fact_sheet_is_used(api):
    sheet = {"company": "Other Co", "facts": [{"id": "FACT-1", "statement": "We exist."}]}
    llm = FakeLLM(extraction(), drafter=lambda r: grounded("We exist.", "FACT-1"))
    files = upload() | {"fact_sheet": ("facts.json", json.dumps(sheet).encode(), "application/json")}
    body = api(llm).post("/v0/draft", files=files).json()
    assert body["company"] == "Other Co"
    assert body["stats"]["grounded"] == 2


def test_invalid_fact_sheet_is_422(api):
    files = upload() | {"fact_sheet": ("facts.json", b"{broken", "application/json")}
    response = api(FakeLLM(extraction())).post("/v0/draft", files=files)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_fact_sheet"


def test_missing_file_is_422_in_the_standard_error_shape(api):
    response = api(FakeLLM(extraction())).post("/v0/draft")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_unsupported_type_is_415(api):
    response = api(FakeLLM(extraction())).post("/v0/draft", files=upload("rfp.pptx", b"x", "application/octet-stream"))
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_file_type"


def test_file_over_the_upload_limit_is_413(api):
    big = b"a" * (1024 * 1024 + 1)
    response = api(FakeLLM(extraction()), max_upload_mb=1).post("/v0/draft", files=upload("rfp.txt", big, "text/plain"))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_document_over_the_character_limit_is_413(api):
    response = api(FakeLLM(extraction()), max_document_chars=10).post(
        "/v0/draft", files=upload("rfp.txt", b"a long document", "text/plain")
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "document_too_long"


def test_unreadable_file_is_422(api):
    response = api(FakeLLM(extraction())).post("/v0/draft", files=upload("rfp.docx", b"garbage"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unreadable_file"


def test_extraction_failure_is_502(api):
    response = api(FakeLLM(LLMError("malformed", "requirement extraction: bad output"))).post("/v0/draft", files=upload())
    assert response.status_code == 502
    assert response.json()["error"] == {"code": "extraction_failed", "message": "requirement extraction: bad output"}


def test_missing_credentials_is_reported_clearly(api):
    response = api(FakeLLM(LLMError("auth", "No Anthropic credentials found."))).post("/v0/draft", files=upload())
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_auth_failed"


def test_too_many_requirements_is_422(api):
    response = api(FakeLLM(extraction()), max_requirements=1).post("/v0/draft", files=upload())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "too_many_requirements"


def test_runs_are_listed_newest_first_and_reloadable(api):
    client = api(FakeLLM(extraction()))
    assert client.get("/v0/runs").json() == []
    first = client.post("/v0/draft", files=upload()).json()
    runs = client.get("/v0/runs").json()
    assert [r["run_id"] for r in runs] == [first["run_id"]]
    assert runs[0]["stats"]["requirements"] == 2
    assert runs[0]["created"].startswith(first["run_id"][:4])
    reloaded = client.get(f"/v0/runs/{first['run_id']}").json()
    assert reloaded == first


def test_unknown_or_malicious_run_ids_are_404(api):
    client = api(FakeLLM(extraction()))
    for run_id in ("20260101-000000-abcdef", "..%2F..%2Fdata%2Ffact_sheet", "not-a-run"):
        response = client.get(f"/v0/runs/{run_id}")
        assert response.status_code == 404
        # Traversal attempts never match the route at all; either way the body is the standard shape.
        assert response.json()["error"]["code"] in {"run_not_found", "not_found"}


def test_samples_are_served_from_a_fixed_list(api):
    client = api(FakeLLM(extraction()))
    response = client.get("/v0/samples/sample_rfp.docx")
    assert response.status_code == 200
    assert response.content[:2] == b"PK"
    assert client.get("/v0/samples/..%2Fdata%2Ffact_sheet.json").status_code == 404
    assert client.get("/v0/samples/fact_sheet.json").status_code == 404
