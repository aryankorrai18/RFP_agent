"""V1 baseline harness: drives the running app over HTTP, exactly as the UI does.

    python baselines/v1/run_baseline.py library        # import the 3 sample proposals, print their pairs
    python baselines/v1/run_baseline.py confirm        # confirm every extracted pair as "kept", then sync
    python baselines/v1/run_baseline.py run [N]        # N projects from sample_rfp.docx, drafted, NOT reviewed

Runs are never reviewed, so they don't add answers to the library: every run sees the same library.
Results are saved to baselines/v1/run-<timestamp>.json. Fictional samples only.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8001"
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SAMPLES = [
    ("past_northwind_bank_2025.docx", {"client": "Northwind Bank", "industry": "finance", "submitted_on": "2025-07-14", "result": "won"}),
    ("past_meridian_health_2026.docx", {"client": "Meridian Health Partners", "industry": "healthcare", "submitted_on": "2026-02-03", "result": "lost", "loss_reason": "price"}),
    ("past_cobalt_insurance_2025.docx", {"client": "Cobalt Insurance", "industry": "insurance", "submitted_on": "2025-10-21", "result": "won"}),
]

client = httpx.Client(base_url=BASE, timeout=120)


def ok(response: httpx.Response) -> dict | list:
    if response.is_error:
        sys.exit(f"{response.request.method} {response.request.url} -> {response.status_code}: {response.text}")
    return response.json()


def wait_job(job_id: int, label: str) -> dict:
    last = None
    while True:
        job = ok(client.get(f"/v1/jobs/{job_id}"))
        progress = (job["status"], job["done"], job["total"])
        if progress != last:
            print(f"  {label}: {job['status']} {job['done']}/{job['total']}", flush=True)
            last = progress
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(2)


def library() -> None:
    for name, meta in SAMPLES:
        data = (ROOT / "samples" / name).read_bytes()
        proposal = ok(client.post("/v1/library", files={"file": (name, data, DOCX)}, data=meta))
        job = wait_job(proposal["job"]["id"], name)
        detail = ok(client.get(f"/v1/library/proposals/{proposal['id']}"))
        print(f"\n{name}: {detail['status']}, {len(detail['pairs'])} pairs {job.get('error') or ''}")
        for p in detail["pairs"]:
            print(f"  [{p['reference'] or p['order']}] Q: {p['question']}\n      A: {p['answer']}")


def confirm() -> None:
    for proposal in ok(client.get("/v1/library/proposals")):
        if proposal["status"] != "extracted":
            continue
        detail = ok(client.get(f"/v1/library/proposals/{proposal['id']}"))
        created = ok(client.post(f"/v1/library/proposals/{proposal['id']}/confirm",
                                 json={"pairs": [{"id": p["id"], "decision": "kept"} for p in detail["pairs"]]}))
        print(f"{detail['filename']}: {len(created['created'])} answers {created['created'][0]}..{created['created'][-1]}")
    for _ in range(10):
        report = ok(client.post("/v1/sync"))
        if report["pending"] == 0:
            break
        time.sleep(3)
    print("sync:", report, "| status:", ok(client.get("/v1/status"))["library"])


def category(r: dict) -> str:
    d = r["draft"]
    if d is None:
        return "no_draft"
    if d["status"] == "failed":
        return "failed"
    if d["status"] == "needs_sme":
        return "needs_sme"
    return "check" if d["flags"] else "grounded"


def run(n: int) -> None:
    status = ok(client.get("/v1/status"))
    assert status["hindsight"]["baseline_ok"], f"Hindsight not in chunks mode: {status['hindsight']}"
    data = (ROOT / "samples" / "sample_rfp.docx").read_bytes()
    for i in range(1, n + 1):
        started = time.monotonic()
        project = ok(client.post("/v1/projects", files={"file": ("sample_rfp.docx", data, DOCX)},
                                 data={"name": f"V1 baseline run {i}", "client": "Harborview Credit Union", "industry": "finance"}))
        wait_job(project["job"]["id"], f"run {i} extraction")
        view = ok(client.get(f"/v1/projects/{project['id']}"))
        if view["state"] != "requirements_extracted":
            sys.exit(f"run {i}: extraction ended in {view['state']}: {view['error']}")
        job = ok(client.post(f"/v1/projects/{project['id']}/draft", json={}))
        job = wait_job(job["id"], f"run {i} drafting")
        view = ok(client.get(f"/v1/projects/{project['id']}"))
        seconds = round(time.monotonic() - started, 1)

        rows = []
        for r in view["requirements"]:
            d = r["draft"] or {}
            rows.append({
                "code": r["code"], "reference": r["reference"], "question": r["question"], "category": category(r),
                "answer": d.get("answer"), "sources": d.get("sources", []), "flags": d.get("flags", []),
                "invalid_citations": d.get("invalid_citations", []), "unsupported_claims": d.get("unsupported_claims", []),
                "sme_question": d.get("sme_question"), "word_count": d.get("word_count"),
                "retrieved": [{"id": x["id"], "position": x["position"], "final": x["final"], "question": x["question"]} for x in d.get("retrieved", [])],
                "retrieval_warning": d.get("retrieval_warning"), "error": d.get("error"),
            })
        counts = {c: sum(row["category"] == c for row in rows) for c in ("grounded", "check", "needs_sme", "failed", "no_draft")}
        cites_ans = sum(any(s.startswith("ANS-") for s in row["sources"]) for row in rows)
        summary = {
            "run": i, "project_id": view["id"], "seconds": seconds, "requirements": len(rows), **counts,
            "drafts_citing_past_answers": cites_ans,
            "invalid_citations": sum(bool(row["invalid_citations"]) for row in rows),
            "retrieval_warnings": sum(bool(row["retrieval_warning"]) for row in rows),
            "job_warning": job.get("warning"),
        }
        record = {
            "protocol": "V1 baseline (V1_TECHNICAL_DESIGN.md §11)", "created": datetime.now().isoformat(timespec="seconds"),
            "config": {"provider": status["provider"], "model": status["model"], "prompt_version": rows and view["requirements"][0]["draft"]["prompt_version"],
                       "hindsight": {k: status["hindsight"][k] for k in ("url", "bank", "extraction_mode")},
                       "retrieval_top_k": len(rows[0]["retrieved"]) if rows else None, "library_answers": status["library"]["answers"]},
            "summary": summary, "requirements": rows,
        }
        path = OUT / f"run-{datetime.now():%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"run {i}: {json.dumps(summary)} -> {path.name}", flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "library":
        library()
    elif command == "confirm":
        confirm()
    elif command == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    else:
        sys.exit(__doc__)
