"""Live regressions for the relevance fix, through the running app (Hindsight lookups only, no model calls).

In the demo workspace where a reviewer really rejected the SCIM answer (larkspur-data-demo), check:
  SCIM:      the rejected answer is no longer first, and an off-topic answer (MFA) doesn't take its place.
  Pen test:  the won, specific Ironbridge answers are still promoted above the vague answer from a lost bid.

    .\\.venv\\Scripts\\python -m eval.v2_relevance.live_regressions [workspace-id]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

B = "http://127.0.0.1:8001/v1"
OUT = Path(__file__).with_name("live_regressions.json")
CHECKS = {
    "scim": "How are user accounts created and removed automatically when staff join or leave?",
    "pen_test": "How often is your platform penetration tested, and by whom?",
}


def call(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(B + path, data, {"Content-Type": "application/json"} if data else {}, method=method)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main() -> None:
    workspace = sys.argv[1] if len(sys.argv) > 1 else "larkspur-data-demo"
    for _ in range(20):
        try:
            previous = call("/workspace")["id"]
            break
        except Exception:
            time.sleep(3)
    call(f"/workspaces/{workspace}/activate", method="POST")
    report = {"workspace": workspace, "checks": {}}
    try:
        for name, question in CHECKS.items():
            rows = call(f"/library/answers?query={urllib.parse.quote(question)}&limit=6")["answers"]
            report["checks"][name] = [{
                "position": (a["retrieval"] or {}).get("position"), "id": a["id"], "client": a["client"],
                "lessons": (a["retrieval"] or {}).get("lessons"), "policy": (a["retrieval"] or {}).get("relevance_policy"),
                "answer": a["answer"][:90],
            } for a in rows]
            print(f"\n{name}: {question}")
            for r in report["checks"][name]:
                print(f"  #{r['position']} {r['id']} x{r['lessons']} [{r['policy']}] | {r['client']} | {r['answer']}")
    finally:
        call(f"/workspaces/{previous}/activate", method="POST")
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nback in workspace {previous}; wrote {OUT.name}")


if __name__ == "__main__":
    main()
