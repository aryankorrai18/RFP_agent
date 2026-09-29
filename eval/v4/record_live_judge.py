"""Record the live V4 judge and human spot-check results from the app's database.

    .\\.venv\\Scripts\\python -m eval.v4.record_live_judge

Writes eval/v4/live_judge_results.json: for every project with a finished before/after comparison,
the drafting conditions, the judge's un-blinded verdicts (both orders), token counts, the judge and
served model versions, and the person's spot-check picks. Makes no model or Hindsight calls.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import select

from backend.config import ROOT, Settings
from backend.main import apply_env_file
from backend import workspaces
from backend.v1 import judge
from backend.v1.db import Database, Job, Project
from backend.v1.experiment import KIND as COMPARE_KIND

OUT = Path(__file__).with_name("live_judge_results.json")
DEVIATIONS = [
    "Corpus: before/after drafts from completed projects in the active app workspace, not the protocol's 200 "
    "synthetic held-out requirements. Only pairs whose answers differ are judged; identical pairs are ties.",
    "Runs: one run, each pair judged in both presentation orders, instead of three repeated runs.",
    "The judge sees official facts and past-answer texts, never which proposals were won or lost.",
    "Each project's drafting.ranking_version says which V2 ranking rule made its drafts.",
]


def main() -> None:
    apply_env_file()
    # Match the application: evaluation evidence belongs to the workspace currently selected in
    # data/workspaces.json, not always to the legacy database from .env.
    settings = workspaces.apply_workspace(Settings.from_env())
    ctx = SimpleNamespace(db=Database(settings.db_path), settings=settings)
    projects = []
    totals = {"pairs": 0, "differing": 0, "hindsight": 0, "plain": 0, "tie": 0, "depends_on_order": 0,
              "identical": 0, "not_judged": 0, "verdicts": 0, "input_tokens": 0, "output_tokens": 0,
              "spot_checked": 0, "spot_check_agreed_with_judge": 0}
    with ctx.db.session() as session:
        ids = session.scalars(select(Job.target_id).where(Job.kind == COMPARE_KIND, Job.status == "completed").distinct()).all()
        names = {p.id: p for p in session.scalars(select(Project).where(Project.id.in_(ids)))}
    for project_id in sorted(ids):
        found = judge.results(ctx, project_id)["judge"]
        if not found or not found["verdicts"]:
            continue
        plan = judge.plan(ctx, project_id, found["judge_model"])
        with ctx.db.session() as session:
            comparison = session.get(Job, found["comparison_job_id"])
            conditions = dict(comparison.payload)
            # Comparisons made before the relevance fix didn't record the rule: they used "rank".
            conditions.setdefault("relevance", "rank")
            conditions["ranking_version"] = ("V2 post-relevance-fix" if conditions["relevance"] == "gated"
                                             else "V2 pre-relevance-fix")
        project = names[project_id]
        projects.append({"project": project.name, "client": project.client, "industry": project.industry,
                         "comparison_job_id": found["comparison_job_id"], "drafting": conditions,
                         "pairs": plan["pairs"], "differing": plan["differing"], "judge": found})
        totals["pairs"] += plan["pairs"]
        totals["differing"] += plan["differing"]
        for key in ("hindsight", "plain", "tie", "depends_on_order", "identical", "not_judged"):
            totals[key] += found["tally"][key]
        totals["verdicts"] += found["verdicts"]
        totals["input_tokens"] += found["tokens"]["input"]
        totals["output_tokens"] += found["tokens"]["output"]
        totals["spot_checked"] += found["human_agreement"]["compared"]
        totals["spot_check_agreed_with_judge"] += found["human_agreement"]["agreed"]
    report = {
        "status": "live_pairwise_judge" if projects else "not_run",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": "eval/v4/protocol-v1.json", "prompt_version": judge.JUDGE_PROMPT_VERSION,
        "deviations_from_protocol": DEVIATIONS, "totals": totals, "projects": projects,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"written": str(OUT.relative_to(ROOT)), **totals}, indent=2))


if __name__ == "__main__":
    main()
