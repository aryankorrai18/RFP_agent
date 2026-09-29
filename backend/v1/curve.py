"""The learning curve: how reviewers actually treated each project's drafts, oldest project first.

Everything here is counted from real review history (the Review rows). Nothing is simulated. For each
requirement the latest draft that has been reviewed counts, with its latest review:

    accepted unchanged      the draft was approved as written
    light edit              edited, and the text changed by 25% or less
    heavy edit / rewrite    edited by more than 25%, or written again from scratch
    rejected                thrown out

The review score turns those counts into one number with the fixed weights below. It is a summary,
not a measurement of its own: the counts are always shown beside it.
"""

from __future__ import annotations

from sqlalchemy import select

from .db import DraftRow, Project, RequirementRow

LIGHT_EDIT_LIMIT = 0.25  # the same threshold the answer track record uses
WEIGHTS = {"accepted": 1.0, "light_edit": 0.75, "heavy_edit": 0.25, "rejected": 0.0}


def review_bucket(action: str, distance: float | None) -> str:
    if action == "accepted":
        return "accepted"
    if action == "edited":
        return "light_edit" if (distance or 0) <= LIGHT_EDIT_LIMIT else "heavy_edit"
    if action == "rewritten":
        return "heavy_edit"
    return "rejected"


def learning_curve(session) -> dict:  # noqa: ANN001
    points = []
    for project in session.scalars(select(Project).order_by(Project.created_at, Project.id)):
        counts = {"accepted": 0, "light_edit": 0, "heavy_edit": 0, "rejected": 0}
        distances: list[float] = []
        drafted = needs_sme = unsupported = 0
        modes: set[str] = set()
        models: set[str] = set()
        for requirement in session.scalars(select(RequirementRow).where(RequirementRow.project_id == project.id)):
            drafts = list(session.scalars(select(DraftRow).where(DraftRow.requirement_id == requirement.id).order_by(DraftRow.version)))
            if not drafts or drafts[-1].status == "failed":
                continue
            latest = drafts[-1]
            drafted += 1
            needs_sme += latest.status == "needs_sme"
            unsupported += bool(latest.unsupported_claims) or "evidence_unsupported" in (latest.flags or [])
            if latest.model:
                models.add(latest.model)
            if latest.retrieved and latest.retrieved[0].get("mode"):
                modes.add(latest.retrieved[0]["mode"])
            reviewed = next((d for d in reversed(drafts) if d.reviews), None)
            if reviewed is None:
                continue
            review = reviewed.reviews[-1]
            counts[review_bucket(review.action, review.edit_distance)] += 1
            if review.edit_distance is not None:
                distances.append(review.edit_distance)
        reviewed_count = sum(counts.values())
        if not reviewed_count:
            continue
        points.append({
            "project_id": project.id, "name": project.name, "client": project.client, "industry": project.industry,
            "created_at": project.created_at, "drafted": drafted, "reviewed": reviewed_count, **counts,
            "accepted_rate": round(counts["accepted"] / reviewed_count, 3),
            "avg_edit_distance": round(sum(distances) / len(distances), 3) if distances else None,
            "needs_sme_rate": round(needs_sme / drafted, 3) if drafted else None,
            "unsupported_rate": round(unsupported / drafted, 3) if drafted else None,
            "score": round(sum(WEIGHTS[k] * counts[k] for k in counts) / reviewed_count, 3),
            "retrieval_modes": sorted(modes), "models": sorted(models),
        })
    trend = None
    if len(points) >= 2:
        trend = round(points[-1]["score"] - points[0]["score"], 3)
    mixed = len({m for p in points for m in p["retrieval_modes"]}) > 1 or len({m for p in points for m in p["models"]}) > 1
    return {
        "projects": points, "weights": WEIGHTS, "light_edit_limit": LIGHT_EDIT_LIMIT, "score_change": trend,
        "mixed_conditions": mixed,
        "caveat": "Each project is a different RFP, so a single point is anecdotal. Read the trend over several projects, "
                  "and check the retrieval mode and model beside each point: the curve is only a fair comparison when they match.",
    }
