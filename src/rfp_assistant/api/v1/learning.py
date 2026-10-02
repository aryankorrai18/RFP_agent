"""Review-signal attribution and the human-readable memory journal."""

from __future__ import annotations

from sqlalchemy import select

from ...errors import PipelineError
from .db import (
    Answer,
    AnswerStats,
    ClientPreference,
    DraftRow,
    MemoryEvent,
    Project,
    RequirementRow,
    Review,
    ReviewFeedback,
    parse_answer_code,
    utcnow,
)

REASON_TAGS = {
    "outdated", "wrong_product", "too_long", "too_short", "too_vague", "incorrect", "client_specific", "tone", "other"
}
# Review reasons that say how this client wants answers written; they become drafting instructions.
PREFERENCE_KEYS = {"too_long", "too_short", "too_vague", "tone", "client_specific"}


def validate_feedback(reason_tags: list[str] | None, rating: int | None) -> list[str]:
    tags = list(dict.fromkeys(reason_tags or []))
    unknown = sorted(set(tags) - REASON_TAGS)
    if unknown:
        raise PipelineError("invalid_request", f"Unknown review reason tags: {', '.join(unknown)}.", 422)
    if rating is not None and rating not in range(1, 6):
        raise PipelineError("invalid_request", "rating must be between 1 and 5.", 422)
    return tags


def apply_review_signals(
    session,
    *,
    project: Project,
    draft: DraftRow,
    review: Review,
    reason_tags: list[str] | None,
    rating: int | None,
) -> None:  # noqa: ANN001
    tags = validate_feedback(reason_tags, rating)
    session.flush()
    session.add(ReviewFeedback(review_id=review.id, reason_tags=tags, rating=rating))

    answer_ids = {
        answer_id for answer_id in (parse_answer_code(code) for code in (draft.sources or [])) if answer_id is not None
    }
    live_ids = set(session.scalars(select(Answer.id).where(Answer.id.in_(answer_ids), Answer.status == "approved")))
    for answer_id in live_ids:
        stats = session.get(AnswerStats, answer_id)
        if stats is None:
            stats = AnswerStats(answer_id=answer_id)
            session.add(stats)
        stats.times_used = (stats.times_used or 0) + 1
        if review.action == "accepted":
            stats.accepted = (stats.accepted or 0) + 1
        elif review.action == "edited":
            if "too_long" in tags:
                stats.length_edits = (stats.length_edits or 0) + 1  # preference, not a quality penalty
            elif (review.edit_distance or 0) <= 0.25:
                stats.light_edits = (stats.light_edits or 0) + 1
            else:
                stats.heavy_edits = (stats.heavy_edits or 0) + 1
        elif review.action == "rewritten":
            stats.rewritten = (stats.rewritten or 0) + 1
        else:
            stats.rejected = (stats.rejected or 0) + 1
        if rating is not None:
            stats.rating_total = (stats.rating_total or 0) + rating
            stats.rating_count = (stats.rating_count or 0) + 1
        if "outdated" in tags:
            stats.outdated_signals = (stats.outdated_signals or 0) + 1
            stats.suggested_supersede = True
        stats.updated_at = utcnow()
        answer = session.get(Answer, answer_id)
        session.add(MemoryEvent(
            kind="review_signal", answer_id=answer_id, project_id=project.id,
            detail=f"{answer.code} was {review.action}" + (f" ({', '.join(tags)})" if tags else "") + ".",
        ))

    if not live_ids:
        # The draft cited no past answer (it answered from official facts, or from nothing), so no
        # answer's track record changes. The review is still recorded, so Hindsight learns it as a
        # lesson about the question and the client, and the offered-but-unused answers are named.
        requirement = session.get(RequirementRow, draft.requirement_id)
        facts = [s for s in (draft.sources or []) if not s.startswith("ANS-")]
        offered = [r.get("id") for r in (draft.retrieved or []) if r.get("id")]
        cited = f"It cited only official facts ({', '.join(facts)})" if facts else "It cited no sources"
        unused = f"; past answers offered but not used: {', '.join(offered)}" if offered else ""
        question = requirement.question if len(requirement.question) <= 160 else requirement.question[:157] + "..."
        session.add(MemoryEvent(
            kind="review_signal", project_id=project.id,
            detail=f"A draft was {review.action}" + (f" ({', '.join(tags)})" if tags else "")
            + f" for the question '{question}'. {cited}{unused}.",
        ))

    if project.client:
        for key in set(tags) & PREFERENCE_KEYS:
            pref = session.scalars(
                select(ClientPreference).where(ClientPreference.client == project.client, ClientPreference.key == key)
            ).first()
            if pref is None:
                pref = ClientPreference(client=project.client, key=key)
                session.add(pref)
            else:
                pref.evidence_count += 1
                pref.updated_at = utcnow()
            session.add(MemoryEvent(
                kind="client_preference", project_id=project.id,
                detail=f"{project.client}: preference signal '{key}' observed.",
            ))


def client_instructions(session, client: str | None) -> str | None:  # noqa: ANN001
    if not client:
        return None
    prefs = session.scalars(
        select(ClientPreference).where(ClientPreference.client == client).order_by(ClientPreference.evidence_count.desc())
    ).all()
    instructions = []
    for pref in prefs:
        if pref.key == "too_long":
            instructions.append("Prefer a concise answer and stay comfortably below any word limit.")
        elif pref.key == "too_short":
            instructions.append("Give a complete answer with useful supporting detail.")
        elif pref.key == "too_vague":
            instructions.append(
                "Be specific: where the official facts or the approved past answers give concrete details (named "
                "standards, firms, numbers, timings), include them instead of general statements."
            )
        elif pref.key == "tone":
            instructions.append("Match the client's formal, direct tone.")
        elif pref.key == "client_specific":
            instructions.append("Prioritise details explicitly relevant to this client.")
    return " ".join(instructions) or None
