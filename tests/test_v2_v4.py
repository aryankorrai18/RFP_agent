"""Token-free coverage for the outcome-learning loop."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.v1.db import (
    Answer,
    AnswerStats,
    ClientPreference,
    Document,
    DraftRow,
    Project,
    RequirementRow,
    Review,
)
from backend.v1.evidence import check_claim
from backend.v1.learning import apply_review_signals
from backend.v1.outcomes import outcome_credit, record_outcome, supersede_answer
from backend.v1.ranking import quality_score, rank_factors
from tests.v1_fakes import FakeV1LLM, make_context


def _answer(answer_id: int, *, client: str = "Other") -> Answer:
    return Answer(
        id=answer_id,
        question="How is access managed?",
        answer="We support SAML 2.0.",
        source="library",
        client=client,
        industry="finance",
        status="approved",
        superseded_by=None,
        hindsight_status="retained",
        updated_at=datetime(2026, 9, 1, tzinfo=UTC),
    )


def _seed_cited_project(ctx) -> tuple[int, int, int]:
    with ctx.db.session() as session:
        answer = _answer(1, client="Harborview")
        document = Document(
            kind="rfp", filename="test.docx", sha256="0" * 64, stored_path="test.docx"
        )
        project = Project(
            document=document, name="Test", client="Harborview", industry="finance", state="in_review"
        )
        requirement = RequirementRow(
            project=project, order=1, section="Security", reference="1",
            question="Do you support SSO?", mandatory=True, word_limit=None,
        )
        session.add_all([answer, project, requirement])
        session.flush()
        draft = DraftRow(
            requirement_id=requirement.id, job_id=None, version=1, status="drafted",
            answer="We support SAML 2.0.", claims=[], sources=[answer.code],
            unsupported_claims=[], invalid_citations=[], flags=[], word_count=4,
            retrieved=[], prompt_version="v1.0", model="fake",
        )
        session.add(draft)
        session.commit()
        return project.id, requirement.id, draft.id


def test_evidence_checker_is_token_free_and_conservative():
    sources = {"FACT-001": "Data is encrypted at rest with AES-256."}
    assert check_claim("Data is encrypted at rest with AES-256.", ["FACT-001"], sources).status == "supported"
    assert check_claim("Data is encrypted at rest and replicated globally.", ["FACT-001"], sources).status == "partial"
    assert check_claim("Support is available by telephone.", ["FACT-001"], sources).status == "unsupported"
    assert check_claim("Anything", ["FACT-404"], sources).status == "unverifiable"


def test_outcome_ranking_can_promote_a_proven_answer():
    now = datetime(2026, 9, 28, tzinfo=UTC)
    rejected = AnswerStats(answer_id=1, times_used=8, rejected=8, suggested_supersede=True)
    accepted = AnswerStats(answer_id=2, times_used=8, accepted=8)
    first = rank_factors(
        _answer(1), rejected, recall_rank=1, client="Harborview",
        industry="finance", now=now,
    )
    fourth = rank_factors(
        _answer(2, client="Harborview"), accepted, recall_rank=4,
        client="Harborview", industry="finance", now=now,
    )
    assert fourth.score > first.score
    assert "same client" in fourth.reasons
    assert "flagged outdated" in first.reasons


def test_historical_outcome_credit_works_before_the_first_review():
    won = AnswerStats(answer_id=1, times_used=0, outcome_credit=outcome_credit("won", None))
    lost_on_price = AnswerStats(
        answer_id=2, times_used=0, outcome_credit=outcome_credit("lost", "price")
    )
    assert quality_score(won) > quality_score(lost_on_price)


def test_review_signals_learn_without_penalising_length_edits(tmp_path):
    ctx = make_context(tmp_path, FakeV1LLM())
    _, _, draft_id = _seed_cited_project(ctx)
    with ctx.db.session() as session:
        draft = session.get(DraftRow, draft_id)
        requirement = session.get(RequirementRow, draft.requirement_id)
        review = Review(
            draft_id=draft.id, action="edited", final_text="Shorter.", edit_distance=0.8
        )
        session.add(review)
        apply_review_signals(
            session, project=requirement.project, draft=draft, review=review,
            reason_tags=["too_long"], rating=4,
        )
        session.commit()
    with ctx.db.session() as session:
        stats = session.get(AnswerStats, 1)
        pref = session.query(ClientPreference).one()
        assert stats.times_used == 1 and stats.length_edits == 1
        assert stats.heavy_edits == 0
        assert stats.rating_total == 4 and stats.rating_count == 1
        assert (pref.client, pref.key, pref.evidence_count) == ("Harborview", "too_long", 1)


def test_project_outcome_credit_is_idempotent_and_causal(tmp_path):
    ctx = make_context(tmp_path, FakeV1LLM())
    project_id, _, _ = _seed_cited_project(ctx)
    record_outcome(ctx, project_id, result="won")
    record_outcome(ctx, project_id, result="won")
    with ctx.db.session() as session:
        assert session.get(AnswerStats, 1).outcome_credit == pytest.approx(0.1)
    record_outcome(ctx, project_id, result="lost", loss_reason="technical fit")
    with ctx.db.session() as session:
        assert session.get(AnswerStats, 1).outcome_credit == pytest.approx(-0.1)
    record_outcome(ctx, project_id, result="lost", loss_reason="price")
    with ctx.db.session() as session:
        assert session.get(AnswerStats, 1).outcome_credit == pytest.approx(0.0)


def test_answer_can_be_explicitly_superseded(tmp_path):
    ctx = make_context(tmp_path, FakeV1LLM())
    with ctx.db.session() as session:
        session.add_all([_answer(1), _answer(2)])
        session.commit()
    supersede_answer(ctx, "ANS-0001", "ANS-0002")
    with ctx.db.session() as session:
        old = session.get(Answer, 1)
        assert old.superseded_by == 2
        assert old.live is False
        assert old.hindsight_status == "pending_delete"


