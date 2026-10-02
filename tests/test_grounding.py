from __future__ import annotations

from rfp_assistant.grounding import count_words, evaluate_draft
from rfp_assistant.schemas import DraftClaimOut, DraftResult, Requirement

LIVE = {"FACT-001", "FACT-002"}


def req(word_limit: int | None = None) -> Requirement:
    return Requirement(
        id="REQ-001", section="Security", question="Do you support SSO?", mandatory=True,
        word_limit=word_limit, reference="3.1",
    )


def result(answer="We support SAML 2.0 SSO.", claims=None, unsupported=None, needs_sme=False, sme_question=None):
    return DraftResult(
        answer=answer,
        claims=claims if claims is not None else [DraftClaimOut(text=answer, source_ids=["FACT-001"])],
        unsupported_claims=unsupported or [],
        needs_sme=needs_sme,
        sme_question=sme_question,
    )


def test_fully_cited_answer_is_grounded():
    draft = evaluate_draft(req(), result(), LIVE, "m")
    assert draft.status == "drafted"
    assert draft.flags == []
    assert draft.grounded
    assert draft.sources == ["FACT-001"]
    assert draft.claims[0].valid


def test_citing_an_unknown_or_expired_fact_is_flagged():
    claims = [DraftClaimOut(text="a", source_ids=["FACT-001", "FACT-404"]), DraftClaimOut(text="b", source_ids=["FACT-404"])]
    draft = evaluate_draft(req(), result(claims=claims), LIVE, "m")
    assert draft.flags == ["invalid_citation"]
    assert draft.invalid_citations == ["FACT-404"]
    assert draft.sources == ["FACT-001"]
    assert [c.valid for c in draft.claims] == [False, False]
    assert not draft.grounded


def test_a_claim_with_no_citation_is_flagged():
    draft = evaluate_draft(req(), result(claims=[DraftClaimOut(text="a", source_ids=[])]), LIVE, "m")
    assert "invalid_citation" in draft.flags
    assert draft.claims[0].valid is False


def test_bracketed_ids_are_normalised():
    claims = [DraftClaimOut(text="a", source_ids=[" [FACT-002] "])]
    draft = evaluate_draft(req(), result(claims=claims), LIVE, "m")
    assert draft.grounded
    assert draft.sources == ["FACT-002"]


def test_declared_unsupported_claims_are_flagged():
    draft = evaluate_draft(req(), result(unsupported=["We also support SCIM."]), LIVE, "m")
    assert draft.flags == ["unsupported_claims"]
    assert draft.unsupported_claims == ["We also support SCIM."]


def test_answer_without_claims_cannot_be_verified():
    draft = evaluate_draft(req(), result(claims=[]), LIVE, "m")
    assert draft.flags == ["no_claims"]


def test_word_limit_is_flagged_not_truncated():
    long_answer = " ".join(["word"] * 12)
    draft = evaluate_draft(req(word_limit=10), result(answer=long_answer), LIVE, "m")
    assert draft.over_word_limit
    assert "over_word_limit" in draft.flags
    assert draft.word_count == 12
    assert draft.answer == long_answer


def test_needs_sme_with_partial_answer_keeps_the_answer():
    draft = evaluate_draft(req(), result(needs_sme=True, sme_question="Do we support SCIM?"), LIVE, "m")
    assert draft.status == "needs_sme"
    assert draft.answer
    assert draft.sme_question == "Do we support SCIM?"


def test_needs_sme_without_question_gets_a_default_question():
    draft = evaluate_draft(req(), result(answer="", claims=[], needs_sme=True), LIVE, "m")
    assert draft.status == "needs_sme"
    assert draft.sme_question and "Do you support SSO?" in draft.sme_question
    assert draft.flags == []


def test_empty_answer_without_needs_sme_becomes_needs_sme_and_is_flagged():
    draft = evaluate_draft(req(), result(answer="  ", claims=[]), LIVE, "m")
    assert draft.status == "needs_sme"
    assert draft.flags == ["empty_answer"]


def test_sme_question_is_dropped_when_status_is_drafted():
    draft = evaluate_draft(req(), result(sme_question="unused"), LIVE, "m")
    assert draft.sme_question is None


def test_count_words():
    assert count_words("") == 0
    assert count_words("  one two\nthree  ") == 3
