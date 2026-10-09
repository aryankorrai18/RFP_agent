"""A question about the whole pipeline: the counts are computed in code, the citations are checked. Offline."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest
from fastapi.testclient import TestClient

from deal_intelligence.api.v1 import demo, outcomes
from deal_intelligence.api.v1.portfolio import MAX_DEALS, ask_portfolio, portfolio_view
from deal_intelligence.errors import PipelineError
from deal_intelligence.main import app
from deal_intelligence.providers.base import LLMError
from deal_intelligence.schemas import DealAnswerResult

from .builders import add_deal, add_play, make_context
from .fake_llm import FakeLLM
from .fakes import FakeLessons, FakeMemory


def build(tmp_path, llm=None):
    llm = llm or FakeLLM()
    ctx = make_context(tmp_path, llm=llm)
    add_play(ctx.db, "PLAY-01", "Security pack")
    add_deal(ctx.db, name="Won one", account="Globex", result="won", closed_on=date(2026, 5, 1), plays=["PLAY-01"],
             objections=[("sso", "addressed")], industry="fintech")
    add_deal(ctx.db, name="Lost one", account="Hooli", result="lost", loss_reason="security_compliance",
             closed_on=date(2026, 4, 1), objections=[("sso", "unresolved")], competitors=["Brightline"], industry="fintech")
    add_deal(ctx.db, name="Lost two", account="Initech", result="lost", loss_reason="price", closed_on=date(2026, 3, 1),
             industry="retail", competitors=["Brightline"])
    add_deal(ctx.db, name="Open one", account="Acme", objections=[("sso", "raised")], industry="fintech")
    return ctx, llm


def ask(ctx, question="Why do we lose fintech deals?") -> dict:
    return asyncio.run(ask_portfolio(ctx, question))


def test_the_counts_are_computed_in_code(tmp_path):
    ctx, _llm = build(tmp_path)
    stats = portfolio_view(ctx)["stats"]
    assert (stats["deals"], stats["won"], stats["lost"], stats["open"]) == (4, 1, 2, 1) and stats["win_rate"] == 0.33
    assert stats["loss_reasons"] == {"security_compliance": 1, "price": 1}
    assert stats["by_objection"]["sso"] == {"won": 1, "lost": 1, "open": 1}
    assert stats["by_competitor"]["Brightline"] == {"lost": 2}
    assert stats["by_industry"]["fintech"] == {"won": 1, "lost": 1, "open": 1}


def test_the_prompt_holds_the_stats_and_every_deal_and_the_question(tmp_path):
    ctx, llm = build(tmp_path)
    ask(ctx)
    user = llm.calls_for(DealAnswerResult)[0].user
    assert "<question>Why do we lose fintech deals?</question>" in user and "<stats>" in user
    assert "loss reasons: price 1, security_compliance 1" in user and user.count("<deal id=") == 4


def test_citations_are_checked_against_the_deals_that_were_offered(tmp_path):
    ctx, llm = build(tmp_path)
    llm.scripts[DealAnswerResult] = DealAnswerResult(found=True, answer="Two deals.", source_ids=["D-002", "D-999", "INT-0001"])
    result = ask(ctx)
    assert result["sources"] == ["D-002"] and result["grounded"]
    assert {f["code"] for f in result["findings"]} == {"unknown_citation", "wrong_id_class"}


def test_an_answer_with_no_valid_citation_is_not_grounded(tmp_path):
    ctx, llm = build(tmp_path)
    llm.scripts[DealAnswerResult] = DealAnswerResult(found=True, answer="Trust me.", source_ids=["D-999"])
    assert ask(ctx)["grounded"] is False


def test_not_found_is_honest(tmp_path):
    ctx, llm = build(tmp_path)
    llm.scripts[DealAnswerResult] = DealAnswerResult(found=False, answer="The deals do not say.", source_ids=[])
    result = ask(ctx)
    assert result["found"] is False and result["grounded"] is True


def test_an_empty_workspace_is_a_409_and_costs_nothing(tmp_path):
    llm = FakeLLM()
    ctx = make_context(tmp_path, llm=llm)
    with pytest.raises(PipelineError) as exc:
        ask(ctx)
    assert exc.value.code == "no_deals" and exc.value.http_status == 409 and llm.calls == []


@pytest.mark.parametrize("question", ["", " ", "x" * 501])
def test_bad_questions_are_refused_before_any_model_call(tmp_path, question):
    ctx, llm = build(tmp_path)
    with pytest.raises(PipelineError) as exc:
        ask(ctx, question)
    assert exc.value.http_status == 422 and llm.calls == []


def test_a_failing_model_is_explained(tmp_path):
    ctx, llm = build(tmp_path, FakeLLM(ask=LLMError("api_error", "boom", reason="quota_exhausted")))
    with pytest.raises(PipelineError) as exc:
        ask(ctx)
    assert exc.value.http_status == 502


def test_a_big_workspace_is_cut_and_says_so(tmp_path):
    ctx, llm = build(tmp_path)
    for n in range(MAX_DEALS):
        add_deal(ctx.db, name=f"Extra {n}", account=f"Extra {n}", result="won", closed_on=date(2026, 1, 1))
    result = ask(ctx)
    assert result["truncated"] and result["deals_considered"] == MAX_DEALS and result["deals_total"] == MAX_DEALS + 4
    assert result["stats"]["deals"] == MAX_DEALS + 4  # the counts always cover every deal


def test_the_route_on_the_demo_workspace(tmp_path):
    ctx = make_context(tmp_path, memory=FakeMemory(), lessons=FakeLessons(), llm=FakeLLM())
    demo.seed_demo(ctx)
    outcomes.rebuild_play_stats(ctx.db)
    app.state.v1_factory = lambda: ctx
    try:
        with TestClient(app) as client:
            good = client.post("/v1/portfolio/ask", json={"question": "Which objection loses us the most deals?"})
            body = good.json()
            assert good.status_code == 200 and body["sources"] and body["stats"]["deals"] == 24
            assert client.post("/v1/portfolio/ask", json={"question": " "}).status_code == 422
    finally:
        app.state.v1_factory = None


def test_the_prompt_says_whose_pipeline_it_is_so_the_team_name_is_not_taken_for_a_customer(tmp_path):
    from deal_intelligence import workspaces
    from deal_intelligence.config import Settings

    workspaces.ensure_registry(Settings())
    ctx, llm = build(tmp_path)
    ask(ctx, "Do you have deals for My deals?")
    call = llm.calls_for(DealAnswerResult)[0]
    assert '<workspace name="My deals"/>' in call.user and "never a customer" in call.system
