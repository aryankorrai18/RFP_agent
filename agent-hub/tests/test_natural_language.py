"""Understanding loose requests: the model sees the company's own deals and projects as labels, files as labels, offers
choices when a message could mean different things, and keeps every detail it read. Spending still asks first, in code."""

from __future__ import annotations

import pytest

from agent_hub.planner import HubPlan, PlanOption, Planner

from .conftest import make_engine
from .fake_llm import FakeHubLLM, tool
from .helpers import action, actions_of, all_text, cards, click, say, texts

pytestmark = pytest.mark.anyio

PROPOSAL_MD = ("Northstar response.md", b"# Technical proposal\n\n**Prepared for:** Northstar Community Bank  \n"
                                       b"**Submission date:** 18 February 2026\n\n### 1.1 Do you support SSO?\n\n"
                                       b"**Response:** Not today.\n\n### 1.2 Where is data stored?\n\n**Response:** Locally.\n")


@pytest.fixture
def make(fakes, tmp_path):
    made = []

    def build(*script):  # noqa: ANN002
        llm = FakeHubLLM(list(script))
        eng = make_engine(fakes, tmp_path / f"nl{len(made)}", planner=Planner(lambda: llm))
        made.append(eng)
        return eng, llm

    yield build
    for eng in made:
        eng.store.close()


def choose(message: str, *options: PlanOption) -> HubPlan:
    return HubPlan(action="choose", message=message, options=list(options))


# -- A. the model sees the company's own data, as labels only -----------------------------------------------------------

async def test_the_model_sees_this_workspaces_deals_as_labels_and_resolves_a_typo(make, fakes):
    eng, llm = make(tool("record_outcome", deal="D-002", result="won"))
    conv = eng.new_conversation()
    out = await say(eng, conv, "the junipr one closed, we won it")
    prompt = llm.prompts[0]
    assert "deal D-002: " in prompt and "| account: " in prompt
    assert "Zephyr" not in prompt  # a deal of another workspace is never listed
    assert "Ready to record D-002" in texts(out) and fakes.all_spending() == []


async def test_the_data_list_can_be_switched_off(make, fakes, monkeypatch):
    monkeypatch.setenv("HUB_PLANNER_DATA", "off")
    eng, llm = make(tool("help"))
    await say(eng, eng.new_conversation(), "hi there")
    assert "the user's own data" not in llm.prompts[0] and "deal D-00" not in llm.prompts[0]


# -- C. a message that could mean different things becomes buttons ------------------------------------------------------

async def test_an_unclear_file_offers_choices_and_nothing_happens_until_one_is_clicked(make, fakes):
    eng, llm = make(choose("What should I do with Northstar response.md?",
                           PlanOption(label="Save it as a past proposal we won", tool="add_past_proposal", result="won"),
                           PlanOption(label="Answer it as a new RFP", tool="start_rfp")))
    conv = eng.new_conversation()
    out = await say(eng, conv, "", [PROPOSAL_MD])
    assert "(md; 2 questions, 2 with answers; looks like a completed proposal" in llm.prompts[0]
    assert [a["label"] for a in actions_of(out)] == ["Save it as a past proposal we won", "Answer it as a new RFP"]
    assert fakes.all_spending() == [] and fakes.rfp.proposal_bodies == [] and fakes.rfp.create_bodies == []
    saved = await click(eng, conv, action(out, "Save it as a past proposal")["id"])
    rows = dict(cards(saved)[0]["sections"][0]["rows"])
    assert rows["Client"] == "Northstar Community Bank (read from the file)"
    assert rows["Submitted"] == "2026-02-18 (read from the file)" and rows["Outcome"] == "won"
    assert action(saved, "Yes, read it")["cost"] == "1 model call" and fakes.all_spending() == []


async def test_the_other_choice_runs_the_rfp_flow_with_its_own_confirmation(make, fakes):
    eng, _ = make(choose("Which one?", PlanOption(label="Save it as a past proposal", tool="add_past_proposal"),
                         PlanOption(label="Answer it as a new RFP", tool="start_rfp")))
    conv = eng.new_conversation()
    out = await say(eng, conv, "", [PROPOSAL_MD])
    rfp = await click(eng, conv, action(out, "Answer it as a new RFP")["id"])
    assert "Ready to set up Northstar response.md" in texts(rfp) and fakes.all_spending() == []


async def test_a_choice_with_one_option_just_does_it(make, fakes):
    eng, _ = make(choose("Only one thing fits", PlanOption(label="Brief Juniper", tool="brief_deal", deal="D-002")))
    out = await say(eng, eng.new_conversation(), "juniper brief pls")
    assert "Juniper" in texts(out) and action(out, "Yes")


# -- E. a past proposal keeps every detail, and a missing one can be corrected in plain words ---------------------------

async def test_proposal_details_come_from_the_model_and_the_file_and_can_be_changed(make, fakes):
    eng, _ = make(tool("add_past_proposal", result="won", industry="Banking"),
                  tool("answer_pending", client="Northstar Synthetic Bank"))
    conv = eng.new_conversation()
    out = await say(eng, conv, "we won this one, it's a bank", [PROPOSAL_MD])
    rows = dict(cards(out)[0]["sections"][0]["rows"])
    assert rows == {"Client": "Northstar Community Bank (read from the file)", "Industry": "banking",
                    "Submitted": "2026-02-18 (read from the file)", "Outcome": "won"}
    ask = await click(eng, conv, action(out, "Change the details")["id"])
    assert "Tell me what to change" in texts(ask)
    again = await say(eng, conv, "actually it was for northstar synthetic bank")
    rows = dict(cards(again)[0]["sections"][0]["rows"])
    assert rows["Client"] == "Northstar Synthetic Bank" and rows["Submitted"] == "2026-02-18 (read from the file)"
    await click(eng, conv, action(again, "Yes, read it")["id"])
    assert fakes.rfp.proposal_bodies[-1] == {"filename": "Northstar response.md", "client": "Northstar Synthetic Bank",
                                             "industry": "banking", "submitted_on": "2026-02-18", "result": "won", "loss_reason": None}


async def test_missing_details_are_named_before_the_call_is_spent(engine, fakes):
    conv = engine.new_conversation()
    out = await say(engine, conv, "past proposal we won", [("bid.docx", b"not a real docx")])
    rows = dict(cards(out)[0]["sections"][0]["rows"])
    assert rows["Client"] == "not given" and rows["Submitted"] == "not given"
    assert "client and industry and submitted are not given" in texts(out) and action(out, "Change the details")


# -- F. a new deal in loose words -----------------------------------------------------------------------------------------

async def test_a_new_deal_in_loose_words_gets_its_name_industry_and_size(make, fakes):
    eng, _ = make(tool("new_deal", account="Lumen Grid Software", industry="Software", segment="smb"))
    conv = eng.new_conversation()
    out = await say(eng, conv, "add a deal for lumen grid, they're a small software company", [("2026-04-08_call.md", b"notes")])
    assert fakes.deal.created_bodies[-1] == {"name": "Lumen Grid Software", "account": "Lumen Grid Software", "amount": None,
                                             "segment": "smb", "files": ["2026-04-08_call.md"], "industry": "software"}
    assert "Created D-007 Lumen Grid Software" in texts(out)


async def test_no_decision_from_the_model_is_recorded_as_a_no_decision_loss(make, fakes):
    eng, _ = make(tool("record_outcome", deal="Juniper", result="no_decision"))
    out = await say(eng, eng.new_conversation(), "juniper just went quiet, they never decided")
    assert "LOST because: no decision" in all_text(out)
