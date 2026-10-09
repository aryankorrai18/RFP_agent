"""Intent parsing: pure functions, no network."""

from __future__ import annotations

import pytest

from agent_hub.intents import (
    LOSS_REASONS, detect, free_reason, match_deal, parse_deal_slots, parse_outcome, parse_reason, parse_rfp_slots,
    reply_slots, resolve_candidates,
)

DEALS = [
    {"id": 1, "code": "D-001", "name": "Cedarline Renewal", "account": "Cedarline Systems"},
    {"id": 2, "code": "D-002", "name": "Juniper Expansion", "account": "Juniper Health"},
    {"id": 3, "code": "D-003", "name": "Larkfield Platform", "account": "Larkfield Bank"},
    {"id": 4, "code": "D-004", "name": "Larkfield Security Add-on", "account": "Larkfield Bank"},
]


def best(text: str) -> list[int]:
    return [c.deal["id"] for c in match_deal(text, DEALS)]


@pytest.mark.parametrize("text, expected", [
    ("cedarline", [1]),
    ("the CEDARLINE deal", [1]),
    ("Juniper Health", [2]),
    ("cedarlin", [1]),  # a typo
    ("D-002", [2]),
    ("d 2", [2]),
    ("what about juniper?", [2]),
])
def test_deal_matching_finds_the_one_deal(text, expected):
    assert best(text) == expected


def test_matching_is_case_insensitive_and_returns_nothing_for_strangers():
    assert best("CeDaRlInE") == [1]
    assert best("Zebra Logistics") == []
    assert best("") == []


def test_ambiguous_names_are_reported_as_ambiguous():
    found = match_deal("Larkfield", DEALS)
    assert {c.deal["id"] for c in found} == {3, 4}
    verdict, picks = resolve_candidates(found)
    assert verdict == "ask" and len(picks) == 2
    verdict, picks = resolve_candidates(match_deal("Larkfield Security", DEALS))
    assert verdict == "one" and picks[0].deal["id"] == 4
    assert resolve_candidates([])[0] == "none"


@pytest.mark.parametrize("text, reason", [
    ("because the price was too high", "price"),
    ("they went with a competitor", "competitor"),
    ("nobody decided, the status quo won", "no_decision"),
    ("our champion left the company", "champion_left"),
    ("the security review failed", "security_compliance"),
    ("we lacked the SSO feature", "feature_gap"),
    ("bad timing, they postponed to next year", "timing"),
    ("the SSO objection never got resolved", "unresolved_objection"),
])
def test_all_eight_loss_reasons_parse(text, reason):
    assert reason in LOSS_REASONS
    assert parse_reason(text) == reason


def test_unclear_or_missing_reasons_are_not_guessed():
    assert parse_reason("we just lost it") is None
    assert parse_reason("the competitor was cheaper and the price was lower") is None  # two reasons: ask
    assert parse_outcome("We lost Larkfield") == ("lost", None)
    assert parse_outcome("We won Cedarline because of the demo") == ("won", None)
    assert parse_outcome("We lost Larkfield because of price") == ("lost", "price")
    assert parse_outcome("what is the weather") == (None, None)


@pytest.mark.parametrize("text, kind", [
    ("yes", "confirm"), ("Yes, go ahead!", "confirm"), ("ok", "confirm"), ("sure thing", "none"), ("go ahead", "confirm"),
    ("no", "cancel"), ("Not now", "cancel"), ("cancel", "cancel"), ("never mind", "cancel"), ("stop", "cancel"),
])
def test_confirm_and_cancel_words(text, kind):
    assert detect(text, False).kind == kind


def test_a_longer_sentence_is_not_a_confirmation():
    assert detect("yes but only for the Cedarline deal", False).kind != "confirm"


@pytest.mark.parametrize("text, kind, subject", [
    ("Brief me on the Cedarline deal", "deal_brief", "Cedarline"),
    ("brief me on Juniper", "deal_brief", "Juniper"),
    ("What's going on with Larkfield?", "deal_brief", "Larkfield"),
    ("prep me for the Cedarline call", "deal_brief", "Cedarline"),
])
def test_brief_verbs(text, kind, subject):
    intent = detect(text, False)
    assert intent.kind == kind and intent.subject == subject and intent.agent == "deals"


def test_outcome_verbs_carry_result_subject_and_reason():
    lost = detect("We lost Larkfield because the SSO issue never got resolved", False)
    assert (lost.kind, lost.result, lost.loss_reason, lost.target) == ("outcome", "lost", "unresolved_objection", "deal")
    assert lost.subject == "Larkfield"
    won = detect("We won the Cedarline deal", False)
    assert (won.kind, won.result, won.loss_reason, won.subject) == ("outcome", "won", None, "Cedarline")
    rfp = detect("We lost this RFP because the price was higher than the incumbent's", False)
    assert (rfp.kind, rfp.target, rfp.result) == ("outcome", "rfp", "lost")
    assert rfp.slots["free_reason"].startswith("the price was higher")


def test_new_deal_slots_and_replies():
    assert parse_deal_slots("Tidewater at Tidewater Freight") == {"name": "Tidewater", "account": "Tidewater Freight"}
    assert parse_deal_slots("with Tidewater Freight") == {"account": "Tidewater Freight"}
    assert parse_deal_slots("Tidewater at Tidewater Freight worth $120k enterprise") == {
        "name": "Tidewater", "account": "Tidewater Freight", "amount": 120000, "segment": "enterprise"}
    intent = detect("New deal Tidewater at Tidewater Freight", True, None, ["thread.eml"])
    assert intent.kind == "deal_new" and intent.slots == {"name": "Tidewater", "account": "Tidewater Freight"}
    assert reply_slots("Tidewater Freight", {"name": "Tidewater"})["account"] == "Tidewater Freight"
    assert reply_slots("account: Tidewater Freight", {"name": "Tidewater"}) == {"name": "Tidewater", "account": "Tidewater Freight"}


def test_files_and_rfp_verbs():
    assert detect("", True, None, ["Acme security questionnaire.docx"]).kind == "rfp_start"
    assert detect("", True, None, ["call notes.eml"]).kind == "deal_files"
    assert detect("", True, None, ["data.bin"]).kind == "files_unclear"
    assert detect("Answer this RFP", True, None, ["x.docx"]).kind == "rfp_start"
    assert detect("accept all grounded answers", False).kind == "rfp_accept_all"
    assert detect("export it as Word", False).kind == "rfp_export"
    assert detect("export to excel", False).fmt == "xlsx"
    assert detect("hello", False).kind == "greeting"
    assert detect("what is the capital of France", False).kind == "question"  # the engine falls back to help


def test_rfp_slots_only_when_stated_and_free_reason():
    assert parse_rfp_slots("it is for Northwind Traders") == {"client": "Northwind Traders"}
    assert parse_rfp_slots("called Q3 Bid, client Northwind, industry retail") == {
        "name": "Q3 Bid", "client": "Northwind", "industry": "retail"}
    assert parse_rfp_slots("here you go") == {}
    assert free_reason("we lost because we were too slow") == "we were too slow"
    assert free_reason("we lost") is None


def test_one_word_finds_a_multi_word_account():
    from agent_hub.intents import match_deal

    deals = [
        {"code": "D-024", "name": "Juniper & Vale Stores - Halcyon Pipeline", "account": "Juniper & Vale Stores"},
        {"code": "D-001", "name": "Fennick Payments - Halcyon Pipeline", "account": "Fennick Payments"},
    ]
    found = match_deal("Juniper", deals)
    assert found and found[0].deal["code"] == "D-024"
    assert [c.deal["code"] for c in match_deal("brief me on the Juniper deal", deals)][:1] == ["D-024"]


def test_trailing_industry_does_not_pollute_the_account():
    from agent_hub.intents import parse_deal_slots

    slots = parse_deal_slots("Hub Test Freight at Hub Test Logistics, mid market logistics")
    assert slots["name"] == "Hub Test Freight" and slots["account"] == "Hub Test Logistics"
    assert slots["segment"] == "mid_market" and slots["industry"] == "logistics"
