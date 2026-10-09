"""Engine rules: one-shot confirmations, cancel, nothing spent unasked, agent down, restart re-attach."""

from __future__ import annotations

import asyncio

import pytest

from .conftest import make_engine
from .helpers import RFP_DOC, action, actions_of, cards, click, say, texts

pytestmark = pytest.mark.anyio


async def test_a_confirmation_token_fires_only_once(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Brief me on Cedarline")
    token = action(ask, "Yes")["id"]
    first = await click(engine, conv, token)
    second = await click(engine, conv, token)
    assert "already done" in texts(second) and not cards(second)
    assert fakes.deal.model_calls == 1 and len(fakes.deal.spending_calls()) == 1
    assert any(c["title"].startswith("Brief:") for c in cards(first))


async def test_typed_yes_confirms_once_and_a_second_yes_does_nothing(engine, fakes):
    conv = engine.new_conversation()
    await say(engine, conv, "Brief me on Cedarline")
    done = await say(engine, conv, "yes")
    assert cards(done) and fakes.deal.model_calls == 1
    again = await say(engine, conv, "yes")
    assert "Nothing is waiting for a yes" in texts(again) and fakes.deal.model_calls == 1


async def test_cancel_by_button_and_by_word_spends_nothing_and_kills_the_token(engine, fakes):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Brief me on Cedarline")
    cancel = [a for a in actions_of(ask) if a["id"].startswith("cancel:")][0]
    out = await click(engine, conv, cancel["id"])
    assert "I haven't done anything" in texts(out)
    late = await click(engine, conv, action(ask, "Yes")["id"])
    assert "cancelled" in texts(late) and fakes.all_spending() == []

    ask2 = await say(engine, conv, "Brief me on Cedarline")
    out2 = await say(engine, conv, "no")
    assert "I haven't done anything" in texts(out2)
    assert "cancelled" in texts(await click(engine, conv, action(ask2, "Yes")["id"]))
    assert "There's nothing to cancel" in texts(await say(engine, conv, "cancel"))
    assert fakes.all_spending() == []


async def test_a_new_request_expires_the_old_confirmation(engine, fakes):
    conv = engine.new_conversation()
    old = await say(engine, conv, "Brief me on Cedarline")
    await say(engine, conv, "Brief me on Juniper")
    stale = await click(engine, conv, action(old, "Yes")["id"])
    assert "expired" in texts(stale) and fakes.all_spending() == []


async def test_unknown_or_forged_action_ids_do_nothing(engine, fakes):
    conv = engine.new_conversation()
    out = await click(engine, conv, "deadbeef")
    assert "isn't available any more" in texts(out)
    out2 = await click(engine, conv, "cancel:nope")
    assert "already done" in texts(out2)
    assert fakes.all_spending() == [] and fakes.deal.calls == [] and fakes.rfp.calls == []


async def test_nothing_that_spends_is_ever_called_without_a_confirmation(engine, fakes):
    """Every message that can lead to a model call stops at a question first."""
    conv = engine.new_conversation()
    for text, files in [
        ("Brief me on Cedarline", []), ("Brief me on Juniper", []), ("brief me on Larkfield", []),
        ("Answer this RFP", [RFP_DOC]), ("", [RFP_DOC]), ("New deal Tidewater at Tidewater Freight", []),
        ("We lost the Juniper deal because of price", []), ("accept all grounded answers", []), ("draft it", []),
        ("export it", []), ("yes please do it all", []),
    ]:
        await say(engine, conv, text, files)
        assert fakes.all_spending() == [], f"{text!r} spent something without a confirmation"
    assert fakes.deal.model_calls == 0 and fakes.rfp.model_calls == 0


async def test_the_budget_never_allows_more_calls_than_approved(engine, fakes):
    st = {"actions": {}}
    engine.approve(st, 2)
    assert engine.spend(st) and engine.spend(st) and not engine.spend(st)
    assert st["budget"] == {"approved": 2, "spent": 2}


async def test_a_second_run_cannot_start_while_one_is_running_and_the_token_survives(engine, fakes):
    fakes.deal.job_polls = 30
    conv = engine.new_conversation()
    first = await say(engine, conv, "Brief me on Cedarline")
    await engine.handle_action(conv, action(first, "Yes")["id"])  # starts a background run
    await engine.handle_message(conv, "Brief me on Larkfield Platform")
    second = engine.store.events(conv)[-1]
    assert second["actions"], "the second request should still ask first"
    busy = await click_nowait(engine, conv, [a for a in second["actions"] if a["label"].startswith("Yes")][0]["id"])
    assert "still working on the previous request" in busy
    fakes.deal.job_polls = 1
    await engine.wait_idle(conv)
    assert fakes.deal.model_calls == 1  # the second request did not spend
    retry = await click(engine, conv, [a for a in second["actions"] if a["label"].startswith("Yes")][0]["id"])
    assert fakes.deal.model_calls == 2 and any(c["title"].startswith("Brief:") for c in cards(retry))


async def click_nowait(engine, conv, action_id) -> str:
    before = engine.store.last_n(conv)
    await engine.handle_action(conv, action_id)
    return texts(engine.store.events(conv, before))


async def test_agent_down_says_how_to_start_it(engine, fakes):
    fakes.deal.down = True
    conv = engine.new_conversation()
    reply = await say(engine, conv, "Brief me on Cedarline")
    error = [e for e in engine.store.events(conv) if e["kind"] == "error"][0]
    assert "Deal Intelligence isn't running right now" in error["text"]
    assert "run.ps1 in the deal-intelligence folder (port 8002)" in error["text"] and reply

    fakes.rfp.down = True
    rfp = await say(engine, conv, "", [RFP_DOC])
    err = [e for e in rfp if e["kind"] == "error"][0]
    assert "RFP Memory Assistant isn't running" in err["text"] and "rfp-v0" in err["text"]


async def test_agent_going_down_mid_job_ends_with_a_message_and_no_retry(engine, fakes):
    fakes.deal.go_down_after_job_start = True
    conv = engine.new_conversation()
    ask = await say(engine, conv, "Brief me on Cedarline")
    done = await click(engine, conv, action(ask, "Yes")["id"])
    assert any(e["kind"] == "error" and "isn't running" in e["text"] for e in done)
    assert fakes.deal.model_calls == 1
    assert not engine.busy(conv) and engine.store.running_jobs() == []  # marked lost, so a restart won't chase it
    assert engine.load(conv)["run"] is None


async def test_an_unexpected_bug_ends_as_a_chat_message_not_a_dead_task(engine, fakes, monkeypatch):
    async def boom(*_a, **_k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(engine.deal, "list_deals", boom)
    conv = engine.new_conversation()
    out = await say(engine, conv, "Brief me on Cedarline")
    assert any(e["kind"] == "error" and "Something unexpected went wrong" in e["text"] for e in out)


# -- restart re-attach -------------------------------------------------------------------------------------


async def start_long_run(engine, fakes, conv):
    fakes.deal.job_polls = 1000
    ask = await say(engine, conv, "Brief me on Cedarline")
    await engine.handle_action(conv, action(ask, "Yes")["id"])
    for _ in range(300):
        if fakes.deal.jobs and engine.store.running_jobs():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the job never started")


async def test_restart_reattaches_to_a_running_job_without_spending_again(engine, fakes, tmp_path):
    conv = engine.new_conversation()
    await start_long_run(engine, fakes, conv)
    assert fakes.deal.model_calls == 1
    await engine.shutdown()
    rows = engine.store.running_jobs()
    assert [(r["app"], r["step"]) for r in rows] == [("deals", "brief")]
    engine.store.close()

    fakes.deal.job_polls = 1
    revived = make_engine(fakes, tmp_path / "hub-data")
    try:
        assert await revived.resume() == 1
        await revived.wait_idle()
        events = revived.store.events(conv)
        assert any("I'm back after a restart" in (e.get("text") or "") for e in events)
        assert any(c["title"].startswith("Brief:") for c in cards(events))
        assert fakes.deal.model_calls == 1, "re-attaching must not repeat the model call"
        assert revived.store.running_jobs() == [] and revived.load(conv)["run"] is None
    finally:
        revived.store.close()


async def test_restart_between_a_start_step_and_its_job_id_never_repeats_the_spend(engine, fakes, tmp_path):
    conv = engine.new_conversation()
    await start_long_run(engine, fakes, conv)
    await engine.shutdown()
    st = engine.load(conv)  # make the saved state look like the stop happened before the job id was stored
    st["run"].update(step="brief_start", job_id=None)
    engine.save(conv, st)
    engine.store.close()

    revived = make_engine(fakes, tmp_path / "hub-data")
    try:
        assert await revived.resume() == 1
        await revived.wait_idle()
        text = texts(revived.store.events(conv))
        assert "I was interrupted just before starting that job" in text
        assert fakes.deal.model_calls == 1 and revived.load(conv)["run"] is None
    finally:
        revived.store.close()


async def test_restart_reattaches_to_rfp_drafting(engine, fakes, tmp_path):
    conv = engine.new_conversation()
    ask = await say(engine, conv, "", [RFP_DOC])
    extract = await click(engine, conv, action(ask, "Yes")["id"])
    fakes.rfp.job_polls = 1000
    await engine.handle_action(conv, actions_of(extract)[0]["id"])
    for _ in range(300):
        if fakes.rfp.model_calls == 7 and engine.store.running_jobs():
            break
        await asyncio.sleep(0.01)
    await engine.shutdown()
    engine.store.close()

    fakes.rfp.job_polls = 1
    revived = make_engine(fakes, tmp_path / "hub-data")
    try:
        assert await revived.resume() == 1
        await revived.wait_idle()
        events = revived.store.events(conv)
        assert any(c["title"].startswith("Drafts:") for c in cards(events))
        assert fakes.rfp.model_calls == 7 and len(fakes.rfp.called("POST", "/draft")) == 1
    finally:
        revived.store.close()


async def test_a_conversation_with_stale_job_rows_but_no_run_is_closed_out_not_resumed(engine, fakes, tmp_path):
    conv = engine.new_conversation()
    engine.store.add_job(conv, "deals", 5, "brief")
    assert await engine.resume() == 0
    assert engine.store.running_jobs() == []
