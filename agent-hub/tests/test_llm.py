"""The hub's model setup: no network, the key comes from the environment or agent-hub/.env."""

from __future__ import annotations

import pytest

from agent_hub import llm
from agent_hub.planner import HubPlan


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for name in (*llm.KEY_VARS, "HUB_MODEL", "HUB_PLANNER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(llm, "_cached", None)


def test_no_key_means_no_model():
    assert llm.has_key() is False and llm.get_llm() is None


def test_a_key_in_the_environment_turns_it_on_and_the_model_can_be_chosen(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert llm.get_llm().model == llm.DEFAULT_MODEL
    monkeypatch.setenv("HUB_MODEL", "gemini-other")
    assert llm.get_llm().model == "gemini-other"


def test_a_key_in_the_env_file_is_picked_up_without_a_restart(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    monkeypatch.setattr(llm, "ENV_FILE", env)
    assert llm.get_llm() is None
    env.write_text("GEMINI_API_KEY=pasted-later\n", encoding="utf-8")
    assert llm.get_llm() is not None


def test_the_planner_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("HUB_PLANNER", "off")
    assert llm.get_llm() is None


def test_the_plan_schema_is_flat_and_has_every_tool():
    schema = llm.json_schema_for(HubPlan)
    assert "$defs" not in schema and "$ref" not in str(schema)
    tools = {v for v in str(schema["properties"]["tool"]).replace("'", " ").split() if "_" in v or v in ("help",)}
    assert {"list_deals", "ask_pipeline", "record_outcome", "answer_pending", "help"} <= tools


@pytest.mark.anyio
async def test_a_call_without_a_key_is_a_clear_error():
    with pytest.raises(llm.LLMError) as exc:
        await llm.GeminiLLM().structured(system="s", user="u", output_format=HubPlan)
    assert exc.value.reason == "no_key" and "agent-hub/.env" in exc.value.message
