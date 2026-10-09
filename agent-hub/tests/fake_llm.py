"""A scripted stand-in for the hub's language model: no network, and it records every prompt it was given.

    llm = FakeHubLLM(HubPlan(action="tool", tool="list_deals"))        # the same plan every time
    llm = FakeHubLLM([plan_one, plan_two])                              # one per call, the last repeats
    llm = FakeHubLLM(lambda prompt: HubPlan(...))                       # decided from the prompt
    llm = FakeHubLLM(LLMError("quota", "The model quota is used up."))  # raised
"""

from __future__ import annotations

from agent_hub.llm import LLMError, LLMResult
from agent_hub.planner import HubPlan


class FakeHubLLM:
    model = "fake-hub-model"
    input_tokens, output_tokens = 100, 20  # what each call reports having used

    def __init__(self, script) -> None:  # noqa: ANN001
        self.script = script
        self.prompts: list[str] = []
        self.systems: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    async def structured(self, *, system, user, output_format, max_tokens=700, temperature=0.0) -> LLMResult:  # noqa: ANN001
        self.prompts.append(user)
        self.systems.append(system)
        script = self.script
        if isinstance(script, list):
            script = script[min(len(self.prompts) - 1, len(script) - 1)]
        if callable(script) and not isinstance(script, (HubPlan, Exception)):
            script = script(user)
        if isinstance(script, Exception):
            raise script
        return LLMResult(script, self.model, self.input_tokens, self.output_tokens)


def tool(which: str, /, **args) -> HubPlan:  # noqa: ANN003
    return HubPlan(action="tool", tool=which, **args)


def reply(message: str) -> HubPlan:
    return HubPlan(action="reply", message=message)


def clarify(message: str) -> HubPlan:
    return HubPlan(action="clarify", message=message)


__all__ = ["FakeHubLLM", "HubPlan", "LLMError", "clarify", "reply", "tool"]
