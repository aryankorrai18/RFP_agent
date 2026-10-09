"""A scripted stand-in for the model: no network, deterministic, and it records every call.

    llm = FakeLLM(brief=my_script)              # a callable taking the FakeCall
    llm = FakeLLM(brief=[bad_output, good])     # a sequence: one per call, the last repeats
    llm = FakeLLM(signals=LLMError("api_error", "boom", reason="quota_exhausted"))   # raised

A script is a pydantic output, an Exception (raised), a callable (FakeCall -> output or Exception), or a
list of those. Without a script the defaults below answer from what the prompt offers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from deal_intelligence.providers.base import LLMResult, TokenUsage
from deal_intelligence.schemas import (
    BriefClaim, BriefStep, DealAnswerResult, DealBriefResult, DealSignalsResult, FollowupResult, ObjectionOut,
    StakeholderOut,
)

_INT = re.compile(r'<interaction id="(INT-\d+)"')
_DEAL = re.compile(r'<deal id="(D-\d+)"')
_PLAY = re.compile(r'<play code="(PLAY-\w+)"')


@dataclass
class FakeCall:
    purpose: str
    output_format: type
    system: str
    user: str
    temperature: float | None
    effort: str
    max_tokens: int
    index: int  # 0-based position among the calls to this output format


def default_signals(call: FakeCall) -> DealSignalsResult:
    first = _INT.search(call.user)
    return DealSignalsResult(
        stage="evaluation",
        objections=[ObjectionOut(type="sso", text="Needs SSO", status="raised", evidence=[first.group(1)] if first else [])],
        competitors=["Brightline"],
        stakeholders=[StakeholderOut(name="Dana Cho", title="VP Ops", stance="champion", engaged=True)],
    )


def default_brief(call: FakeCall) -> DealBriefResult:
    """Cites the first INT- id and the first D- id in the prompt, and picks the first offered play."""
    interaction = _INT.search(call.user)
    deal = _DEAL.search(call.user)
    play = _PLAY.search(call.user)
    this_deal = [BriefClaim(text="The deal has an open evaluation.", source_ids=[interaction.group(1)])] if interaction else []
    memory = [BriefClaim(text="A comparable closed deal is on record.", source_ids=[deal.group(1)])] if deal else []
    steps = [BriefStep(play_code=play.group(1), rationale="Offered for this deal.", source_ids=[deal.group(1)] if deal else [])] if play else []
    return DealBriefResult(
        summary="The deal is in evaluation.",
        summary_sources=[interaction.group(1)] if interaction else [],
        this_deal=this_deal, memory=memory, next_steps=steps, missing_info=[],
    )


def default_answer(call: FakeCall) -> DealAnswerResult:
    cited = _INT.search(call.user) or _DEAL.search(call.user)  # a deal's question cites an interaction, a portfolio one a deal
    return DealAnswerResult(found=True, answer="The notes mention it.", source_ids=[cited.group(1)] if cited else [])


def default_followup(call: FakeCall) -> FollowupResult:
    interaction = _INT.search(call.user)
    return FollowupResult(subject="Following up", body="Hi, a short follow-up.", source_ids=[interaction.group(1)] if interaction else [])


class FakeLLM:
    model = "fake-model"

    def __init__(self, *, signals: Any = None, brief: Any = None, ask: Any = None, followup: Any = None, usage: TokenUsage | None = None):
        self.scripts: dict[type, Any] = {DealSignalsResult: signals, DealBriefResult: brief, DealAnswerResult: ask, FollowupResult: followup}
        self.defaults = {DealSignalsResult: default_signals, DealBriefResult: default_brief,
                         DealAnswerResult: default_answer, FollowupResult: default_followup}
        self.usage = usage or TokenUsage(input_tokens=100, output_tokens=20)
        self.calls: list[FakeCall] = []

    def calls_for(self, output_format: type) -> list[FakeCall]:
        return [c for c in self.calls if c.output_format is output_format]

    async def structured(
        self, *, purpose: str, output_format: type, system: str, user: str, effort: str = "medium",
        max_tokens: int = 4000, temperature: float | None = None,
    ) -> LLMResult:
        if output_format not in self.defaults:
            raise AssertionError(f"FakeLLM has no script for {output_format.__name__}")
        index = len(self.calls_for(output_format))
        call = FakeCall(purpose, output_format, system, user, temperature, effort, max_tokens, index)
        self.calls.append(call)
        script = self.scripts[output_format]
        if isinstance(script, list):
            script = script[min(index, len(script) - 1)]
        if script is None:
            script = self.defaults[output_format]
        if callable(script) and not isinstance(script, Exception):
            script = script(call)
        if isinstance(script, Exception):
            raise script
        return LLMResult(output=script, model=self.model, usage=TokenUsage(self.usage.input_tokens, self.usage.output_tokens))
