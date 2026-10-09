"""Pick the agent a question belongs to. Plain keyword scoring: no model call, no cost, and every
decision can be explained by the words that matched. A model can be added later as a fallback for
questions that score nothing (see `route`)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY = Path(__file__).with_name("agents.json")
GREETING = re.compile(r"^\s*(hi|hello|hey|help|what can you do|who are you|agents?|list)\b[\s?!.]*$", re.I)


def load_registry(path: Path = REGISTRY) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["agents"]


@dataclass
class Match:
    agent: dict
    score: float
    matched: list[str] = field(default_factory=list)


@dataclass
class Routing:
    kind: str  # greeting | match | none
    best: Match | None = None
    alternatives: list[Match] = field(default_factory=list)
    confidence: float = 0.0


def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def score(agent: dict, message: str) -> Match:
    text = _words(message)
    matched: list[str] = []
    total = 0.0
    for keyword in agent["keywords"]:
        if _words(keyword) in text:
            matched.append(keyword)
            total += 1.0 + 0.75 * (len(keyword.split()) - 1)  # a phrase is stronger evidence than a word
    if _words(agent["name"]) in text:
        total += 3.0
        matched.append(agent["name"])
    return Match(agent, total, matched)


def route(message: str, agents: list[dict]) -> Routing:
    if GREETING.match(message or ""):
        return Routing("greeting")
    ranked = sorted(
        (score(a, message) for a in agents), key=lambda m: (-m.score, m.agent["status"] != "live", m.agent["number"])
    )
    matches = [m for m in ranked if m.score > 0]
    if not matches:
        return Routing("none")
    best, rest = matches[0], matches[1:]
    confidence = best.score / (best.score + sum(m.score for m in rest) + 1.0)
    return Routing("match", best, rest[:2], round(confidence, 2))
