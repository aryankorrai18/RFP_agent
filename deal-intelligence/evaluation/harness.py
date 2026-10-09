"""Loads a synthetic world into a real (temporary) Deal Intelligence database, and the systems being compared on it.

Everything the product itself does is the product's own code (seeding, play statistics, lessons, memory sync, retrieval,
brief generation). The baselines are written here, in a few lines each, so a reader can see exactly what they are."""

from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from deal_intelligence.api.v1 import demo, lessons, outcomes
from deal_intelligence.api.v1.context import V1Context, build_context
from deal_intelligence.api.v1.db import Deal
from deal_intelligence.api.v1.retrieval import build_recommendations
from deal_intelligence.config import Settings
from sqlalchemy import select

from .world import SITUATIONS, DealTruth, World, play_class

TODAY = date(2026, 10, 6)  # the date open deals are measured from, so a run is repeatable
TOP = 3


def truncate(world: World, n_train: int) -> World:
    """The same world with only its first `n_train` closed deals (the test deals are unchanged), for the learning curve."""
    keep = world.deals[:n_train] + world.deals[len(world.train):]
    return World(world.seed, world.plays, keep, world.train[:n_train], world.test)


@dataclass
class Loaded:
    ctx: V1Context
    ids: dict[str, int]  # account -> deal id

    async def close(self) -> None:
        """Release the database and memory files so the temporary folder can be removed (Windows keeps them locked)."""
        await self.ctx.shutdown()
        self.ctx.db.engine.dispose()


async def load_world(world: World, folder: Path, *, settings: Settings | None = None, llm_provider=None) -> Loaded:  # noqa: ANN001
    """A fresh database holding the world, with play statistics, lessons and the local memory built the way the product builds them."""
    base = settings or Settings()
    settings = replace(base, db_path=folder / "eval.db", uploads_dir=folder / "uploads", memory_backend="local",
                       hindsight_api_key=None, lessons_enabled=True)
    ctx = build_context(lambda: settings, llm_provider or (lambda _s: None))
    path = world.write(folder / "world.json")
    previous, demo.SEED_PATH = demo.SEED_PATH, path
    try:
        demo.seed_demo(ctx, today=TODAY)
    finally:
        demo.SEED_PATH = previous
    outcomes.rebuild_play_stats(ctx.db)
    lessons.collect_lessons(ctx.db)
    await ctx.sync()
    with ctx.db.session() as session:
        ids = {d.account: d.id for d in session.scalars(select(Deal))}
    return Loaded(ctx, ids)


def temp_folder() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="deal-eval-", ignore_cleanup_errors=True)


# ---- the systems that are compared -------------------------------------------------------------------------

def popularity(train: list[DealTruth]) -> list[str]:
    """Situation-blind: rank plays by their smoothed win rate across every closed deal."""
    uses, wins = Counter(), Counter()
    for deal in train:
        for play in deal.plays_used:
            uses[play] += 1
            wins[play] += deal.result == "won"
    return sorted(uses, key=lambda p: ((wins[p] + 1) / (uses[p] + 2), uses[p]), reverse=True)


def _neighbours(train: list[DealTruth], situation: str) -> list[DealTruth]:
    """Similar deals by the same structural gate the product uses first: they share the deal's primary objection."""
    return [d for d in train if d.situation == situation]


def naive_rag(train: list[DealTruth], situation: str) -> list[str]:
    """Outcome-blind retrieval: 'what did similar deals do?' Plays ranked by how often similar deals used them."""
    counts = Counter(p for d in _neighbours(train, situation) for p in d.plays_used)
    return [p for p, _ in counts.most_common()]


def neighbour_wins(train: list[DealTruth], situation: str) -> list[str]:
    """Outcome-aware but simple: 'what did similar deals that WON use?' No contrast with losses, no warning about plays that lose."""
    counts = Counter(p for d in _neighbours(train, situation) if d.result == "won" for p in d.plays_used)
    return [p for p, _ in counts.most_common()]


async def product_plays(loaded: Loaded, truth: DealTruth, mode: str) -> tuple[list[str], list[str]]:
    """What Deal Intelligence's own retrieval recommends for an open deal and what it warns to avoid (no model call)."""
    source = "sqlite" if mode == "similar" else "memory"
    rec = await build_recommendations(loaded.ctx, loaded.ids[truth.account], mode, source=source)
    return [p.play_code for p in rec.plays], [a.play_code for a in rec.avoid]


def score(situation_name: str, ranked: list[str], avoid: list[str] | None = None) -> dict:
    """Mark a ranked list of plays against the planted rules for this situation."""
    sit = SITUATIONS[situation_name]
    top = ranked[:TOP]
    avoid = avoid or []
    return {
        "hit1": bool(ranked) and play_class(sit, ranked[0]) == "good",
        "hit3": any(play_class(sit, p) == "good" for p in top),
        "trap1": bool(ranked) and play_class(sit, ranked[0]) == "trap",
        "trap3": any(play_class(sit, p) == "trap" for p in top),
        "flagged": any(play_class(sit, p) == "trap" for p in avoid),
        "false_alarm": any(play_class(sit, p) == "good" for p in avoid),
        "empty": not ranked,
        "ranked": top, "avoid": avoid[:TOP],
    }


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return out
