"""Plain dataclasses that carry results between the memory layer and the brief layer.

retrieval.py produces `Recommendations`; signals.py produces `Flag`s; briefs.py consumes both and
stores them (as dicts) on the Brief row. Keeping them here lets each side be built and tested alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SimilarDeal:
    deal_id: int
    code: str  # D-xxx
    account: str
    result: str  # won | lost
    loss_reason: str | None
    rank: int  # 1 = most relevant according to Hindsight
    relevance: float  # Hindsight's own score for the summary document (the tiebreak)
    shared_keys: list[str]  # e.g. ["objection:sso", "segment:mid_market"]: why it passed the structural gate
    plays_used: list[str]
    summary: str  # the retained summary text

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlayRecommendation:
    play_code: str
    name: str
    used_in_similar: int  # similar closed deals that used this play (after the gate)
    won_in_similar: int
    lost_quality_in_similar: int
    source_deals: list[str]  # D- ids the counts come from
    lesson_signal: float = 0.0  # net lesson evidence, a tiebreak only (n is small)
    lesson_evidence: list[str] = field(default_factory=list)  # lesson document ids; ranked on, never cited in a brief
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)  # plain-language audit lines, e.g. "used in 3 similar deals, 3 won"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AvoidPlay:
    """A play that was tried in several similar deals and never won one: the obvious move that backfires."""

    play_code: str
    name: str
    used_in_similar: int
    won_in_similar: int
    lost_in_similar: int
    source_deals: list[str]  # D- ids the counts come from
    text: str  # e.g. "Discount offer: used in 4 similar deals, 0 won."

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Warning:
    kind: str  # unresolved_objection | silent_sponsor | no_champion | competitor | stale
    text: str
    objection_type: str | None = None
    similar_lost: int = 0
    similar_total: int = 0
    source_deals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Recommendations:
    deal_id: int
    mode: str  # config.RETRIEVAL_MODES
    n_closed: int  # closed deals in memory (stated in the UI: the sample is small)
    similar: list[SimilarDeal] = field(default_factory=list)
    plays: list[PlayRecommendation] = field(default_factory=list)  # best first; plays already done are excluded
    warnings: list[Warning] = field(default_factory=list)
    avoid: list[AvoidPlay] = field(default_factory=list)  # plays to steer away from; never also in `plays`
    degraded: str | None = None  # why the requested mode was downgraded (Hindsight or lessons unavailable)
    lessons_used: bool = False
    memory_state: str = ""  # stable hash of the ids and counts it was built from

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Flag:
    """A deterministic finding about a deal, computed from SQLite with no model call."""

    code: str  # no_champion | blocker_unengaged | overdue_promise | missed_promise | open_objection | no_economic_buyer | stale_deal | competitor_active
    severity: str  # high | medium | low
    text: str
    evidence: list[str] = field(default_factory=list)  # INT- ids, or stakeholder names

    def to_dict(self) -> dict:
        return asdict(self)
