"""Deterministic, explainable similarity and play ranking. Pure functions: no IO, no model, no Hindsight.

Hindsight proposes candidate closed deals by meaning; this module decides which of them count as
similar and what to recommend from them. The invariant is "memory never lifts an off-topic item above
a relevant one":

  STRUCTURAL GATE. A closed deal counts as similar to an open deal only if it shares at least
  `min_shared_keys` PROBLEM keys with it (an objection type or a competitor), OR it has the same
  segment AND the same industry. Hindsight's rank is only the final tiebreak, and lesson evidence
  never moves a deal or a play past the gate.

Counts are honest: a closed deal that won despite the pattern is counted like any other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .contracts import AvoidPlay, PlayRecommendation, Warning
from .db import QUALITY_LOSSES

if TYPE_CHECKING:
    from .db import Deal, DealSignals, Stakeholder

SEGMENT_LABELS = {"smb": "small-business", "mid_market": "mid-market", "enterprise": "enterprise"}
LOSS_LABELS = {
    "competitor": "a competitor won", "price": "price", "no_decision": "no decision was made",
    "champion_left": "the champion left", "security_compliance": "security and compliance",
    "feature_gap": "a feature gap", "timing": "timing", "unresolved_objection": "an unresolved objection",
}
OBJECTION_LABELS = {"sso": "SSO"}
STATUS_PHRASES = {"raised": "raised", "addressed": "addressed", "unresolved": "stayed unresolved"}
OPEN_STATUS_PHRASES = {"raised": "raised", "addressed": "addressed", "unresolved": "unresolved"}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def label(key: str) -> str:
    return OBJECTION_LABELS.get(key) or key.replace("_", " ")


def segment_label(segment: str | None) -> str | None:
    return SEGMENT_LABELS.get(segment or "", segment.replace("_", "-") if segment else None)


@dataclass(frozen=True)
class DealFacts:
    """What the gate, the warnings and the recommendations know about one deal."""

    deal_id: int
    code: str
    account: str
    industry: str | None
    segment: str | None
    stage: str
    result: str
    loss_reason: str | None
    objections: tuple[tuple[str, str], ...] = ()  # (type, status)
    competitors: tuple[str, ...] = ()  # display names
    plays_used: tuple[str, ...] = ()
    sponsor_state: str = "none"  # champion_engaged | champion_silent | none
    economic_buyer: str = "none"  # engaged | silent | none

    @property
    def objection_types(self) -> frozenset[str]:
        return frozenset(t for t, _s in self.objections)

    @property
    def unresolved_types(self) -> frozenset[str]:
        return frozenset(t for t, s in self.objections if s == "unresolved")

    @property
    def competitor_slugs(self) -> frozenset[str]:
        return frozenset(slug(c) for c in self.competitors if slug(c))

    @property
    def problem_keys(self) -> frozenset[str]:
        return frozenset(f"objection:{t}" for t in self.objection_types) | frozenset(
            f"competitor:{c}" for c in self.competitor_slugs
        )

    @property
    def industry_key(self) -> str | None:
        return slug(self.industry) if self.industry else None


def deal_facts(deal: Deal, signals: DealSignals | None, stakeholders: list[Stakeholder]) -> DealFacts:
    champions = [s for s in stakeholders if s.stance == "champion"]
    sponsor = ("none" if not champions else "champion_engaged" if any(s.engaged for s in champions)
               else "champion_silent")
    buyers = [s for s in stakeholders if s.economic_buyer]
    buyer = "none" if not buyers else "engaged" if any(s.engaged for s in buyers) else "silent"
    objections = tuple(
        (str(o.get("type")), str(o.get("status") or "raised"))
        for o in (signals.objections if signals else []) or [] if o.get("type")
    )
    return DealFacts(
        deal_id=deal.id, code=deal.code, account=deal.account, industry=deal.industry, segment=deal.segment,
        stage=deal.stage, result=deal.result, loss_reason=deal.loss_reason, objections=objections,
        competitors=tuple(str(c) for c in (signals.competitors if signals else []) or [] if str(c).strip()),
        plays_used=tuple(str(p) for p in (signals.plays_used if signals else []) or []),
        sponsor_state=sponsor, economic_buyer=buyer,
    )


# ---- text built from facts: the same words go into a closed deal's summary and an open deal's query ----


def situation_text(facts: DealFacts, *, closed: bool) -> str:
    """The situation, in plain sentences, with no account name: what recall should match on."""
    kind = " ".join(p for p in (segment_label(facts.segment), facts.industry) if p)
    stage = "" if facts.stage in ("closed", "") else f" at the {facts.stage} stage"
    parts = [f"A {kind} deal{stage}." if kind else f"A deal{stage}."]
    phrases = STATUS_PHRASES if closed else OPEN_STATUS_PHRASES
    if facts.objections:
        parts.append("Objections: " + ", ".join(f"{label(t)} ({phrases.get(s, s)})" for t, s in facts.objections) + ".")
    else:
        parts.append("No objections were recorded.")
    if facts.competitors:
        names = ", ".join(facts.competitors)
        parts.append(f"The competitor was {names}." if closed else f"Competing against {names}.")
    sponsor = {
        "champion_engaged": "A champion was engaged." if closed else "A champion is engaged.",
        "champion_silent": "The champion went silent." if closed else "The champion has gone silent.",
        "none": "There was no champion." if closed else "There is no champion.",
    }[facts.sponsor_state]
    parts.append(sponsor)
    parts.append({
        "engaged": "The economic buyer was engaged." if closed else "The economic buyer is engaged.",
        "silent": "The economic buyer was not engaged." if closed else "The economic buyer is not engaged.",
        "none": "No economic buyer was identified." if closed else "No economic buyer is identified.",
    }[facts.economic_buyer])
    return " ".join(parts)


def recall_query(facts: DealFacts) -> str:
    return situation_text(facts, closed=False)


# ---- the structural gate ---------------------------------------------------------------------------


def shared_keys(open_facts: DealFacts, closed_facts: DealFacts) -> list[str]:
    """Why a closed deal resembles the open one: shared problem keys first, then industry and segment."""
    keys = sorted(open_facts.problem_keys & closed_facts.problem_keys)
    if open_facts.industry_key and open_facts.industry_key == closed_facts.industry_key:
        keys.append(f"industry:{open_facts.industry_key}")
    if open_facts.segment and open_facts.segment == closed_facts.segment:
        keys.append(f"segment:{open_facts.segment}")
    return keys


def problem_key_count(keys: list[str]) -> int:
    return sum(1 for k in keys if k.startswith(("objection:", "competitor:")))


def passes_gate(open_facts: DealFacts, closed_facts: DealFacts, min_shared_keys: int = 1) -> bool:
    if closed_facts.deal_id == open_facts.deal_id:
        return False
    keys = shared_keys(open_facts, closed_facts)
    same_context = any(k.startswith("industry:") for k in keys) and any(k.startswith("segment:") for k in keys)
    return problem_key_count(keys) >= max(1, min_shared_keys) or same_context


def gate_and_order(
    open_facts: DealFacts, candidates: list[tuple[DealFacts, int]], min_shared_keys: int = 1
) -> list[tuple[DealFacts, int, list[str]]]:
    """Keep the candidates that pass the gate, strongest structural match first; the Hindsight rank
    (second element) only breaks ties."""
    kept = []
    for facts, rank in candidates:
        if passes_gate(open_facts, facts, min_shared_keys):
            kept.append((facts, rank, shared_keys(open_facts, facts)))
    kept.sort(key=lambda item: (-problem_key_count(item[2]), -len(item[2]), item[1], item[0].code))
    return kept


# ---- contrast: what worked in similar deals that this deal has not tried -------------------------------


@dataclass
class PlayContrast:
    play_code: str
    used: int = 0
    won: int = 0
    lost_quality: int = 0
    lost_other: int = 0
    source_deals: list[str] = field(default_factory=list)
    won_deals: list[str] = field(default_factory=list)
    problem_keys: int = 0  # most problem keys one winning deal shares with the open deal
    shared: set[str] = field(default_factory=set)  # problem keys shared with the winning deals


@dataclass
class LessonEvidence:
    """What the lessons bank remembers about one play. A tiebreak only: n is small."""

    net: float = 0.0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    evidence: list[str] = field(default_factory=list)  # lesson document ids


def contrast_candidates(open_facts: DealFacts, similar: list[DealFacts]) -> list[PlayContrast]:
    """Plays used in similar WON deals, minus what the open deal already did, with exact counts over
    every similar deal that used the play (won or lost)."""
    done = set(open_facts.plays_used)
    won_plays = {p for f in similar if f.result == "won" for p in f.plays_used}
    out: dict[str, PlayContrast] = {}
    for facts in sorted(similar, key=lambda f: f.code):
        for play in dict.fromkeys(facts.plays_used):
            if play in done or play not in won_plays:
                continue
            row = out.setdefault(play, PlayContrast(play_code=play))
            row.used += 1
            row.source_deals.append(facts.code)
            if facts.result == "won":
                row.won += 1
                row.won_deals.append(facts.code)
                keys = [k for k in shared_keys(open_facts, facts) if k.startswith(("objection:", "competitor:"))]
                row.problem_keys = max(row.problem_keys, len(keys))
                row.shared.update(keys)
            elif facts.result == "lost":
                if (facts.loss_reason or "") in QUALITY_LOSSES:
                    row.lost_quality += 1
                else:
                    row.lost_other += 1
    return list(out.values())


def _key_phrase(key: str) -> str:
    kind, _, value = key.partition(":")
    if kind == "competitor":
        return f"{value.replace('-', ' ').title()} competitor"
    return f"{label(value)} objection"


# The catalogue says which objection a play is meant to resolve. That label puts the play first, but only until this
# company's own outcomes contradict it: once a play has been used in at least this many similar deals and won fewer than
# half of them, the label stops counting. (Both numbers are plain defaults, not tuned on the evaluation worlds.)
LABEL_OVERRULED_MIN_USED = 3
LABEL_OVERRULED_BELOW_WIN_RATE = 0.5


def rank_plays(
    contrast: list[PlayContrast],
    names: dict[str, str],
    *,
    lessons: dict[str, LessonEvidence] | None = None,
    hit_ranks: dict[str, int] | None = None,
    addresses: dict[str, frozenset[str]] | None = None,
    open_objections: frozenset[str] = frozenset(),
) -> list[PlayRecommendation]:
    """Order: plays that address an objection this deal has open (unless this company's outcomes contradict the
    label, see LABEL_OVERRULED_*), shared problem keys desc, won/used ratio desc, used desc, lesson signal desc, best
    Hindsight rank asc. Lesson evidence only breaks ties; it never overrides the gate (it only sees plays that already
    came from gated deals)."""
    lessons = lessons or {}
    hit_ranks = hit_ranks or {}
    addresses = addresses or {}

    def addressed(row: PlayContrast) -> frozenset[str]:
        return addresses.get(row.play_code, frozenset()) & open_objections

    def best_rank(row: PlayContrast) -> int:
        return min((hit_ranks.get(code, 10**6) for code in row.source_deals), default=10**6)

    def lesson_net(row: PlayContrast) -> float:
        return lessons[row.play_code].net if row.play_code in lessons else 0.0

    def label_counts(row: PlayContrast) -> int:
        """How many open objections the catalogue says this play addresses, or 0 when the history has overruled the label."""
        if row.used >= LABEL_OVERRULED_MIN_USED and row.won / row.used < LABEL_OVERRULED_BELOW_WIN_RATE:
            return 0
        return len(addressed(row))

    ordered = sorted(
        contrast,
        key=lambda r: (-label_counts(r), -r.problem_keys, -(r.won / r.used), -r.used, -lesson_net(r), best_rank(r), r.play_code),
    )
    recommendations = []
    for row in ordered:
        reasons = [f"used in {row.used} similar deal{'s' if row.used != 1 else ''}, {row.won} won"]
        if addressed(row):
            reasons.insert(1, "meant to resolve the open " + ", ".join(label(t) for t in sorted(addressed(row))) + " objection")
        if row.lost_quality:
            reasons.append(f"{row.lost_quality} of the lost deals went down on an issue a play could change")
        if row.shared:
            reasons.append("the winning deals shared the " + ", ".join(_key_phrase(k) for k in sorted(row.shared))
                           + " with this deal")
        evidence = lessons.get(row.play_code)
        if evidence and (evidence.positive or evidence.negative):
            reasons.append(f"Hindsight lessons: {evidence.positive} positive, {evidence.negative} negative")
        recommendations.append(PlayRecommendation(
            play_code=row.play_code, name=names.get(row.play_code, row.play_code), used_in_similar=row.used,
            won_in_similar=row.won, lost_quality_in_similar=row.lost_quality, source_deals=list(row.source_deals),
            lesson_signal=round(evidence.net, 4) if evidence else 0.0,
            lesson_evidence=list(evidence.evidence) if evidence else [],
            score=round(row.won / row.used, 4), reasons=reasons,
        ))
    return recommendations


# ---- plays to avoid ----------------------------------------------------------------------------------


def avoid_plays(
    open_facts: DealFacts, similar: list[DealFacts], names: dict[str, str], min_used: int = 3
) -> list[AvoidPlay]:
    """Plays used in at least `min_used` similar closed deals and in none of the similar WON ones.

    `similar` is the gated set the warnings use, so the counts are over deals that really resemble
    `open_facts`. A play the open deal already used is still listed: it is a warning about what the
    team did. Most used first, then by code."""
    users: dict[str, list[DealFacts]] = {}
    for facts in sorted(similar, key=lambda f: f.code):
        if facts.deal_id == open_facts.deal_id or facts.result not in ("won", "lost"):
            continue
        for play in dict.fromkeys(facts.plays_used):
            users.setdefault(play, []).append(facts)
    out = []
    for play, deals in users.items():
        won = sum(f.result == "won" for f in deals)
        if won or len(deals) < max(1, min_used):
            continue
        name = names.get(play, play)
        out.append(AvoidPlay(
            play_code=play, name=name, used_in_similar=len(deals), won_in_similar=0, lost_in_similar=len(deals),
            source_deals=[f.code for f in deals],
            text=f"{name}: used in {len(deals)} similar deal{'s' if len(deals) != 1 else ''}, 0 won.",
        ))
    return sorted(out, key=lambda a: (-a.used_in_similar, a.play_code))


# ---- warnings ---------------------------------------------------------------------------------------


def _warning(kind: str, subject: str, matching: list[DealFacts], objection_type: str | None = None) -> Warning | None:
    if not matching:
        return None
    lost = [f for f in matching if f.result == "lost"]
    won = len(matching) - len(lost)
    text = f"{subject}: {len(lost)} of {len(matching)} similar closed deals {'were' if len(matching) != 1 else 'was'} lost"
    if won:
        text += f" ({won} won despite it)"
    return Warning(kind=kind, text=text + ".", objection_type=objection_type, similar_lost=len(lost),
                   similar_total=len(matching), source_deals=sorted(f.code for f in matching))


def warnings(open_facts: DealFacts, similar: list[DealFacts]) -> list[Warning]:
    """Honest base rates among the similar closed deals: a win despite the pattern counts as a win."""
    out: list[Warning | None] = []
    for objection in sorted(open_facts.unresolved_types):
        matching = [f for f in similar if objection in f.unresolved_types]
        out.append(_warning("unresolved_objection", f"{label(objection)} objection is unresolved", matching, objection))
    if open_facts.sponsor_state == "champion_silent":
        out.append(_warning("silent_sponsor", "The champion has gone silent",
                            [f for f in similar if f.sponsor_state == "champion_silent"]))
    if open_facts.sponsor_state == "none":
        out.append(_warning("no_champion", "There is no champion",
                            [f for f in similar if f.sponsor_state == "none"]))
    for competitor in sorted(open_facts.competitor_slugs):
        name = next((c for c in open_facts.competitors if slug(c) == competitor), competitor)
        out.append(_warning("competitor", f"{name} is in the deal",
                            [f for f in similar if competitor in f.competitor_slugs]))
    return [w for w in out if w is not None]
