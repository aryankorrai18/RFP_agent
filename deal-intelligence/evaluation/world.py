"""A synthetic sales world with PLANTED, KNOWN rules, so a recommendation can be marked right or wrong without a judge.

The world is generated from a seed (no model call). Every deal has one primary situation (the objection that decides it)
and the plays the team used. What happens next follows rules written down here, not guessed afterwards:

* In a "control" situation the obvious play (the one the play catalogue says addresses the objection) really is the best
  one. A system that knows nothing should do fine here; memory must not make it worse.
* In a "counter-intuitive" situation this company's history says the obvious play LOSES (a trap) and a less obvious play
  wins. Only a system that learns from outcomes can know that. This is the claim being tested.

Teams in this world over-use the obvious play (45% of deals), so an outcome-blind "what did similar deals do?" lookup
learns the trap. Outcomes are noisy on purpose: a good play wins 88% of the time, a neutral one 50%, a trap 12%.

Nothing here is real sales data. It proves what a mechanism does under stated rules, never what it would earn a real team."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

HALCYON_SEED = Path(__file__).resolve().parent.parent / "src" / "deal_intelligence" / "seed" / "halcyon.json"

P_WIN = {"good": 0.88, "neutral": 0.50, "trap": 0.12}
OBVIOUS_SHARE = 0.45  # how often a team reaches for the obvious play
EXTRA_PLAY_RATE = 0.40  # a second, outcome-neutral play on the deal


@dataclass(frozen=True)
class Situation:
    name: str
    objection: str
    kind: str  # "control" | "counterintuitive"
    obvious: str  # the play the catalogue says addresses this objection
    good: tuple[str, ...]
    trap: tuple[str, ...]
    neutral: tuple[str, ...]
    loss_reasons: tuple[str, ...]  # what a loss on a trap tends to be recorded as
    notes: tuple[str, ...]  # what the customer says (three phrasings; none names a play)


SITUATIONS: dict[str, Situation] = {s.name: s for s in (
    Situation("sso", "sso", "control", "PLAY-10", ("PLAY-10", "PLAY-01"), ("PLAY-02",), ("PLAY-08", "PLAY-09"),
              ("unresolved_objection", "security_compliance"), (
        "{who}, their identity admin, said every system that holds customer data must sign in through their Okta tenant with SAML, no local logins.",
        "{who} asked us to show single sign-on working against their identity provider before the security group will let this go to procurement.",
        "{who} flagged that our product has no SSO story they can approve; their policy blocks any tool without it.")),
    Situation("security_review", "security_review", "control", "PLAY-01", ("PLAY-01",), ("PLAY-02",),
              ("PLAY-08", "PLAY-09", "PLAY-04"), ("security_compliance", "unresolved_objection"), (
        "{who} from compliance sent a 70-question vendor risk questionnaire and asked for our SOC 2 report and a pen-test summary.",
        "{who} said their security review is a gate before any contract and asked how long ours takes.",
        "{who} wants to understand tenant isolation, encryption and where the data lives before going further.")),
    Situation("pricing", "pricing", "counterintuitive", "PLAY-05", ("PLAY-03",), ("PLAY-05",),
              ("PLAY-02", "PLAY-09", "PLAY-08"), ("price", "unresolved_objection"), (
        "{who} said the quote is above what they budgeted and asked what flexibility we have on price.",
        "{who} told us finance compared our price with an alternative and wants us to come down.",
        "{who} asked whether the annual figure is negotiable; the committee thinks it is high.")),
    Situation("integration", "integration", "counterintuitive", "PLAY-06", ("PLAY-04",), ("PLAY-06",),
              ("PLAY-02", "PLAY-09", "PLAY-08"), ("feature_gap", "timing"), (
        "{who} said the deal depends on a clean connection to their ERP and warehouse, and asked how we handle that.",
        "{who} is worried that integrating with their billing system will take their engineers away from other work.",
        "{who} asked whether we have connectors for their stack or whether this means custom work.")),
    Situation("legal_terms", "legal_terms", "counterintuitive", "PLAY-07", ("PLAY-09",), ("PLAY-07",),
              ("PLAY-02", "PLAY-08", "PLAY-03"), ("timing", "unresolved_objection"), (
        "{who} from legal returned our paper with redlines on liability, data processing and termination.",
        "{who} said procurement cannot sign on our standard terms and wants a call with their counsel.",
        "{who} asked for changes to the indemnity and data clauses before they can take this to signature.")),
)}
OTHER_OBJECTIONS = ("sso", "security_review", "pricing", "integration", "timeline", "legal_terms", "support", "data_residency")

ADJ = ("Alder", "Brook", "Cedar", "Dune", "Ember", "Fjord", "Glen", "Harbor", "Iron", "Juniper", "Kestrel", "Lumen", "Maple",
       "Nimbus", "Orchid", "Pine", "Quarry", "Ridge", "Summit", "Tidal", "Umber", "Vale", "Willow", "Zenith")
NOUN = ("Logistics", "Health", "Foods", "Capital", "Retail", "Energy", "Labs", "Bank", "Freight", "Systems", "Media", "Works")
INDUSTRIES = ("fintech", "healthcare", "retail", "logistics", "manufacturing")
FIRST = ("Ana", "Ben", "Chloe", "Dev", "Elena", "Femi", "Grace", "Hugo", "Ines", "Jon", "Kira", "Leo", "Mira", "Noor", "Omar", "Pia")
LAST = ("Adler", "Bose", "Cruz", "Dale", "Eze", "Frost", "Gill", "Hart", "Ito", "Jain", "Khan", "Lund", "Moss", "Nair", "Ortiz", "Park")


@dataclass
class DealTruth:
    """What the generator knows about a deal and the evaluator may check against."""
    account: str
    name: str
    situation: str
    result: str  # won | lost | open
    primary_play: str | None
    play_class: str | None  # good | neutral | trap (closed deals)
    plays_used: list[str] = field(default_factory=list)


@dataclass
class World:
    seed: int
    plays: list[dict]
    deals: list[dict]  # in the demo-seed format (so the product's own loader reads it)
    train: list[DealTruth]
    test: list[DealTruth]

    def seed_document(self) -> dict:
        return {"vendor": "Evalworks (synthetic)", "product": "Revenue analytics",
                "disclosure": "Synthetic evaluation data with planted rules. Not real sales data.",
                "plays": self.plays, "deals": self.deals}

    def write(self, path: Path) -> Path:
        path.write_text(json.dumps(self.seed_document(), indent=1), encoding="utf-8")
        return path


def play_class(situation: Situation, code: str) -> str:
    return "good" if code in situation.good else "trap" if code in situation.trap else "neutral"


def _weights(sit: Situation) -> dict[str, float]:
    """How often the team reaches for each play in this situation."""
    weights: dict[str, float] = {sit.obvious: OBVIOUS_SHARE}
    goods = [c for c in sit.good if c != sit.obvious]
    traps = [c for c in sit.trap if c != sit.obvious]
    neutrals = [c for c in sit.neutral if c != sit.obvious]
    budget = 1 - OBVIOUS_SHARE
    # counter-intuitive: the obvious play is the trap, so goods get 25% and neutrals 30%.
    # control: the obvious play is good, so other goods 10%, traps 15-20%, neutrals the rest.
    shares = {"good": 0.25 if sit.kind == "counterintuitive" else 0.10, "trap": 0.0 if sit.kind == "counterintuitive" else 0.17}
    shares["neutral"] = budget - shares["good"] - shares["trap"]
    for group, codes in (("good", goods), ("trap", traps), ("neutral", neutrals)):
        for code in codes:
            weights[code] = shares[group] / len(codes)
    total = sum(weights.values())
    return {c: w / total for c, w in weights.items()}


def _choose(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _closed_deal(rng: random.Random, index: int, sit: Situation, account: str, plays: list[dict]) -> tuple[dict, DealTruth]:
    primary = _choose(rng, _weights(sit))
    cls = play_class(sit, primary)
    won = rng.random() < P_WIN[cls]
    extras = [c for c in sit.neutral if c != primary and rng.random() < EXTRA_PLAY_RATE][:1]
    used = [primary, *extras]
    unresolved = (not won) and rng.random() < 0.75
    loss_reason = None
    if not won:
        pool = sit.loss_reasons if cls == "trap" or unresolved else ("competitor", "no_decision", "timing", "champion_left")
        loss_reason = rng.choice(pool)
    opened = date(2023, 1, 9) + timedelta(days=rng.randrange(0, 600))
    closed = opened + timedelta(days=rng.randrange(35, 120))
    who, champion, buyer = _person(rng), _person(rng), _person(rng)
    key = f"c{index}"
    objections = [{"type": sit.objection, "text": rng.choice(sit.notes).format(who=who), "status": "unresolved" if unresolved else "addressed",
                   "first_seen_on": (opened + timedelta(days=10)).isoformat(), "evidence": [key]}]
    if rng.random() < 0.30:
        other = rng.choice([o for o in OTHER_OBJECTIONS if o != sit.objection])
        objections.append({"type": other, "text": f"{who} also raised {other.replace('_', ' ')}.",
                           "status": rng.choice(("addressed", "raised")), "first_seen_on": (opened + timedelta(days=14)).isoformat(),
                           "evidence": [key]})
    deal = {
        "account": account, "name": f"{account} - Evalworks", "industry": rng.choice(INDUSTRIES),
        "segment": rng.choice(("smb", "mid_market", "enterprise")), "amount": rng.randrange(40, 400) * 1000,
        "owner": _person(rng), "opened_on": opened.isoformat(), "closed_on": closed.isoformat(),
        "result": "won" if won else "lost", **({"loss_reason": loss_reason} if loss_reason else {}),
        "stakeholders": [{"name": champion, "title": "Operations lead", "stance": "champion", "engaged": True, "economic_buyer": False},
                         {"name": buyer, "title": "VP Finance", "stance": rng.choice(("supporter", "neutral")), "engaged": True,
                          "economic_buyer": True}],
        "interactions": [{"key": key, "kind": "call_note", "date": (opened + timedelta(days=10)).isoformat(), "author": _person(rng),
                          "subject": f"{sit.objection.replace('_', ' ')} discussion", "text": objections[0]["text"]}],
        "signals": {"objections": objections, "competitors": rng.choice(([], [], ["Brightline"], ["Northwind"])),
                    "pricing": {"discount_requested": sit.objection == "pricing" or "PLAY-05" in used, "notes": ""},
                    "promises": [], "plays_used": used},
    }
    return deal, DealTruth(account, deal["name"], sit.name, deal["result"], primary, cls, used)


def _open_deal(rng: random.Random, index: int, sit: Situation, account: str) -> tuple[dict, DealTruth]:
    who, champion, buyer = _person(rng), _person(rng), _person(rng)
    keys = [f"t{index}-{n}" for n in (1, 2, 3)]
    notes = list(sit.notes)
    rng.shuffle(notes)
    ago = rng.randrange(20, 60)
    interactions = [
        {"key": keys[0], "kind": "call_note", "days_ago": ago - 2, "author": _person(rng), "subject": "Discovery call",
         "text": f"Discovery with {champion}. They want to consolidate reporting and are evaluating this quarter."},
        {"key": keys[1], "kind": "email", "days_ago": ago // 2, "author": who, "subject": f"Re: {sit.objection.replace('_', ' ')}",
         "text": notes[0].format(who=who)},
        {"key": keys[2], "kind": "call_note", "days_ago": 4, "author": _person(rng), "subject": "Follow-up",
         "text": f"Follow-up with {champion}. {notes[1].format(who=who)} We have not yet agreed how to answer it."},
    ]
    deal = {
        "account": account, "name": f"{account} - Evalworks", "industry": rng.choice(INDUSTRIES),
        "segment": rng.choice(("smb", "mid_market", "enterprise")), "amount": rng.randrange(40, 400) * 1000, "owner": _person(rng),
        "stage": rng.choice(("evaluation", "proposal")), "opened_days_ago": ago, "result": "open",
        "stakeholders": [{"name": champion, "title": "Operations lead", "stance": "champion", "engaged": True, "economic_buyer": False},
                         {"name": buyer, "title": "VP Finance", "stance": "neutral", "engaged": False, "economic_buyer": True}],
        "interactions": interactions,
        "signals": {"objections": [{"type": sit.objection, "text": notes[0].format(who=who), "status": "unresolved",
                                    "first_seen_days_ago": ago // 2, "evidence": [keys[1], keys[2]]}],
                    "competitors": rng.choice(([], ["Brightline"])), "pricing": {"discount_requested": sit.objection == "pricing", "notes": ""},
                    "promises": [], "plays_used": []},
    }
    return deal, DealTruth(account, deal["name"], sit.name, "open", None, None, [])


def make_world(seed: int, n_train: int = 60, n_test: int = 20) -> World:
    """`n_train` closed deals (the memory) and `n_test` open deals to advise on, split evenly over the five situations."""
    rng = random.Random(seed)
    halcyon = json.loads(HALCYON_SEED.read_text(encoding="utf-8"))
    names = list(SITUATIONS)
    accounts: set[str] = set()

    def new_account() -> str:
        while True:
            account = f"{rng.choice(ADJ)} {rng.choice(NOUN)} {rng.randrange(10, 99)}"
            if account not in accounts:
                accounts.add(account)
                return account

    train_names = [names[i % len(names)] for i in range(n_train)]
    rng.shuffle(train_names)
    test_names = [names[i % len(names)] for i in range(n_test)]
    deals: list[dict] = []
    train: list[DealTruth] = []
    test: list[DealTruth] = []
    for i, name in enumerate(train_names):
        deal, truth = _closed_deal(rng, i, SITUATIONS[name], new_account(), halcyon["plays"])
        deals.append(deal)
        train.append(truth)
    for i, name in enumerate(test_names):
        deal, truth = _open_deal(rng, i, SITUATIONS[name], new_account())
        deals.append(deal)
        test.append(truth)
    # The product's seed loader wants one "demo" and one "closable" open deal; any two will do here.
    for role, deal in zip(("demo", "closable"), [d for d in deals if d["result"] == "open"][:2], strict=False):
        deal["role"] = role
    return World(seed, halcyon["plays"], deals, train, test)
