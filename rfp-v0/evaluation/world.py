"""A synthetic proposal-team world with planted rules, generated from a seed (no model call).

A company has a fact sheet, a library of approved answers imported from past proposals (some won, one lost), and a set of
new questions. Each question is one of eight kinds, and for each the right answer is known by construction:

  exact        one approved answer from a won proposal; nothing else on the topic (control)
  stale        two won answers, a 2021 one with an outdated value and a 2025 one with the current value
  context      two won answers from the same year for different industries; the right one matches the new client's industry
  reviewed     two won answers; reviewers kept accepting one and kept rewriting the other because its value was wrong
  lost_trap    a good answer from a won proposal and a wrong one from a proposal lost on technical fit
  conflict     an approved answer says one value but the current fact sheet says another (the fact sheet must win)
  fact_only    only the fact sheet knows the value (control: memory has nothing relevant to offer)
  unanswerable nobody knows the value: the right behaviour is to hand it to an expert, not to invent a number

Because the rules are planted, every retrieval and every drafted answer is marked right or wrong by code. No model judges
another model. The values are numbers and words drawn per seed, so nothing can be memorised.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

KINDS = ("exact", "stale", "context", "reviewed", "lost_trap", "conflict", "fact_only", "unanswerable")
KIND_COUNTS = {"exact": 3, "stale": 4, "context": 3, "reviewed": 4, "lost_trap": 2, "conflict": 3, "fact_only": 2, "unanswerable": 3}
LIBRARY_KINDS = ("exact", "stale", "context", "reviewed", "lost_trap")  # the right answer lives in the library
WORD_LIMIT = 80


@dataclass(frozen=True)
class Topic:
    key: str
    question: str  # how past proposals phrased it
    asks: tuple[str, str]  # how a new RFP phrases it
    answer: str  # {v} is the value
    fact: str
    values: tuple[str, ...]


TOPICS: tuple[Topic, ...] = (
    Topic("uptime", "What monthly uptime percentage do you commit to for the production service?",
          ("What is the guaranteed monthly uptime for production use?", "State the uptime percentage committed for the production service each month."),
          "We commit to {v} monthly uptime for the production service, measured per calendar month.",
          "The production uptime commitment is {v} per calendar month.", ("99.5%", "99.9%", "99.95%", "99.99%")),
    Topic("retention", "How long do you retain customer data after contract termination?",
          ("After the contract ends, how long is customer data retained?", "State the customer data retention period following termination."),
          "Customer data is retained for {v} after contract termination and then securely deleted.",
          "Customer data is retained for {v} after contract termination.", ("30 days", "60 days", "90 days", "180 days")),
    Topic("backup", "How often are customer database backups taken?",
          ("What is the frequency of database backups for customer data?", "State how frequently backups of the customer database run."),
          "Customer database backups run {v}.", "Customer database backups run {v}.", ("hourly", "every 6 hours", "daily", "weekly")),
    Topic("incident", "Within what time do you notify customers of a confirmed security incident?",
          ("How quickly are customers notified once a security incident is confirmed?", "State the notification window for confirmed security incidents."),
          "We notify affected customers within {v} of confirming a security incident.",
          "Affected customers are notified within {v} of a confirmed security incident.", ("4 hours", "24 hours", "48 hours", "72 hours")),
    Topic("pentest", "How often do you commission independent penetration tests?",
          ("What is the cadence of third-party penetration testing?", "State how frequently independent penetration tests are commissioned."),
          "Independent penetration tests are commissioned {v}.", "Independent penetration tests are commissioned {v}.",
          ("quarterly", "twice a year", "annually")),
    Topic("rpo", "What recovery point objective applies to the production database?",
          ("State the RPO for the production database.", "What is the recovery point objective for production data?"),
          "The recovery point objective for the production database is {v}.",
          "The recovery point objective for the production database is {v}.", ("5 minutes", "15 minutes", "1 hour", "4 hours")),
    Topic("rto", "What recovery time objective applies after a regional outage?",
          ("State the RTO following a regional outage.", "How quickly is service restored after a regional outage under your recovery time objective?"),
          "The recovery time objective after a regional outage is {v}.",
          "The recovery time objective after a regional outage is {v}.", ("30 minutes", "2 hours", "8 hours", "24 hours")),
    Topic("support", "What support hours do you provide for priority-one issues?",
          ("When is support available for priority-one tickets?", "State the support coverage hours for priority-one issues."),
          "Priority-one issues receive support coverage of {v}.", "Priority-one issues receive support coverage of {v}.",
          ("8x5", "12x5", "24x7")),
    Topic("cipher", "Which encryption standard protects customer data at rest?",
          ("What encryption is used for data at rest?", "State the encryption standard applied to stored customer data."),
          "Customer data at rest is encrypted with {v}.", "Customer data at rest is encrypted with {v}.", ("AES-128", "AES-256", "ChaCha20")),
    Topic("soc2", "Which SOC 2 report do you currently hold?",
          ("State the type of SOC 2 attestation your company holds.", "What SOC 2 report type is currently issued?"),
          "We currently hold a SOC 2 {v} report.", "The company currently holds a SOC 2 {v} report.", ("Type I", "Type II")),
    Topic("auditlog", "How long are administrative audit logs retained?",
          ("State the retention period for admin audit logs.", "For how long do you keep audit logs of administrator actions?"),
          "Administrator audit logs are retained for {v}.", "Administrator audit logs are retained for {v}.",
          ("90 days", "1 year", "2 years", "7 years")),
    Topic("onboarding", "How long does customer onboarding typically take?",
          ("What is the typical onboarding duration for a new customer?", "State the usual time to onboard a new customer."),
          "Onboarding a new customer typically takes {v}.", "Onboarding a new customer typically takes {v}.",
          ("2 weeks", "4 weeks", "6 weeks", "10 weeks")),
    Topic("residency", "In which region is customer data hosted?",
          ("State the hosting region for customer data.", "Where is customer data hosted geographically?"),
          "Customer data is hosted in the {v} region.", "Customer data is hosted in the {v} region.",
          ("EU-West", "US-East", "APAC-South", "UK-South")),
    Topic("mfa", "Which multi-factor authentication method is enforced for administrators?",
          ("State the MFA method required of administrator accounts.", "What multi-factor method must administrators use?"),
          "Administrators must use {v} for multi-factor authentication.",
          "Administrators must use {v} for multi-factor authentication.", ("a hardware security key", "an authenticator app", "SMS codes")),
    Topic("subproc", "How many sub-processors handle customer personal data?",
          ("State the number of sub-processors with access to personal data.", "How many third-party sub-processors process customer personal data?"),
          "{v} sub-processors handle customer personal data.", "{v} sub-processors handle customer personal data.",
          ("Six", "Nine", "Fourteen", "Twenty-one")),
    Topic("seats", "What is the maximum number of named users per tenant?",
          ("State the named-user limit for a single tenant.", "How many named users can one tenant have?"),
          "A single tenant supports up to {v} named users.", "A single tenant supports up to {v} named users.",
          ("500", "2,000", "10,000", "50,000")),
    Topic("apirate", "What API rate limit applies to the standard plan?",
          ("State the standard plan's API rate limit.", "How many API requests per minute are allowed on the standard plan?"),
          "The standard plan allows {v} API requests per minute.", "The standard plan allows {v} API requests per minute.",
          ("60", "300", "1,200", "6,000")),
    Topic("patching", "How quickly are critical vulnerabilities patched?",
          ("State the patching timeline for critical vulnerabilities.", "Within what time are critical security vulnerabilities remediated?"),
          "Critical vulnerabilities are patched within {v}.", "Critical vulnerabilities are patched within {v}.",
          ("36 hours", "5 days", "14 days", "45 days")),
    Topic("drtest", "How frequently is the disaster recovery plan tested?",
          ("State how often the disaster recovery plan is exercised.", "What testing frequency applies to the disaster recovery plan?"),
          "The disaster recovery plan is tested {v}.", "The disaster recovery plan is tested {v}.",
          ("every quarter", "every six months", "every year")),
    Topic("credits", "What service credit applies if the uptime commitment is missed?",
          ("State the service credit for a missed availability target.", "What credit is given when the service level agreement is breached?"),
          "A missed commitment earns a service credit of {v} of the monthly fee.",
          "A missed commitment earns a service credit of {v} of the monthly fee.", ("5%", "10%", "25%", "50%")),
    Topic("password", "What minimum password length does the platform enforce?",
          ("State the enforced minimum password length.", "What is the minimum password length policy?"),
          "The platform enforces a minimum password length of {v}.", "The platform enforces a minimum password length of {v}.",
          ("8 characters", "12 characters", "14 characters", "16 characters")),
    Topic("training", "How often do staff complete security awareness training?",
          ("State the security awareness training cadence for employees.", "How frequently must employees complete security training?"),
          "All staff complete security awareness training {v}.", "All staff complete security awareness training {v}.",
          ("each quarter", "twice yearly", "once a year")),
    Topic("export", "In which formats can customers export their data?",
          ("State the data export formats available to customers.", "What file formats are supported for customer data export?"),
          "Customers can export data as {v}.", "Customers can export data as {v}.", ("CSV only", "CSV and JSON", "CSV, JSON and Parquet")),
    Topic("datacentres", "How many data centres host the production service?",
          ("State the number of data centres used for production.", "How many production data centres do you operate?"),
          "The production service runs across {v} data centres.", "The production service runs across {v} data centres.",
          ("two", "three", "four", "five")),
)

# Past answers unrelated to any question: the library is not only the topics that get asked about.
FILLER: tuple[tuple[str, str], ...] = (
    ("Describe your company background and history.", "The company was founded in 2012 and builds analytics software for operations teams."),
    ("What is your approach to customer success?", "Each customer has a named success manager and a quarterly business review."),
    ("How do you handle change management for releases?", "Releases follow a documented change process with peer review and staged rollout."),
    ("Describe your software development lifecycle.", "We use trunk-based development with automated tests and mandatory code review."),
    ("What training do you provide to new customers?", "New customers receive live onboarding sessions and a library of recorded tutorials."),
    ("How do you manage third-party vendor risk?", "Vendors are assessed before onboarding and reviewed each year against our security baseline."),
)

GENERAL_FACTS: tuple[str, ...] = (
    "The company was founded in 2012 and is headquartered in Leeds.",
    "The company employs 240 people across engineering, support and sales.",
    "The company holds ISO 27001 certification for its information security management system.",
    "The company serves customers in 14 countries.",
    "Customer support is delivered from offices in Leeds and Austin.",
)

PROJECTS = {
    "neutral": ("Meridian Council", "Public Sector"),
    "health": ("Calder Health Trust", "Healthcare"),
    "finance": ("Kingsford Credit", "Finance"),
}

# name -> (client, industry, submitted_on, result, loss_reason)
PROPOSALS = {
    "base": ("Harrow Telecom", "Telecom", date(2024, 11, 12), "won", None),
    "old1": ("Delta Mills", "Manufacturing", date(2021, 3, 15), "won", None),
    "new1": ("Cygnus Freight", "Logistics", date(2025, 2, 20), "won", None),
    "new2": ("Brightway Foods", "Food", date(2025, 7, 1), "won", None),
    "old2": ("Harbor Metals", "Mining", date(2021, 9, 9), "won", None),
    "ctxfin": ("Aster Capital", "Finance", date(2024, 5, 14), "won", None),
    "ctxhea": ("Kestrel Clinics", "Healthcare", date(2024, 5, 14), "won", None),
    "ctxhea2": ("Marlowe Care", "Healthcare", date(2024, 5, 14), "won", None),
    "ctxfin2": ("Aster Capital Partners", "Finance", date(2024, 5, 14), "won", None),
    "rev1": ("Pinewood Energy", "Energy", date(2024, 2, 1), "won", None),
    "rev2": ("Solace Power", "Energy", date(2024, 3, 1), "won", None),
    "lost": ("Orion Retail", "Retail", date(2024, 6, 20), "lost", "technical fit"),
    "wonref": ("Fenwick Stores", "Retail", date(2024, 6, 25), "won", None),
}

# What the review history looks like: (which answer, action, reason tags). The wrong answer keeps being rewritten.
REVIEW_PLAN: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("bad", "rewritten", ("incorrect",)),
    ("good", "accepted", ()),
    ("bad", "rewritten", ("incorrect",)),
    ("good", "accepted", ()),
    ("bad", "rejected", ("outdated",)),
)


@dataclass
class Question:
    id: str
    kind: str
    topic: str
    text: str
    project: str  # key in PROJECTS
    gold_value: str | None  # None for an unanswerable question
    forbidden: list[str]  # values that must not appear in a correct answer
    gold_ref: tuple[str, str] | None = None  # (proposal, topic) of the right library answer
    bad_ref: tuple[str, str] | None = None  # (proposal, topic) of the wrong library answer
    gold_fact: str | None = None  # fact id when the fact sheet holds the right value


@dataclass
class Fact:
    id: str
    topic: str
    statement: str
    valid_from: str | None = None


@dataclass
class World:
    seed: int
    company: str
    proposals: list[str]  # import order
    pairs: dict[str, list[tuple[str, str, str]]]  # proposal -> [(topic, question, answer)]
    facts: list[Fact]
    questions: list[Question]

    def history(self, rounds: int | None = None) -> list[tuple[tuple[str, str], str, tuple[str, ...]]]:
        """The planted review events in order: (answer ref, action, reason tags). A round is one step of REVIEW_PLAN for
        every reviewed question; `rounds` keeps only the first few (None keeps them all)."""
        out = []
        for step in range(len(REVIEW_PLAN) if rounds is None else min(rounds, len(REVIEW_PLAN))):
            for q in self.questions:
                if q.kind != "reviewed":
                    continue
                who, action, tags = REVIEW_PLAN[step]
                out.append((q.gold_ref if who == "good" else q.bad_ref, action, tags))
        return out

    def write(self, path: Path) -> Path:
        Path(path).write_text(json.dumps(asdict(self), indent=1, default=str), encoding="utf-8")
        return Path(path)


def _flat(topic: Topic, value: str) -> str:
    return topic.answer.format(v=value)


def make_world(seed: int) -> World:
    rng = random.Random(seed * 7919 + 13)
    topics = list(TOPICS)
    rng.shuffle(topics)
    by_kind: dict[str, list[Topic]] = {}
    for kind in KINDS:
        by_kind[kind] = [topics.pop() for _ in range(KIND_COUNTS[kind])]

    pairs: dict[str, list[tuple[str, str, str]]] = {key: [] for key in PROPOSALS}
    questions: list[Question] = []
    facts = [Fact(f"FACT-{n:03d}", "company", text) for n, text in enumerate(GENERAL_FACTS, start=1)]
    fact_n = 100

    def put(proposal: str, topic: Topic, value: str) -> tuple[str, str]:
        pairs[proposal].append((topic.key, topic.question, _flat(topic, value)))
        return proposal, topic.key

    def ask(kind: str, topic: Topic, **kw) -> None:
        questions.append(Question(id=f"{kind}-{topic.key}", kind=kind, topic=topic.key, text=rng.choice(topic.asks),
                                  project=kw.pop("project", "neutral"), **kw))

    for topic in by_kind["exact"]:
        value = rng.choice(topic.values)
        ref = put("base", topic, value)
        ask("exact", topic, gold_value=value, forbidden=[], gold_ref=ref)

    for topic in by_kind["stale"]:
        old, new = rng.sample(topic.values, 2)
        old_ref = put(rng.choice(["old1", "old2"]), topic, old)
        new_ref = put(rng.choice(["new1", "new2"]), topic, new)
        ask("stale", topic, gold_value=new, forbidden=[old], gold_ref=new_ref, bad_ref=old_ref)

    for topic in by_kind["context"]:
        v_fin, v_hea = rng.sample(topic.values, 2)
        # two proposals per industry, imported fin, hea, hea, fin: which of the pair has the lower id is a coin flip per question
        fin_ref, hea_ref = put(rng.choice(["ctxfin", "ctxfin2"]), topic, v_fin), put(rng.choice(["ctxhea", "ctxhea2"]), topic, v_hea)
        if rng.random() < 0.5:
            ask("context", topic, project="finance", gold_value=v_fin, forbidden=[v_hea], gold_ref=fin_ref, bad_ref=hea_ref)
        else:
            ask("context", topic, project="health", gold_value=v_hea, forbidden=[v_fin], gold_ref=hea_ref, bad_ref=fin_ref)

    for topic in by_kind["reviewed"]:
        good, bad = rng.sample(topic.values, 2)
        good_prop, bad_prop = rng.sample(["rev1", "rev2"], 2)
        good_ref, bad_ref = put(good_prop, topic, good), put(bad_prop, topic, bad)
        ask("reviewed", topic, gold_value=good, forbidden=[bad], gold_ref=good_ref, bad_ref=bad_ref)

    for topic in by_kind["lost_trap"]:
        good, bad = rng.sample(topic.values, 2)
        good_ref, bad_ref = put("wonref", topic, good), put("lost", topic, bad)
        ask("lost_trap", topic, gold_value=good, forbidden=[bad], gold_ref=good_ref, bad_ref=bad_ref)

    for topic in by_kind["conflict"]:
        old, new = rng.sample(topic.values, 2)
        bad_ref = put("base", topic, old)
        fact_n += 1
        facts.append(Fact(f"FACT-{fact_n}", topic.key, topic.fact.format(v=new), valid_from="2026-01-01"))
        ask("conflict", topic, gold_value=new, forbidden=[old], bad_ref=bad_ref, gold_fact=f"FACT-{fact_n}")

    for topic in by_kind["fact_only"]:
        value = rng.choice(topic.values)
        fact_n += 1
        facts.append(Fact(f"FACT-{fact_n}", topic.key, topic.fact.format(v=value)))
        ask("fact_only", topic, gold_value=value, forbidden=[], gold_fact=f"FACT-{fact_n}")

    for topic in by_kind["unanswerable"]:
        ask("unanswerable", topic, gold_value=None, forbidden=[])

    for question, answer in FILLER:
        pairs["base"].append(("filler", question, answer))

    order = list(PROPOSALS)
    # Which of two near-identical imports comes first decides which one a tie-breaking search returns first,
    # so it is drawn per world instead of being fixed in favour of or against any system.
    for a, b in (("lost", "wonref"),):
        if rng.random() < 0.5:
            i, j = order.index(a), order.index(b)
            order[i], order[j] = order[j], order[i]
    order = [key for key in order if pairs[key]]
    rng.shuffle(questions)
    return World(seed=seed, company="Northwind Analytics", proposals=order, pairs=pairs, facts=facts, questions=questions)
