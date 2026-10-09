"""Understand a chat message without a model: which flow it starts, which deal it names, how a deal ended.

Everything here is pure (no I/O, no clock), so it is table-tested. The wording it understands is
deliberately plain; when it is unsure it returns a weaker intent and the engine asks a question
instead of guessing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from .router import load_registry, route

LOSS_REASONS = (
    "competitor", "price", "no_decision", "champion_left", "security_compliance", "feature_gap", "timing",
    "unresolved_objection",
)

# Plain words for each loss reason: (what the person might say, short button label).
LOSS_REASON_WORDS: dict[str, tuple[str, str]] = {
    "competitor": ("they went with a competitor", "Went with a competitor"),
    "price": ("price (too expensive)", "Price"),
    "no_decision": ("no decision (they did nothing)", "No decision"),
    "champion_left": ("our champion left", "Champion left"),
    "security_compliance": ("the security or compliance review failed", "Security review"),
    "feature_gap": ("a missing feature", "Missing feature"),
    "timing": ("bad timing", "Bad timing"),
    "unresolved_objection": ("an objection that never got resolved", "Unresolved objection"),
}

# Checked in this order; a message that matches more than one reason is treated as unclear.
_REASON_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (code, re.compile(pattern, re.I))
    for code, pattern in [
        ("champion_left", r"champion\s+(?:has\s+)?(?:left|quit|resigned|departed|moved on|was laid off|got laid off|is gone)"
                          r"|(?:sponsor|contact|buyer)\s+(?:has\s+)?(?:left|quit|resigned)|left the company|lost (?:our|the) champion"),
        ("security_compliance", r"securit|complian|soc ?2|iso ?27001|infosec|pen ?test|gdpr|hipaa|audit|data residency|vendor risk"),
        ("unresolved_objection", r"unresolved|never (?:got |was |were )?(?:resolved|addressed|answered|fixed|solved)"
                                 r"|(?:wasn'?t|was not|weren'?t|didn'?t|did not) (?:get )?(?:resolved|addressed|answered|fixed)"
                                 r"|objections?\b|open concern|blocker|(?:could not|couldn'?t) (?:overcome|resolve|address)"),
        ("feature_gap", r"feature\s*gap|missing (?:a |the )?(?:key |critical )?(?:feature|capabilit|integration|functionality)"
                        r"|(?:lack(?:ed|s|ing)?|without) (?:\w+ ){0,3}?(?:feature|capabilit|integration|functionality)"
                        r"|(?:doesn'?t|does not|didn'?t|did not|can'?t|cannot) (?:have|support|do|integrate)"
                        r"|no (?:integration|support for)|product gap|functionality gap"),
        ("no_decision", r"no[\s-]+decision|status quo|did nothing|do nothing|decided not to (?:buy|proceed|move forward)"
                        r"|went (?:dark|silent|cold)|ghosted"),
        ("timing", r"\btiming\b|wrong time|not the right time|bad time|not (?:right )?now|postpone|pushed (?:out|back)"
                   r"|delayed|next (?:year|quarter|fiscal)|freeze|froze"),
        ("price", r"\bpric(?:e|es|ed|ing)\b|expensive|too (?:costly|much|high)|\bcost\b|budget|cheaper|discount|afford"),
        ("competitor", r"competitor|competing|rival|went with|chose|picked|selected|signed with|going with|went elsewhere"
                       r"|another vendor|someone else|incumbent"),
    ]
]
_WON = re.compile(r"\b(?:won|win|closed[\s-]*won)\b", re.I)
_LOST = re.compile(r"\b(?:lost|lose|loss|closed[\s-]*lost)\b", re.I)

CONFIRM = re.compile(
    r"^\W*(?:(?:yes|yep|yeah|yup|sure|ok|okay|y|go ahead|go|do it|proceed|confirm|please do|sounds good|please|run it|start)\b[\s,!.]*)+$",
    re.I,
)
CANCEL = re.compile(
    r"^\W*(?:(?:no|nope|nah|cancel|stop|not now|never mind|nevermind|abort|don'?t|skip|forget it)\b[\s,!.]*)+$", re.I
)
_RFP_NOUN = re.compile(r"\b(?:rfp|rfi|rfq|itt|tender|questionnaire|proposal|bid)\b", re.I)
_RFP_VERB = re.compile(r"\b(?:draft|answer|respond|response|complete|fill|work on|handle|process|start|write)\b", re.I)
_RFP_FILE_HINT = re.compile(r"rfp|rfi|rfq|itt|tender|questionnaire|security|vendor|due.?diligence|caiq|proposal", re.I)
_DEAL_FILE_HINT = re.compile(r"email|call|notes?|transcript|thread|meeting|deal|crm|minutes", re.I)
_PRONOUN_SUBJECTS = {
    "it", "this", "that", "this deal", "that deal", "the deal", "them", "this one", "that one", "deal", "account",
    "opportunity", "one",
}

_BRIEF_PATTERNS = [
    re.compile(r"^\W*(?:please\s+)?(?:can you\s+|could you\s+)?(?:brief|prep|prepare|catch)\s+me\s+(?:on|for|about|before|up on)\s+(?P<s>.+)$", re.I),
    re.compile(r"^\W*(?:please\s+)?(?:can you\s+|could you\s+)?(?:brief|prep|prepare)\s+me\W*$", re.I),
    re.compile(r"^\W*what(?:'s| is)\s+(?:going on|happening|the latest|the status|new)\s+(?:with|on|for|in)\s+(?P<s>.+)$", re.I),
    re.compile(r"^\W*(?:give me\s+)?(?:a\s+)?(?:call prep|briefing|brief|summary)\s+(?:of|on|for|about)\s+(?P<s>.+)$", re.I),
    re.compile(r"^\W*(?:status|update)\s+(?:of|on|for)\s+(?P<s>.+)$", re.I),
]
_OUTCOME_PATTERNS = [
    re.compile(r"\b(?:we|i)\s+(?:just\s+|have\s+|finally\s+)*(?P<r>won|lost)\s+(?P<s>.+)$", re.I),
    re.compile(r"^\W*(?P<s>.+?)\s+(?:deal\s+)?(?:was|got|is|has been|has)\s+(?:closed[\s-]*)?(?P<r>won|lost)\b", re.I),
    re.compile(r"\b(?:mark|close|set|record|log)\s+(?P<s>.+?)\s+as\s+(?:closed[\s-]*)?(?P<r>won|lost)\b", re.I),
    re.compile(r"^\W*(?P<s>.+?)\s+closed[\s-]*(?P<r>won|lost)\b", re.I),
]
_SUBJECT_END = re.compile(r"\s+(?:because|due to|since|as|to|after|when|on|over|for|but|and)\s+|[,;.!?]|\s[-–—]\s")
_NEW_DEAL = re.compile(
    r"\b(?:new|create(?:\s+a)?|add(?:\s+a)?|start(?:\s+a)?|open(?:\s+a)?|set up(?:\s+a)?)\s+(?:new\s+)?deal\b(?P<rest>.*)$", re.I
)
_ADD_TO_DEAL = re.compile(
    r"\b(?:add|attach|upload|log|save|put|file|here(?:'s| are| is))\b.*?\b(?:to|on|for|into|under)\s+(?P<s>.+?)[.!?]*$", re.I
)
_NOTE = re.compile(
    r"^\W*(?:please\s+)?(?:add|log|save|record)\s+(?:a\s+)?(?P<k>call\s+note|note|email|meeting\s+note)\s+(?:to|on|for)\s+(?:the\s+)?"
    r"(?P<s>[^:—–-]+?)(?:\s+deal)?\s*[:—–-]\s*(?P<t>.+)$",
    re.I | re.S,
)
_FOLLOWUP = [
    re.compile(
        r"^\W*(?:please\s+)?(?:can you\s+|could you\s+)?(?:draft|write|prepare|prep|make|create|compose)\s+(?:me\s+)?"
        r"(?:a\s+|an\s+|the\s+|my\s+)?(?P<k>call\s+agenda|agenda|follow[\s-]?up(?:\s+email)?|email)\b"
        r"(?:\s+(?:for|to|on|about|with)\s+(?P<s>.+))?$", re.I),
    re.compile(r"^\W*(?:what\s+should\s+i\s+send|follow[\s-]?up)\s+(?:to|with|on|for)\s+(?P<s>.+)$", re.I),
]
_QUESTION = re.compile(
    r"^\W*(?:what|who|whom|when|where|why|how|which|did|does|do|is|are|was|were|has|have|had|can|could|should|will|any|tell me|"
    r"show me|explain)\b", re.I)
_WS_WORD = re.compile(r"\bworkspaces?\b", re.I)
_WS_LIST = re.compile(
    r"^\W*(?:(?:list|show|see|what are)\s+(?:me\s+)?(?:the\s+|my\s+|all\s+|our\s+)*workspaces?|(?:which|what)\s+workspaces?\b.*|workspaces?)\W*$", re.I)
_WS_USE = re.compile(
    r"^\W*(?:please\s+)?(?:use|switch\s+to|open|work\s+in|change\s+to|go\s+to|select)\s+(?:the\s+)?(?:workspace\s+)?(?P<s>.+?)"
    r"(?:\s+workspace)?(?:\s+for\s+(?P<a>deals?|rfps?|proposals?|questionnaires?))?\W*$", re.I)
_SAVE_WS = re.compile(
    r"^\W*(?:please\s+)?(?:remember|save|keep|make)\b.*\bworkspaces?\b(?:.*\b(?:as|called|named)\s+(?P<n>[^.!?]+?)|\s+(?:the\s+)?default)?\W*$", re.I)
_FORGET_WS = re.compile(r"^\W*(?:please\s+)?(?:forget|clear|reset|remove)\b.*\b(?:saved|default|remembered)\b.*\bworkspaces?\b.*$", re.I)
# Questions about the whole company's records rather than one deal. Keyword cues only, no model.
_EVERYTHING = re.compile(
    r"\b(?:across everything|everything we have|whole company|company[- ]wide|all (?:of )?our (?:data|knowledge|memory|records))\b", re.I)
_PORTFOLIO_CUE = re.compile(
    r"\b(?:across|overall|portfolio|pipeline|how many|most (?:common|often)|trend|patterns?|win rate|loss reasons?|"
    r"why (?:do|did|are|have|does) (?:we|our)|why we (?:lose|lost|win|won)|(?:all|our|my|the|which) (?:open |closed |lost |won )?deals|"
    r"every deal)\b", re.I)
_LIBRARY_CUE = re.compile(
    r"\b(?:soc ?2|iso ?27001|certif\w*|encrypt\w*|penetration|pen[- ]?tests?|gdpr|hipaa|polic(?:y|ies)|sla|uptime|"
    r"disaster recovery|backups?|sub-?processors?|insurance|past answers?|our answers?|library|fact sheet|"
    r"(?:have|did) we (?:ever )?(?:said|told|answered|written|claimed)|(?:do|does) (?:we|our company) (?:hold|offer|support)|"
    r"security (?:program|posture))\b", re.I)
_LIST_DEALS = [
    re.compile(r"^\W*(?:what|which)\s+(?:are|were|is)\s+(?:those|these|the|all(?:\s+(?:of\s+)?(?:the|our|my))?|our|my)\s+(?P<k>(?:open|closed|won|lost|active)\s+)?(?:deals?|opportunities)\b", re.I),
    re.compile(r"^\W*(?:please\s+)?(?:list|show|display|give me)(?:\s+me)?(?:\s+all)?(?:\s+(?:of\s+)?(?:the|our|my))?\s+(?P<k>(?:open|closed|won|lost|active)\s+)?(?:deals?|opportunities)\b", re.I),
    re.compile(r"^\W*(?:which|what)\s+(?P<k>(?:open|closed|won|lost)\s+)?deals\s+(?:do\s+(?:we|you)\s+have|are\s+(?:there|open|we\s+working))", re.I),
    re.compile(r"^\W*(?:do|does)\s+(?:you|we)\s+have\s+(?:(?:any|the|all)\s+)?(?P<k>(?:open|closed|won|lost)\s+)?deals?\b", re.I),
]
_COUNT = re.compile(r"^\W*how many\s+(?:(?P<k>open|closed|won|lost|active|live)\s+)?(?:deals?|opportunities)\b", re.I)
_ACCEPT_ALL = re.compile(r"\b(?:accept|approve)\s+(?:all|every|the grounded|everything)\b", re.I)
_EXPORT = re.compile(r"\b(?:export|download)\b", re.I)
_FORMAT = re.compile(r"\b(?P<f>word|docx|excel|xlsx|spreadsheet)\b", re.I)

_STOPWORDS = {
    "the", "a", "an", "of", "for", "on", "and", "with", "deal", "deals", "account", "opportunity", "inc", "llc", "ltd",
    "corp", "co", "company", "brief", "briefing", "me", "prep", "prepare", "about", "my", "our", "call", "meeting",
    "please", "what", "whats", "is", "going", "status", "new", "to", "in", "at", "up", "catch", "us", "we", "i", "it",
    "this", "that", "was", "were", "won", "lost", "because", "how", "any", "latest", "update", "tell", "show",
}


@dataclass
class Intent:
    kind: str  # see detect()
    agent: str | None = None
    subject: str | None = None
    result: str | None = None
    loss_reason: str | None = None
    target: str | None = None  # deal | rfp | None (outcomes)
    fmt: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    text: str = ""


@dataclass
class Candidate:
    deal: dict[str, Any]
    score: float
    matched: list[str]


# -- outcome -------------------------------------------------------------------------------------


def reason_matches(text: str) -> list[str]:
    """Every loss-reason code the text could mean, in precedence order."""
    return [code for code, pattern in _REASON_PATTERNS if pattern.search(text or "")]


def parse_reason(text: str) -> str | None:
    """The loss reason, only when exactly one fits; None when missing or unclear."""
    found = reason_matches(text)
    return found[0] if len(found) == 1 else None


def parse_outcome(text: str) -> tuple[str | None, str | None]:
    """(result, loss_reason). result is 'won', 'lost' or None; the reason is only set for a loss."""
    text = text or ""
    won, lost = _WON.search(text), _LOST.search(text)
    if won and lost:
        result = "won" if won.start() < lost.start() else "lost"
    else:
        result = "won" if won else "lost" if lost else None
    if result == "lost":
        return result, parse_reason(text)
    return result, None


def free_reason(text: str) -> str | None:
    """The free-text reason after 'because / due to / since' (used for RFP outcomes)."""
    m = re.search(r"\b(?:because(?:\s+of)?|due to|since|as)\s+(?P<r>.+)$", text or "", re.I | re.S)
    return m.group("r").strip(" .!") if m else None


# -- deal matching ---------------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOPWORDS}


def _token_hit(token: str, pool: set[str]) -> bool:
    if token in pool:
        return True
    if len(token) < 5:
        return False
    return any(len(c) >= 5 and SequenceMatcher(None, token, c).ratio() >= 0.86 for c in pool)  # typos, plurals


def match_deal(text: str, deals: list[dict[str, Any]]) -> list[Candidate]:
    """Deals the text refers to, best first. A D-code is an exact match; otherwise token overlap
    with the deal name and account (case-insensitive, small typos allowed)."""
    code = re.search(r"\bd[-\s]?(\d{1,4})\b", text or "", re.I)
    if code:
        wanted = int(code.group(1))
        for deal in deals:
            m = re.search(r"(\d+)", str(deal.get("code") or ""))
            if m and int(m.group(1)) == wanted:
                return [Candidate(deal, 1.0, [str(deal.get("code"))])]
    query = _tokens(text)
    if not query:
        return []
    out: list[Candidate] = []
    for deal in deals:
        name_t, account_t = _tokens(deal.get("name", "")), _tokens(deal.get("account", ""))
        pool = name_t | account_t
        if not pool:
            continue
        hits = {q for q in query if _token_hit(q, pool)}
        if not hits:
            continue

        def cover(field_tokens: set[str]) -> float:
            if not field_tokens:
                return 0.0
            return sum(1 for t in field_tokens if _token_hit(t, query)) / len(field_tokens)

        # What the user typed matters most: "Juniper" must find "Juniper & Vale Stores" even though it names one word of three.
        score = 0.6 * (len(hits) / len(query)) + 0.25 * max(cover(name_t), cover(account_t)) + 0.15 * cover(pool)
        if score >= 0.45:
            out.append(Candidate(deal, round(score, 3), sorted(hits)))
    return sorted(out, key=lambda c: -c.score)


def mentioned_deals(text: str, deals: list[dict[str, Any]]) -> list[Candidate]:
    """Deals a whole sentence names ("what did Dana say about SSO on Cedarline?"): a D-code, or the first
    distinctive word of a deal's account or name appearing in the text (small typos allowed)."""
    coded = match_deal(text, deals) if re.search(r"\bd[-\s]?\d{1,4}\b", text or "", re.I) else []
    if coded and coded[0].score >= 0.999:
        return coded[:1]
    query = _tokens(text)
    out: list[Candidate] = []
    for deal in deals:
        heads = []
        for field_ in (deal.get("account", ""), deal.get("name", "")):
            words = re.findall(r"[a-z0-9]+", (field_ or "").lower())
            words = [w for w in words if w not in _STOPWORDS]
            if words:
                heads.append(words[0])
        hit = [h for h in dict.fromkeys(heads) if _token_hit(h, query)]
        if hit:
            out.append(Candidate(deal, 0.9, hit))
    return out


def resolve_candidates(candidates: list[Candidate]) -> tuple[str, list[Candidate]]:
    """('one', [best]) when there is a clear pick, ('ask', top few) when ambiguous, ('none', [])."""
    if not candidates:
        return "none", []
    if len(candidates) == 1 or candidates[0].score >= 0.999:
        return "one", candidates[:1]
    if candidates[0].score - candidates[1].score >= 0.2:
        return "one", candidates[:1]
    return "ask", candidates[:4]


def pick_candidate(text: str, candidates: list[Candidate]) -> int | None:
    """Which of the offered candidates a short reply means ('2', 'the second one', or part of a name)."""
    t = (text or "").strip().lower()
    ordinals = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
    m = re.fullmatch(r"(?:number\s+|option\s+|#)?(\d)\W*", t)
    if m and 1 <= int(m.group(1)) <= len(candidates):
        return int(m.group(1)) - 1
    for word, idx in ordinals.items():
        if re.search(rf"\b{word}\b", t) and idx <= len(candidates):
            return idx - 1
    matches = match_deal(text, [c.deal for c in candidates])
    if matches and (len(matches) == 1 or matches[0].score - matches[1].score >= 0.2):
        for i, c in enumerate(candidates):
            if c.deal.get("id") == matches[0].deal.get("id"):
                return i
    return None


# -- slots -------------------------------------------------------------------------------------------


def _clean_subject(raw: str | None) -> str | None:
    s = (raw or "").strip(" \t.,;:!?\"'")
    s = re.sub(r"^(?:(?:the|my|our|that|this)\s+)+", "", s, flags=re.I)
    s = re.sub(r"^(?:call|meeting|demo)\s+(?:with|for|on)\s+", "", s, flags=re.I)
    for _ in range(3):
        s = re.sub(r"\s+(?:deal|account|opportunity|call|meeting|please|today|tomorrow|again)$", "", s, flags=re.I).strip()
    s = re.sub(r"^(?:the|my|our)\s+", "", s, flags=re.I)
    if not s or s.lower() in _PRONOUN_SUBJECTS:
        return None
    return s


def _amount(text: str) -> tuple[int | None, str]:
    m = re.search(r"(?:worth\s+|for\s+)?\$\s?(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<u>[kKmM])?\b|\bworth\s+(?P<n2>\d[\d,]*(?:\.\d+)?)\s*(?P<u2>[kKmM])?\b", text)
    if not m:
        return None, text
    number = float((m.group("n") or m.group("n2")).replace(",", ""))
    unit = (m.group("u") or m.group("u2") or "").lower()
    number *= {"k": 1_000, "m": 1_000_000}.get(unit, 1)
    return int(number), (text[: m.start()] + " " + text[m.end():]).strip()


_DEAL_DETAILS = re.compile(
    r"^\W*(?P<s>.+?)\s+(?:is|are)\s+(?:an?\s+|the\s+)?(?P<d>[\w\s,&-]+?\b(?:company|business|firm|organi[sz]ation|customer|"
    r"account|startup|enterprise))\W*$", re.I,
)
_READ_DEAL = re.compile(
    r"^\W*(?:please |now |can you |could you )?(?:read|analy[sz]e|extract (?:the )?signals (?:from|for|of))\s+(?:the )?(?P<s>.+?)\W*$",
    re.I,
)


def parse_deal_slots(rest: str) -> dict[str, Any]:
    """Slots from the words after 'new deal': 'Tidewater at Tidewater Freight' -> name + account."""
    rest = (rest or "").strip(" \t:.-,\"")
    slots: dict[str, Any] = {}
    amount, rest = _amount(rest)
    if amount:
        slots["amount"] = amount
    seg = re.search(r"\b(enterprise|mid[\s-]?market|smb)\b", rest, re.I)
    if seg:
        slots["segment"] = seg.group(1).lower().replace(" ", "_").replace("-", "_")
        rest = (rest[: seg.start()] + " " + rest[seg.end():]).strip()
    rest = re.sub(r"^(?:called|named|name|is)\b[\s:]*", "", rest, flags=re.I).strip(" :,-\"")
    lead = re.match(r"^(?:with|for|at)\s+(?P<a>.+)$", rest, re.I)
    if lead:
        slots["account"] = lead.group("a").strip(" .")
        return _tidy_account(slots)
    parts = re.split(r"\s+(?:at|for|with|@|-|–|—|/)\s+|\s*,\s*", rest, maxsplit=1)
    parts = [p.strip(" .\"") for p in parts if p.strip(" .\"")]
    if parts:
        slots["name"] = parts[0]
    if len(parts) > 1:
        slots["account"] = parts[1]
    return _tidy_account(slots)


def _tidy_account(slots: dict[str, Any]) -> dict[str, Any]:
    """'Hub Test Logistics, logistics' -> account 'Hub Test Logistics' and industry 'logistics'."""
    account = re.sub(r"\s+", " ", str(slots.get("account") or "")).strip(" ,.")
    if "," in account:
        account, _, tail = account.partition(",")
        tail = tail.strip(" ,.")
        if tail and len(tail.split()) <= 3:
            slots["industry"] = tail.lower()
    if account:
        slots["account"] = account.strip()
    return slots


def reply_slots(text: str, have: dict[str, Any]) -> dict[str, Any]:
    """Update partly filled deal slots from a short reply ('Tidewater', 'Tidewater at Tidewater Freight',
    'account: Tidewater Freight')."""
    slots = dict(have)
    labelled = re.match(r"^(?P<k>name|account|company|customer)\s*(?:is|:|=)\s*(?P<v>.+)$", text.strip(), re.I)
    if labelled:
        key = "name" if labelled.group("k").lower() == "name" else "account"
        slots[key] = labelled.group("v").strip(" .")
        return slots
    parsed = parse_deal_slots(text)
    if "name" in parsed and "account" in parsed:
        return {**slots, **parsed}
    value = parsed.get("name") or parsed.get("account")
    for extra in ("amount", "segment"):
        if extra in parsed:
            slots[extra] = parsed[extra]
    if value:
        slots["name" if not slots.get("name") else "account"] = value
        if "account" in parsed and not slots.get("account") and slots.get("name"):
            slots["account"] = parsed["account"]
    return slots


def parse_rfp_slots(text: str) -> dict[str, str]:
    """name / client / industry only when the person said so explicitly."""
    slots: dict[str, str] = {}
    m = re.search(r"\b(?:called|named|project name|name)\s*[:=]?\s*[\"“]?(?P<v>[^\"”,.;]+?)[\"”]?(?=[,.;]|\s+(?:for|client|industry)\b|$)", text, re.I)
    if m:
        slots["name"] = m.group("v").strip()
    m = re.search(r"\bclient\s*[:=]?\s*(?P<v>[^,.;]+?)(?=[,.;]|\s+(?:industry|called|named)\b|$)", text, re.I)
    if not m:
        m = re.search(r"\bfor\s+(?:the\s+)?(?P<v>[A-Z][\w&'.-]*(?:\s+[A-Z&][\w&'.-]*)*)", text)
    if m:
        slots["client"] = m.group("v").strip()
    m = re.search(r"\bindustry\s*[:=]?\s*(?P<v>[\w &-]+?)(?=[,.;]|$)", text, re.I) or re.search(
        r"\bin the (?P<v>[\w &-]+?) (?:industry|sector)\b", text, re.I
    )
    if m:
        slots["industry"] = m.group("v").strip()
    return slots


def classify_filenames(filenames: list[str]) -> str | None:
    """'rfp' or 'deal' when the file names say so, else None."""
    joined = " ".join(filenames)
    if _RFP_FILE_HINT.search(joined):
        return "rfp"
    if any(f.lower().endswith((".eml", ".csv")) for f in filenames) or _DEAL_FILE_HINT.search(joined):
        return "deal"
    return None


def company_targets(text: str) -> list[str]:
    """Which agents hold the answer to a question about the whole company's records: 'deals' (the pipeline),
    'rfp' (the library and facts), both, or none when it is not that kind of question."""
    portfolio, library = bool(_PORTFOLIO_CUE.search(text or "")), bool(_LIBRARY_CUE.search(text or ""))
    if _EVERYTHING.search(text or "") or (portfolio and library):
        return ["deals", "rfp"]
    return ["deals"] if portfolio else ["rfp"] if library else []


# -- the main entry point -------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _registry() -> list[dict]:
    return load_registry()


def _subject_of(rest: str) -> str | None:
    return _clean_subject(_SUBJECT_END.split(rest.strip(), maxsplit=1)[0])


def _command(text: str, has_files: bool, filenames: list[str], awaiting: dict) -> Intent | None:
    """A recognisable instruction, or None (so a short reply can be read as an answer instead)."""
    if _WS_WORD.search(text):
        if _FORGET_WS.match(text):
            return Intent("workspace_forget", text=text)
        save = _SAVE_WS.match(text)
        if save and re.search(r"\b(?:these|this|current|default|my|our)\b", text, re.I):
            return Intent("workspace_save", subject=_clean_subject(save.group("n")), text=text)
        if _WS_LIST.match(text):
            return Intent("workspace_list", text=text)
        m = _WS_USE.match(text)
        if m:
            agent = (m.group("a") or "").lower()
            return Intent("workspace_use", subject=_clean_subject(m.group("s")), text=text,
                          slots={"agent": "deals" if agent.startswith("deal") else "rfp" if agent else None})
    if re.match(r"^\W*(?:let'?s |i want to |please |can we |now )?(?:review|go through|check)\s+(?:the |my |our |all )?(?:rest|"
                r"answers|drafts|responses|draft answers|rfp answers)\b|^\W*next (?:answer|draft|question)\W*$", text, re.I):
        return Intent("rfp_review", agent="rfp", text=text)
    if _ACCEPT_ALL.search(text):
        return Intent("rfp_accept_all", agent="rfp", text=text)
    if _EXPORT.search(text) and (_FORMAT.search(text) or re.search(r"\b(?:response|answers|it)\b", text, re.I)):
        f = _FORMAT.search(text)
        fmt = None if not f else ("xlsx" if f.group("f").lower() in ("excel", "xlsx", "spreadsheet") else "docx")
        return Intent("rfp_export", agent="rfp", fmt=fmt, text=text)

    note = _NOTE.match(text)
    if note:
        kind = {"email": "email", "meeting note": "meeting"}.get(re.sub(r"\s+", " ", note.group("k").lower()), "call_note")
        return Intent("deal_note", agent="deals", subject=_clean_subject(note.group("s")),
                      slots={"kind": kind, "text": note.group("t").strip()}, text=text)

    for pattern in _FOLLOWUP:
        m = pattern.match(text)
        if m and not (_RFP_NOUN.search(text) and not re.search(r"\bdeal\b", text, re.I)):
            kind = "call_agenda" if "agenda" in (m.groupdict().get("k") or "").lower() else "email"
            return Intent("deal_followup", agent="deals", subject=_clean_subject(m.groupdict().get("s")),
                          slots={"kind": kind}, text=text)

    for pattern in _LIST_DEALS:
        m = pattern.match(text)
        if m:
            return Intent("deal_list", agent="deals", text=text, slots={"filter": (m.group("k") or "").strip().lower()})
    if _COUNT.match(text):
        return Intent("deal_count", agent="deals", text=text)

    for pattern in _OUTCOME_PATTERNS:
        m = pattern.search(text)
        if m:
            subject_raw = m.group("s")
            subject = _subject_of(subject_raw) if pattern is _OUTCOME_PATTERNS[0] else _clean_subject(subject_raw)
            result = m.group("r").lower().replace("closed", "").strip(" -") or None
            reason = parse_reason(text) if result == "lost" else None
            is_rfp = bool(_RFP_NOUN.search(text)) and not re.search(r"\bdeal\b", text, re.I)
            target = "rfp" if is_rfp else ("deal" if subject else None)
            if is_rfp:
                subject = None
            return Intent("outcome", agent="rfp" if is_rfp else "deals", subject=subject, result=result,
                          loss_reason=reason, target=target, slots={"free_reason": free_reason(text)}, text=text)
    if re.search(r"\bno[\s-]+decision\b", text, re.I) and re.search(r"\b(?:rfp|rfi|rfq|bid|proposal|tender|questionnaire)\b", text, re.I):
        return Intent("outcome", agent="rfp", result="no_decision", target="rfp", slots={"free_reason": free_reason(text)}, text=text)

    new = _NEW_DEAL.search(text)
    if new:
        return Intent("deal_new", agent="deals", slots=parse_deal_slots(new.group("rest")), text=text)

    details = _DEAL_DETAILS.match(text)
    if details and not _RFP_NOUN.search(text):
        return Intent("deal_update", agent="deals", subject=_clean_subject(details.group("s")), text=text)

    read = _READ_DEAL.match(text)
    if read and not (_RFP_NOUN.search(text) and not re.search(r"\bdeal\b", text, re.I)):
        subject = _clean_subject(read.group("s"))
        if subject and subject.lower() in ("it", "this", "that", "this deal", "that deal", "the deal"):
            subject = None
        return Intent("deal_read", agent="deals", subject=subject, text=text)

    if has_files:
        add = _ADD_TO_DEAL.search(text)
        if add and not _RFP_NOUN.search(text):
            subject = _clean_subject(add.group("s"))
            if subject:
                return Intent("deal_add", agent="deals", subject=subject, text=text)

    for pattern in _BRIEF_PATTERNS:
        m = pattern.match(text)
        if m:
            rest = m.groupdict().get("s")
            if rest and _RFP_NOUN.search(rest) and not re.search(r"\bdeal\b", rest, re.I):
                break
            return Intent("deal_brief", agent="deals", subject=_clean_subject(rest) if rest else None, text=text)

    if _RFP_NOUN.search(text) and (has_files or _RFP_VERB.search(text) or awaiting.get("type") == "rfp_file"):
        return Intent("rfp_start", agent="rfp", slots=parse_rfp_slots(text), text=text)
    if has_files and _RFP_NOUN.search(text):
        return Intent("rfp_start", agent="rfp", slots=parse_rfp_slots(text), text=text)
    return None



# --- an administrator asking about the platform itself (companies, usage, storage), not about a deal or an RFP ----------------
_METRIC = re.compile(r"\b(?:tokens?|storage|disk|megabytes?|gigabytes?|model calls?|api calls?|llm calls?|usage|consum\w+|spent|used)\b", re.I)
_PLATFORM_SCOPE = re.compile(
    r"\b(?:com\w{1,4}n(?:y|ies)|tenants?|organi[sz]ations?|each client|every client|all (?:the )?clients|clients? (?:use|using|used)"
    r"|(?:this|the|our) (?:application|app|platform|hub|system)|across (?:all|every)|everyone|all users|admin\w*)\b", re.I)
_COUNT_THINGS = re.compile(r"\bhow many (?:com\w{1,4}n(?:y|ies)|tenants?|organi[sz]ations?|users|people|accounts)\b", re.I)
_ABOUT_US = re.compile(
    r"\b(?:(?:this|the|our) (?:application|app|platform|hub|system)|with us|on (?:the )?hub|onboarded|registered|signed up|"
    r"(?:using|use|working with|work with) (?:us|it|the hub|the platform))\b|\bcom\w{1,4}n(?:y|ies) (?:are |is )?(?:currently |now )?(?:working|using|active)\b", re.I)
_BARE_USAGE = re.compile(r"^\W*(?:show |give me |what(?:'s| is) )?(?:the )?(?:usage|usage report|usage stats|admin stats|admin report)\W*$", re.I)
_DEAL_WORDS = re.compile(r"\b(?:deals?|pipeline|opportunit\w+|rfps?|proposals?|questionnaires?|briefs?|soc ?2|security|encryption|lost|won|lose)\b", re.I)


_FACTS_REQ = re.compile(
    r"\b(?:company facts?|fact ?sheet|official facts?|(?:add|save|load|upload|import|set up|update|here are|these are|our)\b[\w ,']{0,30}\bfacts?)\b", re.I)
_PLAYS_REQ = re.compile(r"\b(?:add|save|load|upload|import|set up|update|here are|these are|our|my)\b[\w ,']{0,25}\b(?:sales )?(?:plays|playbook)\b", re.I)
_PROPOSAL_WORDS = re.compile(
    r"\b(?:past|previous|old|older|earlier|submitted|archived)\s+(?:proposals?|responses?|bids?|rfp responses?|tenders?|rfps?)\b"
    r"|\b(?:add|save|import|load|put|upload)\b[\w ,']{0,25}\b(?:to|into) (?:the |our |rfp )*(?:library|answer library|memory)\b"
    r"|\b(?:proposal|response|bid) (?:we|that we) (?:won|lost|submitted)\b|\bwe (?:won|lost) (?:this|it|that) (?:bid|proposal|rfp|tender)\b", re.I)


def memory_request(text: str, has_files: bool) -> str | None:
    """Feeding the memory: "mem_facts" (company facts), "mem_plays" (sales plays) or "mem_proposal" (a past proposal), else None."""
    if _FACTS_REQ.search(text):
        return "mem_facts"
    if _PLAYS_REQ.search(text) or (has_files and re.search(r"\b(?:plays|playbook)\b", text, re.I)):
        return "mem_plays"
    if _PROPOSAL_WORDS.search(text) and (has_files or re.search(r"\b(?:add|import|upload|load|save)\b", text, re.I)):
        return "mem_proposal"
    return None


def is_usage_question(text: str) -> bool:
    """True for "how many companies are using this", "how many tokens did each company use", "storage used by everyone",
    "usage". False for questions about the deals or RFP content that happen to use the same words ("how many companies did we
    lose on price", "what have we said about data storage")."""
    if _BARE_USAGE.match(text):
        return True
    asks_metric = bool(_METRIC.search(text) and _PLATFORM_SCOPE.search(text))
    asks_count = bool(_COUNT_THINGS.search(text) and _ABOUT_US.search(text))
    if not (asks_metric or asks_count):
        return False
    return not _DEAL_WORDS.search(text) or bool(re.search(r"\b(?:tokens?|model calls?|api calls?|usage)\b", text, re.I))


def detect(text: str, has_files: bool, state: dict | None = None, filenames: list[str] | None = None) -> Intent:
    """Classify a message. Kinds: confirm, cancel, greeting, deal_brief, deal_new, deal_add, deal_note,
    outcome, deal_followup, deal_count, deal_list, question, workspace_list, workspace_use, workspace_save, workspace_forget, admin_usage, rfp_start, rfp_accept_all, rfp_export, deal_files, slot, route, none."""
    state = state or {}
    awaiting = state.get("awaiting") or {}
    filenames = filenames or []
    text = " ".join((text or "").split())

    if text and CONFIRM.match(text):
        return Intent("confirm", text=text)
    if text and CANCEL.match(text):
        return Intent("cancel", text=text)

    if text and not has_files and not awaiting and is_usage_question(text):
        return Intent("admin_usage", text=text)
    feed = memory_request(text, has_files) if text and not awaiting else None
    if feed:
        return Intent(feed, agent="deals" if feed == "mem_plays" else "rfp", text=text)

    command = _command(text, has_files, filenames, awaiting) if text else None
    if command:
        return command
    if awaiting and (text or has_files):
        if awaiting.get("type") == "rfp_file":
            if has_files:
                return Intent("rfp_start", agent="rfp", slots=parse_rfp_slots(text), text=text)
        else:
            return Intent("slot", subject=awaiting.get("type"), text=text)

    if text and not has_files and (_QUESTION.match(text) or text.rstrip().endswith("?")):
        return Intent("question", agent="deals", text=text)

    if has_files:
        hint = classify_filenames(filenames)
        if hint == "rfp":
            return Intent("rfp_start", agent="rfp", slots=parse_rfp_slots(text), text=text)
        if hint == "deal":
            return Intent("deal_files", agent="deals", text=text)
        routed = route(text, _registry()) if text else None
        if routed and routed.kind == "match" and routed.best.agent["id"] in ("rfp", "deals"):
            kind = "rfp_start" if routed.best.agent["id"] == "rfp" else "deal_files"
            return Intent(kind, agent=routed.best.agent["id"], slots=parse_rfp_slots(text), text=text)
        return Intent("files_unclear", text=text)

    routed = route(text, _registry())
    if routed.kind == "greeting":
        return Intent("greeting", text=text)
    if routed.kind == "none":
        return Intent("none", text=text)
    return Intent("route", agent=routed.best.agent["id"], text=text)
