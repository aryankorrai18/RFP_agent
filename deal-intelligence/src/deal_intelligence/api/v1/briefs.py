"""The deal brief: one model call, then every claim is checked in code before it is stored.

The model's output is never trusted. Cited ids must exist (INT- ids belong to this deal; D- ids are among
the closed deals offered in this arm), this-deal claims cite INT-, history claims cite D-, recommended
plays come from the offered candidates, and lesson ids are never citable. What fails is recorded in
`Brief.flags`; the deterministic flags and warnings are added to the stored content whatever the model said.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import select

from ...config import RETRIEVAL_MODES
from ...errors import PipelineError
from ...providers.base import LLMError, TokenUsage
from ...providers.errors import explain_message
from ...schemas import BriefClaim, BriefStep, DealBriefResult
from .brief_prompts import BRIEF_PROMPT_VERSION, build_prompt, candidate_plays
from .contracts import Flag, Recommendations
from .db import Brief, Database, Deal, Job, Play, parse_deal_code, parse_interaction_code
from .jobs import JobFailed
from .signals import compute_flags, deal_facts_view, llm_failure_text

if TYPE_CHECKING:
    from .context import V1Context

log = logging.getLogger(__name__)

MAX_THIS_DEAL, MAX_MEMORY, MAX_STEPS = 4, 3, 3
_LESSON_ID = re.compile(r"^(lesson|les|outcome|play)[:_-]", re.IGNORECASE)


# ---- Validation ------------------------------------------------------------------------------------------

class _Checker:
    """Classifies and checks the ids the model cited. Findings accumulate in `findings`."""

    def __init__(self, own_interactions: set[str], deal_codes: set[str], lesson_ids: set[str]):
        self.own, self.deals, self.lessons = own_interactions, deal_codes, lesson_ids
        self.findings: list[dict] = []

    def finding(self, code: str, where: str, detail: str) -> None:
        self.findings.append({"code": code, "where": where, "detail": detail})

    def ids(self, raw: list[str], expect: str, where: str) -> list[str]:
        """The valid ids of `raw` for the class `expect` ("int", "deal" or "any")."""
        valid: list[str] = []
        for value in raw:
            code = str(value).strip()
            upper = code.upper()
            if code in self.lessons or _LESSON_ID.match(code):
                self.finding("lesson_cited", where, f"{code} is a lesson and cannot be cited")
            elif parse_interaction_code(upper) is not None:
                if expect == "deal":
                    self.finding("wrong_id_class", where, f"{upper} is an interaction id; this field cites past deals")
                elif upper in self.own:
                    valid.append(upper)
                else:
                    self.finding("unknown_citation", where, f"{upper} is not an interaction of this deal")
            elif parse_deal_code(upper) is not None:
                if expect == "int":
                    self.finding("wrong_id_class", where, f"{upper} is a past deal; this field cites this deal's interactions")
                elif upper in self.deals:
                    valid.append(upper)
                else:
                    self.finding("unknown_citation", where, f"{upper} was not offered as a past deal")
            else:
                self.finding("unknown_citation", where, f"{code!r} is not a known id")
        return list(dict.fromkeys(valid))

    def claims(self, items: list[BriefClaim], expect: str, label: str, limit: int) -> list[dict]:
        out = []
        for index, item in enumerate(items[:limit]):
            where = f"{label}[{index}]"
            sources = self.ids(item.source_ids, expect, where)
            if not sources:
                self.finding("uncited_claim", where, f"No valid source for: {item.text[:120]}")
            out.append({"text": item.text, "source_ids": sources, "verified": bool(sources)})
        return out

    def steps(self, items: list[BriefStep], plays: dict[str, dict], avoided: frozenset[str] = frozenset()) -> list[dict]:
        out: list[dict] = []
        for index, item in enumerate(items):
            where = f"next_steps[{index}]"
            code = item.play_code.strip().upper()
            sources = self.ids(item.source_ids, "deal", where)
            if code in avoided:
                self.finding("avoided_play", where, f"{code} was tried in similar deals and never won; step dropped")
                continue
            if code not in plays:
                self.finding("play_not_offered", where, f"{item.play_code} is not one of the offered plays; step dropped")
                continue
            if any(step["play_code"] == code for step in out) or len(out) >= MAX_STEPS:
                continue
            play = plays[code]
            out.append({"play_code": code, "name": play["name"], "rationale": item.rationale, "source_ids": sources,
                        "counts": play["counts"], "reasons": play.get("reasons", [])})
        return out


def _validate(output: DealBriefResult, checker: _Checker, plays: dict[str, dict],
              avoided: frozenset[str] = frozenset()) -> dict:
    summary_sources = checker.ids(output.summary_sources, "any", "summary")
    if not summary_sources:
        checker.finding("uncited_summary", "summary", "The summary cites no valid source")
    return {
        "summary": output.summary, "summary_sources": summary_sources,
        "this_deal": checker.claims(output.this_deal, "int", "this_deal", MAX_THIS_DEAL),
        "memory": checker.claims(output.memory, "deal", "memory", MAX_MEMORY),
        "next_steps": checker.steps(output.next_steps, plays, avoided),
        "missing_info": [m for m in output.missing_info if m.strip()],
    }


# ---- Inputs ----------------------------------------------------------------------------------------------

def _catalogue(db: Database) -> list[dict]:
    with db.session() as session:
        return [
            {"code": p.code, "name": p.name, "description": p.description, "category": p.category,
             "addresses": list(p.addresses or [])}
            for p in session.scalars(select(Play).order_by(Play.code))
        ]


def _closed_summary(deal: Deal) -> str:
    signals = deal.signals
    parts = [f"{deal.account} ({deal.industry or 'industry unknown'}, {deal.segment or 'segment unknown'}"
             + (f", {deal.amount}" if deal.amount else "") + f"): {deal.result}"
             + (f" ({deal.loss_reason.replace('_', ' ')})" if deal.loss_reason else "") + "."]
    if signals:
        if signals.objections:
            parts.append("Objections: " + "; ".join(f"{o.get('type')} ({o.get('status')})" for o in signals.objections) + ".")
        if signals.competitors:
            parts.append("Competitors: " + ", ".join(signals.competitors) + ".")
        if signals.plays_used:
            parts.append("Plays used: " + ", ".join(signals.plays_used) + ".")
    if not any(s.stance == "champion" for s in deal.stakeholders):
        parts.append("No champion was identified.")
    return " ".join(parts)


def _closed_deals(db: Database, deal_id: int, recommendations: Recommendations) -> list[dict]:
    """Every closed deal as prompt data (longctx). A retained summary from memory is used when the
    recommendations carry one; otherwise a summary is composed from the record."""
    known = {s.code: s for s in recommendations.similar}
    with db.session() as session:
        deals = session.scalars(
            select(Deal).where(Deal.result.in_(("won", "lost")), Deal.status == "active", Deal.id != deal_id).order_by(Deal.id)
        ).all()
        return [
            {"code": d.code, "account": d.account, "result": d.result, "loss_reason": d.loss_reason,
             "plays_used": list(d.signals.plays_used) if d.signals else [],
             "summary": known[d.code].summary if d.code in known and known[d.code].summary else _closed_summary(d)}
            for d in deals
        ]


async def _recommendations(
    ctx: V1Context, deal_id: int, mode: str, given: Recommendations | None
) -> Recommendations:
    if given is not None:
        if mode == "none":  # the none arm sees no memory, whatever was passed
            return Recommendations(deal_id=deal_id, mode="none", n_closed=given.n_closed)
        return given
    if mode == "none":
        with ctx.db.session() as session:
            closed = len(session.scalars(select(Deal.id).where(Deal.result.in_(("won", "lost")), Deal.status == "active")).all())
        return Recommendations(deal_id=deal_id, mode="none", n_closed=closed)
    from .retrieval import build_recommendations  # written by the memory layer; imported late on purpose

    return await build_recommendations(ctx, deal_id, mode)


_GATE_LINES = {
    "no_champion": ("champion", "No champion is identified: find out who will push this deal internally."),
    "no_economic_buyer": ("economic buyer", "The economic buyer is not identified."),
}


def _missing_info(model_items: list[str], flags: list[Flag], view: dict, n_interactions: int) -> list[str]:
    """The model's gaps plus the ones code can see: the gate never depends on the model remembering."""
    items = list(model_items)
    lowered = " ".join(items).casefold()
    for flag in flags:
        needle, line = _GATE_LINES.get(flag.code, (None, None))
        if needle and needle not in lowered:
            items.append(line)
    if view["deal"]["signals_status"] != "ready":
        items.append("Deal signals have not been extracted yet, so stakeholders, objections and promises may be incomplete.")
    if not n_interactions:
        items.append("No interactions are recorded for this deal yet.")
    return items


# ---- Generation --------------------------------------------------------------------------------------------

async def generate_brief(
    ctx: V1Context,
    deal_id: int,
    mode: str | None = None,
    *,
    recommendations: Recommendations | None = None,
    job_id: int | None = None,
    today: date | None = None,
) -> Brief:
    settings = ctx.settings
    mode = mode or settings.retrieval_mode
    if mode not in RETRIEVAL_MODES:
        raise PipelineError("bad_mode", f"Retrieval mode must be one of {', '.join(RETRIEVAL_MODES)}.", 400)
    if job_id is not None:  # a resumed job must not write the same arm twice
        with ctx.db.session() as session:
            done = session.scalars(
                select(Brief).where(Brief.job_id == job_id, Brief.mode == mode, Brief.status == "ready")
            ).first()
        if done is not None:
            return done

    today = today or date.today()
    view = deal_facts_view(ctx.db, deal_id, today)
    flags = compute_flags(ctx.db, deal_id, today)
    catalogue = _catalogue(ctx.db)
    recs = await _recommendations(ctx, deal_id, mode, recommendations)
    closed = _closed_deals(ctx.db, deal_id, recs) if mode == "longctx" else None
    system, user, prompt_hash = build_prompt(view, flags, recs, catalogue, mode, closed_deals=closed)

    plays = {p["code"]: p for p in candidate_plays(recs, catalogue, view, mode)}
    ranked = {p.play_code: p for p in recs.plays} if mode != "none" else {}
    for code, play in plays.items():
        play["reasons"] = list(ranked[code].reasons) if code in ranked else []
    if mode == "none":
        deal_codes: set[str] = set()
    elif mode == "longctx":
        deal_codes = {d["code"] for d in closed or []}
    else:
        deal_codes = {s.code for s in recs.similar}
    avoided = frozenset(a.play_code for a in recs.avoid) if mode in ("similar", "hindsight") else frozenset()
    lessons = {e for p in recs.plays for e in p.lesson_evidence}
    own = {i["id"] for i in view["interactions"]}

    base = dict(
        deal_id=deal_id, mode=mode, evidence=recs.to_dict(), memory_state=recs.memory_state,
        prompt_version=BRIEF_PROMPT_VERSION, prompt_hash=prompt_hash, job_id=job_id,
    )

    usage = TokenUsage()
    model_name: str | None = None

    async def ask(text: str) -> DealBriefResult:
        nonlocal model_name
        result = await ctx.llm.structured(
            purpose="brief", output_format=DealBriefResult, system=system, user=text,
            effort=settings.brief_effort, max_tokens=3000, temperature=0.0,
        )
        usage.add(result.usage)
        model_name = result.model
        return result.output

    try:
        output = await ask(user)
    except LLMError as exc:
        return _store(ctx.db, Brief(status="failed", content={}, flags=[], error=llm_failure_text(exc, settings, "Writing the brief"),
                                    model=model_name, **base))

    checker = _Checker(own, deal_codes, lessons)
    content = _validate(output, checker, plays, avoided)
    if not content["next_steps"] and plays:
        # The only step a brief must have is one the team can actually take: ask once more, naming the violation.
        named = sorted({f["detail"].split(" is not")[0] for f in checker.findings if f["code"] == "play_not_offered"})
        barred = sorted({f["detail"].split(" was tried")[0] for f in checker.findings if f["code"] == "avoided_play"})
        problem = (f"you recommended play codes that were not offered ({', '.join(named)})" if named
                   else f"you recommended plays that must be avoided ({', '.join(barred)})" if barred
                   else "you gave no next steps")
        correction = (f"\n\n<correction>Your previous answer was rejected: {problem}. Answer again and recommend only "
                      f"these play codes in next_steps: {', '.join(plays)}.</correction>")
        try:
            retry = await ask(user + correction)
        except LLMError as exc:
            checker.finding("retry_failed", "next_steps", llm_failure_text(exc, settings, "The second attempt")[:200])
        else:
            first = checker.findings
            checker = _Checker(own, deal_codes, lessons)
            checker.findings = [f for f in first if f["code"] in ("play_not_offered", "lesson_cited", "avoided_play")]
            checker.finding("retried", "next_steps", problem)
            content = _validate(retry, checker, plays, avoided)
        if not content["next_steps"]:
            checker.finding("no_valid_next_steps", "next_steps", "No recommended play passed validation")

    content.update({
        "missing_info": _missing_info(content["missing_info"], flags, view, len(view["interactions"])),
        "flags": [f.to_dict() for f in flags],
        "warnings": [w.to_dict() for w in recs.warnings] if mode != "none" else [],
        "avoid": [a.to_dict() for a in recs.avoid] if mode in ("similar", "hindsight") else [],
        "similar": sorted(deal_codes) if mode == "longctx" else [s.code for s in recs.similar] if mode != "none" else [],
        "mode": mode,
        "n_closed": len(closed) if closed is not None else recs.n_closed,
        "degraded": recs.degraded,
    })
    return _store(ctx.db, Brief(
        status="ready", content=content, flags=checker.findings, model=model_name,
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, **base,
    ))


def _store(db: Database, brief: Brief) -> Brief:
    with db.session() as session:
        session.add(brief)
        session.commit()
    return brief


# ---- Reading --------------------------------------------------------------------------------------------------

def latest_brief(db: Database, deal_id: int, mode: str | None = None, *, include_failed: bool = False) -> Brief | None:
    query = select(Brief).where(Brief.deal_id == deal_id)
    if mode:
        query = query.where(Brief.mode == mode)
    if not include_failed:
        query = query.where(Brief.status == "ready")
    with db.session() as session:
        return session.scalars(query.order_by(Brief.id.desc())).first()


def brief_view(brief: Brief) -> dict:
    return {
        "id": brief.id, "deal_id": brief.deal_id, "mode": brief.mode, "status": brief.status,
        "content": brief.content, "flags": brief.flags, "evidence": brief.evidence,
        "memory_state": brief.memory_state, "prompt_version": brief.prompt_version, "prompt_hash": brief.prompt_hash,
        "model": brief.model, "input_tokens": brief.input_tokens, "output_tokens": brief.output_tokens,
        "job_id": brief.job_id, "error": brief.error, "error_explanation": explain_message(brief.error),
        "created_at": brief.created_at.isoformat() if brief.created_at else None,
    }


# ---- Job ---------------------------------------------------------------------------------------------------------

async def run_brief(ctx: V1Context, job_id: int) -> None:
    with ctx.db.session() as session:
        job = session.get(Job, job_id)
        deal_id, mode = job.target_id, (job.payload or {}).get("mode") or None
        job.total, job.done = 1, 0
        session.commit()
    try:
        brief = await generate_brief(ctx, deal_id, mode, job_id=job_id)
    except PipelineError as exc:
        raise JobFailed(exc.message) from exc
    if brief.status == "failed":
        raise JobFailed(brief.error or "The brief could not be written.")
    with ctx.db.session() as session:
        session.get(Job, job_id).done = 1
        session.commit()
