"""The V0 pipeline: parse -> AI call #1 (extract) -> AI call #2 per requirement (draft) -> check."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
from datetime import date, datetime
from pathlib import Path

from pydantic import ValidationError

from .config import Settings
from .grounding import evaluate_draft
from .llm import LLM, LLMError, TokenUsage
from .parser import parse_document
from .prompts import PROMPT_VERSION
from .provider_errors import StopOnBlocking
from .schemas import (
    Draft,
    DraftResponse,
    Fact,
    FactOut,
    FactSheet,
    Requirement,
    RunSummary,
    Stats,
    Timings,
    Usage,
)


class PipelineError(Exception):
    """A run-level failure. Maps directly onto the API error body and HTTP status."""

    def __init__(self, code: str, message: str, http_status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


def load_fact_sheet(raw: bytes | str, source: str) -> FactSheet:
    try:
        return FactSheet.model_validate(json.loads(raw))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PipelineError("invalid_fact_sheet", f"{source} is not valid JSON: {exc}", 422) from exc
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'fact sheet'}: {e['msg']}" for e in exc.errors())
        raise PipelineError("invalid_fact_sheet", f"{source} is invalid: {problems}", 422) from exc


def load_default_fact_sheet(settings: Settings) -> FactSheet:
    path = settings.fact_sheet_path
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PipelineError(
            "no_company_facts",
            "This workspace has no company facts yet. Add them on the Company page before drafting.", 409,
        ) from exc
    return load_fact_sheet(raw, path.name)


async def run_pipeline(
    *,
    filename: str,
    data: bytes,
    fact_sheet: FactSheet,
    llm: LLM,
    settings: Settings,
    today: date | None = None,
) -> DraftResponse:
    today = today or date.today()
    started = time.perf_counter()
    warnings: list[str] = []

    document = parse_document(filename, data, settings.max_document_chars)
    parsed_at = time.perf_counter()
    if document.pdf_bytes is not None:
        warnings.append(
            "The PDF had little or no extractable text (a scanned document?), so it was sent to the "
            "model as a PDF. Check the extracted requirements carefully."
        )

    live_facts: list[Fact] = [f for f in fact_sheet.facts if f.is_live(today)]
    expired = len(fact_sheet.facts) - len(live_facts)
    if not live_facts:
        raise PipelineError("invalid_fact_sheet", "Every fact in the fact sheet has expired.", 422)
    if expired:
        warnings.append(f"{expired} expired fact(s) were excluded from the fact sheet.")

    usage = TokenUsage()
    try:
        extraction = await llm.extract_requirements(document)
    except LLMError as exc:
        code = "llm_auth_failed" if exc.kind == "auth" else "extraction_failed"
        raise PipelineError(code, exc.message, 502) from exc
    usage.add(extraction.usage)
    extracted_at = time.perf_counter()

    requirements = [
        Requirement(
            id=f"REQ-{index:03d}",
            section=(item.section or "").strip() or None,
            question=item.question.strip(),
            mandatory=item.mandatory,
            word_limit=item.word_limit if item.word_limit and item.word_limit > 0 else None,
            reference=(item.reference or "").strip() or None,
        )
        for index, item in enumerate((i for i in extraction.output.requirements if i.question.strip()), start=1)
    ]
    if not requirements:
        raise PipelineError("no_requirements_found", f"No requirements were found in {filename}.", 422)
    if len(requirements) > settings.max_requirements:
        raise PipelineError(
            "too_many_requirements",
            f"{filename} has {len(requirements)} requirements; the V0 limit is {settings.max_requirements}. "
            "Split the document or raise RFP_MAX_REQUIREMENTS.",
            422,
        )

    live_ids = {f.id for f in live_facts}
    semaphore = asyncio.Semaphore(settings.draft_concurrency)
    breaker = StopOnBlocking()

    async def draft_one(requirement: Requirement) -> tuple[Draft, TokenUsage]:
        async with semaphore:
            if breaker.tripped:  # every remaining call would fail the same way: don't spend them
                breaker.skipped += 1
                return Draft(requirement_id=requirement.id, status="failed",
                             error=f"Not attempted: an earlier call failed and this one would too. {breaker.message}"), TokenUsage()
            try:
                result = await llm.draft_answer(fact_sheet.company, live_facts, requirement)
            except LLMError as exc:
                # One failed requirement never fails the run.
                breaker.record(exc.message)
                return Draft(requirement_id=requirement.id, status="failed", error=exc.message), TokenUsage()
        return evaluate_draft(requirement, result.output, live_ids, result.model), result.usage

    outcomes = await asyncio.gather(*(draft_one(r) for r in requirements))
    drafted_at = time.perf_counter()
    stopped = breaker.summary(len(requirements) - breaker.skipped, len(requirements), settings.provider, llm.model)
    if stopped:
        warnings.append(stopped)

    drafts = []
    for draft, draft_usage in outcomes:
        drafts.append(draft)
        usage.add(draft_usage)

    response = DraftResponse(
        run_id=f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}",
        rfp_filename=filename,
        company=fact_sheet.company,
        model=llm.model,
        prompt_version=PROMPT_VERSION,
        requirements=requirements,
        drafts=drafts,
        facts=[FactOut(id=f.id, topic=f.topic, statement=f.statement) for f in live_facts],
        stats=Stats(
            requirements=len(requirements),
            drafted=sum(d.status == "drafted" for d in drafts),
            grounded=sum(d.grounded for d in drafts),
            needs_sme=sum(d.status == "needs_sme" for d in drafts),
            failed=sum(d.status == "failed" for d in drafts),
            flagged=sum(bool(d.flags) for d in drafts),
        ),
        usage=Usage(**vars(usage)),
        timings_ms=Timings(
            parse=_ms(started, parsed_at),
            extraction=_ms(parsed_at, extracted_at),
            drafting=_ms(extracted_at, drafted_at),
            total=_ms(started, drafted_at),
        ),
        warnings=warnings,
    )
    saved = save_run(response, settings.runs_dir)
    if saved is None:
        response.warnings.append(f"The run could not be saved to {settings.runs_dir}.")
    return response


RUN_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def list_runs(runs_dir: Path, limit: int = 20) -> list[RunSummary]:
    """Newest first. Unreadable files are skipped rather than failing the list."""
    if not runs_dir.is_dir():
        return []
    summaries: list[RunSummary] = []
    for path in sorted(runs_dir.glob("*.json"), reverse=True):
        if not RUN_ID_PATTERN.match(path.stem):
            continue
        try:
            run = DraftResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            continue
        summaries.append(
            RunSummary(
                run_id=run.run_id,
                created=datetime.strptime(run.run_id[:15], "%Y%m%d-%H%M%S").isoformat(),
                rfp_filename=run.rfp_filename,
                company=run.company,
                model=run.model,
                stats=run.stats,
            )
        )
        if len(summaries) >= limit:
            break
    return summaries


def load_run(runs_dir: Path, run_id: str) -> DraftResponse:
    # The pattern check also blocks path traversal ("../..").
    path = runs_dir / f"{run_id}.json"
    if not RUN_ID_PATTERN.match(run_id) or not path.is_file():
        raise PipelineError("run_not_found", f"No saved run with id {run_id!r}.", 404)
    try:
        return DraftResponse.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise PipelineError("run_unreadable", f"Saved run {run_id} could not be read.", 500) from exc


def save_run(response: DraftResponse, runs_dir: Path) -> Path | None:
    """Best effort: a failed save is reported as a warning, never as a failed run."""
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{response.run_id}.json"
        path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
        return path
    except OSError:
        return None


def _ms(start: float, end: float) -> int:
    return round((end - start) * 1000)
