"""Shared application errors and company fact-sheet loading."""

from __future__ import annotations

import json

from pydantic import ValidationError

from .config import Settings
from .schemas import FactSheet


class PipelineError(Exception):
    """A user-facing application failure with an HTTP status and stable error code."""

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
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'fact sheet'}: {error['msg']}"
            for error in exc.errors()
        )
        raise PipelineError("invalid_fact_sheet", f"{source} is invalid: {problems}", 422) from exc


def load_default_fact_sheet(settings: Settings) -> FactSheet:
    path = settings.fact_sheet_path
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PipelineError(
            "no_company_facts",
            "This workspace has no company facts yet. Add them on the Company page before drafting.",
            409,
        ) from exc
    return load_fact_sheet(raw, path.name)
