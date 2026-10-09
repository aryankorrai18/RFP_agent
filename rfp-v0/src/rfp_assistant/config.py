"""Settings, read once from the environment (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

PROVIDERS = ("anthropic", "gemini", "groq")
RETRIEVAL_MODES = ("none", "plain", "outcome", "hindsight")
RELEVANCE_POLICIES = ("rank", "gated")  # see rfp_assistant/api/v1/ranking.py
# Environment variables each provider's SDK reads its key from.
PROVIDER_KEY_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
}
DEFAULT_MODEL = {"anthropic": "claude-opus-5", "gemini": "gemini-3.5-flash-lite", "groq": "llama-3.3-70b-versatile"}
# The Gemini and Groq free tiers have low per-minute limits, so draft fewer questions at once.
DEFAULT_CONCURRENCY = {"anthropic": 8, "gemini": 2, "groq": 2}


def has_key(provider: str) -> bool:
    return any(os.environ.get(name) for name in PROVIDER_KEY_VARS[provider])


def resolve_provider(raw: str | None) -> str:
    """auto (the default) picks the provider whose key is present: Anthropic, then Gemini, then Groq."""
    value = (raw or "auto").strip().lower()
    if value in PROVIDERS:
        return value
    if value != "auto":
        raise ValueError(f"RFP_LLM_PROVIDER must be auto, anthropic, gemini or groq, got {value!r}")
    if has_key("anthropic"):
        return "anthropic"
    for provider in ("gemini", "groq"):
        if has_key(provider):
            return provider
    return "anthropic"


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not 0 <= value < 1:
        raise ValueError(f"{name} must be at least 0 and below 1, got {value}")
    return value


def _recall_order_env() -> str:
    value = os.environ.get("RFP_RECALL_ORDER", "semantic").strip().lower() or "semantic"
    if value not in ("semantic", "hindsight"):
        raise ValueError(f"RFP_RECALL_ORDER must be semantic or hindsight, got {value!r}")
    return value


def _relevance_env() -> str:
    value = os.environ.get("RFP_RETRIEVAL_RELEVANCE", "gated").strip().lower() or "gated"
    if value not in RELEVANCE_POLICIES:
        raise ValueError(f"RFP_RETRIEVAL_RELEVANCE must be one of {RELEVANCE_POLICIES}, got {value!r}")
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _effort_env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in EFFORT_LEVELS:
        raise ValueError(f"{name} must be one of {EFFORT_LEVELS}, got {value!r}")
    return value


def _path_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Settings:
    provider: str = "anthropic"
    model: str = "claude-opus-5"
    extraction_effort: str = "low"
    draft_effort: str = "medium"
    draft_concurrency: int = 8
    max_requirements: int = 150
    max_upload_mb: int = 20
    max_document_chars: int = 400_000
    fact_sheet_path: Path = ROOT / "data" / "fact_sheet.json"
    # Application data and retrieval
    db_path: Path = ROOT / "data" / "rfp.db"
    uploads_dir: Path = ROOT / "data" / "uploads"
    hindsight_url: str = "http://127.0.0.1:8888"
    hindsight_api_key: str | None = field(default=None, repr=False)  # only for Hindsight Cloud; never logged
    hindsight_bank: str = "rfp-library"
    # Hindsight lessons bank: outcomes, reviews and debriefs that drive ranking, briefs and the playbook.
    hindsight_lessons_bank: str = "rfp-lessons"
    lessons_enabled: bool = True
    retrieval_top_k: int = 3
    # hindsight (default) ranks with the Hindsight lessons bank; outcome uses local SQLite signals only;
    # plain uses semantic recall order; none disables past-answer retrieval.
    retrieval_mode: str = "hindsight"
    retrieval_freshness_half_life_days: int = 730
    # Memory may reorder answers that are about equally relevant, never lift a less relevant one above
    # them ("gated"). "rank" is the V2 pre-fix rule, kept to reproduce the earlier results.
    retrieval_relevance: str = "gated"
    retrieval_relevance_min_share: float = 0.01
    # How Hindsight's candidates are ordered: "semantic" sorts them by Hindsight's own semantic score; "hindsight" keeps the
    # order Hindsight returns. The 2026-10-08 pilot found Hindsight's fused order buried the right answer (1 of 6 in the
    # top 3) while its semantic score ranked it first (6 of 6); see pilot/issues.csv.
    recall_order: str = "semantic"
    evidence_check: bool = True

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        provider = resolve_provider(os.environ.get("RFP_LLM_PROVIDER"))
        retrieval_mode = os.environ.get("RFP_RETRIEVAL_MODE", cls.retrieval_mode).strip().lower()
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"RFP_RETRIEVAL_MODE must be one of {RETRIEVAL_MODES}, got {retrieval_mode!r}")
        return cls(
            provider=provider,
            model=os.environ.get("RFP_MODEL", "").strip() or DEFAULT_MODEL[provider],
            extraction_effort=_effort_env("RFP_EXTRACTION_EFFORT", cls.extraction_effort),
            draft_effort=_effort_env("RFP_DRAFT_EFFORT", cls.draft_effort),
            draft_concurrency=_int_env("RFP_DRAFT_CONCURRENCY", DEFAULT_CONCURRENCY[provider]),
            max_requirements=_int_env("RFP_MAX_REQUIREMENTS", cls.max_requirements),
            max_upload_mb=_int_env("RFP_MAX_UPLOAD_MB", cls.max_upload_mb),
            max_document_chars=_int_env("RFP_MAX_DOCUMENT_CHARS", cls.max_document_chars),
            fact_sheet_path=_path_env("RFP_FACT_SHEET", ROOT / "data" / "fact_sheet.json"),
            db_path=_path_env("RFP_DB_PATH", ROOT / "data" / "rfp.db"),
            uploads_dir=_path_env("RFP_UPLOADS_DIR", ROOT / "data" / "uploads"),
            hindsight_url=os.environ.get("RFP_HINDSIGHT_URL", cls.hindsight_url).strip() or cls.hindsight_url,
            hindsight_api_key=os.environ.get("RFP_HINDSIGHT_API_KEY", "").strip() or None,
            hindsight_bank=os.environ.get("RFP_HINDSIGHT_BANK", cls.hindsight_bank).strip() or cls.hindsight_bank,
            hindsight_lessons_bank=os.environ.get("RFP_HINDSIGHT_LESSONS_BANK", cls.hindsight_lessons_bank).strip()
            or cls.hindsight_lessons_bank,
            lessons_enabled=os.environ.get("RFP_LESSONS", "true").strip().lower() not in ("0", "false", "no", "off"),
            retrieval_top_k=_int_env("RFP_RETRIEVAL_TOP_K", cls.retrieval_top_k),
            retrieval_mode=retrieval_mode,
            retrieval_relevance=_relevance_env(),
            recall_order=_recall_order_env(),
            retrieval_relevance_min_share=_float_env("RFP_RETRIEVAL_RELEVANCE_MIN_SHARE", cls.retrieval_relevance_min_share),
            retrieval_freshness_half_life_days=_int_env(
                "RFP_RETRIEVAL_FRESHNESS_HALF_LIFE_DAYS", cls.retrieval_freshness_half_life_days
            ),
            evidence_check=os.environ.get("RFP_EVIDENCE_CHECK", "true").strip().lower()
            not in ("0", "false", "no", "off"),
        )
