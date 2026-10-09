"""Settings, read once from the environment (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

PROVIDERS = ("anthropic", "gemini", "groq")
# none: this deal's own records only. longctx: every closed-deal summary in the prompt. similar: the
# closed deals Hindsight recalls as similar. hindsight: similar deals plus the structural gate and the
# lessons that rank plays and warnings. See deal_intelligence/api/v1/retrieval.py.
RETRIEVAL_MODES = ("none", "longctx", "similar", "hindsight")
# auto: the free local SQLite memory when no Hindsight key is set and the URL is Hindsight Cloud, else
# Hindsight. hindsight / local force one. See resolve_memory_backend.
MEMORY_BACKENDS = ("auto", "hindsight", "local")
# Environment variables each provider's SDK reads its key from.
PROVIDER_KEY_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
}
DEFAULT_MODEL = {"anthropic": "claude-opus-5", "gemini": "gemini-3.5-flash-lite", "groq": "llama-3.3-70b-versatile"}
# The Gemini and Groq free tiers have low per-minute limits, so run fewer model calls at once.
DEFAULT_CONCURRENCY = {"anthropic": 8, "gemini": 2, "groq": 2}


def has_key(provider: str) -> bool:
    return any(os.environ.get(name) for name in PROVIDER_KEY_VARS[provider])


def resolve_provider(raw: str | None) -> str:
    """auto (the default) picks the provider whose key is present: Anthropic, then Gemini, then Groq."""
    value = (raw or "auto").strip().lower()
    if value in PROVIDERS:
        return value
    if value != "auto":
        raise ValueError(f"DEAL_LLM_PROVIDER must be auto, anthropic, gemini or groq, got {value!r}")
    if has_key("anthropic"):
        return "anthropic"
    for provider in ("gemini", "groq"):
        if has_key(provider):
            return provider
    return "anthropic"


def resolve_memory_backend(settings: Settings) -> str:
    """"hindsight" or "local". A local Hindsight server URL without a key stays "hindsight"."""
    value = (settings.memory_backend or "auto").strip().lower()
    if value in ("hindsight", "local"):
        return value
    if value != "auto":
        raise ValueError(f"DEAL_MEMORY_BACKEND must be one of {MEMORY_BACKENDS}, got {value!r}")
    cloud = "vectorize.io" in settings.hindsight_url
    return "local" if cloud and not settings.hindsight_api_key else "hindsight"


def _recall_order_env() -> str:
    value = os.environ.get("DEAL_RECALL_ORDER", "semantic").strip().lower() or "semantic"
    if value not in ("semantic", "hindsight"):
        raise ValueError(f"DEAL_RECALL_ORDER must be semantic or hindsight, got {value!r}")
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


def _bool_env(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _path_env(name: str, default: Path | None) -> Path | None:
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
    brief_effort: str = "medium"
    concurrency: int = 8
    max_upload_mb: int = 20
    max_document_chars: int = 400_000
    # Application data and memory
    db_path: Path = ROOT / "data" / "deals.db"
    uploads_dir: Path = ROOT / "data" / "uploads"
    memory_backend: str = "auto"
    hindsight_url: str = "http://127.0.0.1:8888"
    hindsight_api_key: str | None = field(default=None, repr=False)  # only for Hindsight Cloud; never logged
    # Bank 1: deal interactions and closed-deal summaries (chunks mode, verbatim, no Hindsight LLM).
    hindsight_bank: str = "deal-interactions"
    # Bank 2: outcome lessons (concise extraction + observations) for the Memory page and playbook.
    hindsight_lessons_bank: str = "deal-lessons"
    lessons_enabled: bool = True
    similar_deals_k: int = 10  # most relevant closed deals kept as "similar"; counts and warnings are computed over these
    retrieval_mode: str = "hindsight"
    # A recalled deal only counts as similar if it shares at least this many structured keys
    # (objection type, competitor, segment, industry) with the open deal: the structural gate.
    min_shared_keys: int = 1
    # "semantic" orders Hindsight's candidates by its own semantic score before the pool is cut; "hindsight" keeps the
    # order Hindsight returns. The RFP pilot (2026-10-08) showed the fused order can bury the closest match.
    recall_order: str = "semantic"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        provider = resolve_provider(os.environ.get("DEAL_LLM_PROVIDER"))
        retrieval_mode = os.environ.get("DEAL_RETRIEVAL_MODE", cls.retrieval_mode).strip().lower()
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"DEAL_RETRIEVAL_MODE must be one of {RETRIEVAL_MODES}, got {retrieval_mode!r}")
        memory_backend = os.environ.get("DEAL_MEMORY_BACKEND", cls.memory_backend).strip().lower() or cls.memory_backend
        if memory_backend not in MEMORY_BACKENDS:
            raise ValueError(f"DEAL_MEMORY_BACKEND must be one of {MEMORY_BACKENDS}, got {memory_backend!r}")
        return cls(
            provider=provider,
            model=os.environ.get("DEAL_MODEL", "").strip() or DEFAULT_MODEL[provider],
            extraction_effort=_effort_env("DEAL_EXTRACTION_EFFORT", cls.extraction_effort),
            brief_effort=_effort_env("DEAL_BRIEF_EFFORT", cls.brief_effort),
            concurrency=_int_env("DEAL_CONCURRENCY", DEFAULT_CONCURRENCY[provider]),
            max_upload_mb=_int_env("DEAL_MAX_UPLOAD_MB", cls.max_upload_mb),
            max_document_chars=_int_env("DEAL_MAX_DOCUMENT_CHARS", cls.max_document_chars),
            db_path=_path_env("DEAL_DB_PATH", ROOT / "data" / "deals.db"),
            uploads_dir=_path_env("DEAL_UPLOADS_DIR", ROOT / "data" / "uploads"),
            memory_backend=memory_backend,
            hindsight_url=os.environ.get("DEAL_HINDSIGHT_URL", cls.hindsight_url).strip() or cls.hindsight_url,
            hindsight_api_key=os.environ.get("DEAL_HINDSIGHT_API_KEY", "").strip() or None,
            hindsight_bank=os.environ.get("DEAL_HINDSIGHT_BANK", cls.hindsight_bank).strip() or cls.hindsight_bank,
            hindsight_lessons_bank=os.environ.get("DEAL_HINDSIGHT_LESSONS_BANK", cls.hindsight_lessons_bank).strip()
            or cls.hindsight_lessons_bank,
            lessons_enabled=_bool_env("DEAL_LESSONS", True),
            similar_deals_k=_int_env("DEAL_SIMILAR_DEALS_K", cls.similar_deals_k),
            retrieval_mode=retrieval_mode,
            min_shared_keys=_int_env("DEAL_MIN_SHARED_KEYS", cls.min_shared_keys),
            recall_order=_recall_order_env(),
        )
