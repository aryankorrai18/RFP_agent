"""SQLite via SQLAlchemy: the application's exact system of record.

Hindsight only ever holds a searchable copy: one summary per CLOSED deal and the interactions of
open deals (bank 1), plus outcome lessons (bank 2). Every recall result is joined back to the tables
below and filtered here before it can reach a brief.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


# ---- Vocabulary (the structured keys the gate and the lessons are built on) ----------------------

SEGMENTS = ("smb", "mid_market", "enterprise")
STAGES = ("discovery", "evaluation", "proposal", "negotiation", "closed")
RESULTS = ("open", "won", "lost")
INTERACTION_KINDS = ("email", "call_note", "meeting", "crm_note")
STANCES = ("champion", "supporter", "neutral", "blocker")
OBJECTION_STATUSES = ("raised", "addressed", "unresolved")
OBJECTION_TYPES = (
    "sso", "security_review", "pricing", "integration", "timeline", "legal_terms",
    "data_residency", "support", "feature_gap",
)
LOSS_REASONS = (
    "competitor", "price", "no_decision", "champion_left", "security_compliance", "feature_gap", "timing",
    "unresolved_objection",
)
# A loss the plays used on the deal could plausibly have changed. Price, no-decision, timing, champion
# left and plain competitor losses give plays no credit either way (the outcomes.py rule).
QUALITY_LOSSES = frozenset({"security_compliance", "feature_gap", "unresolved_objection"})
PLAY_CATEGORIES = ("security", "stakeholder", "commercial", "proof", "process")


class Base(DeclarativeBase):
    pass


def deal_code(deal_id: int) -> str:
    return f"D-{deal_id:03d}"


def parse_deal_code(code: str) -> int | None:
    if code.startswith("D-") and code[2:].isdigit():
        return int(code[2:])
    return None


def interaction_code(interaction_id: int) -> str:
    return f"INT-{interaction_id:04d}"


def parse_interaction_code(code: str) -> int | None:
    if code.startswith("INT-") and code[4:].isdigit():
        return int(code[4:])
    return None


class Document(Base):
    """An uploaded file, stored once by content hash (see storage.py)."""

    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))  # interaction_file | deal_pack
    filename: Mapped[str] = mapped_column(String(300))
    sha256: Mapped[str] = mapped_column(String(64))
    stored_path: Mapped[str] = mapped_column(String(500))
    uploaded_at: Mapped[datetime] = mapped_column(default=utcnow)


class Deal(Base):
    __tablename__ = "deals"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(300))
    account: Mapped[str] = mapped_column(String(200))
    industry: Mapped[str | None] = mapped_column(String(100))
    segment: Mapped[str | None] = mapped_column(String(20))  # SEGMENTS
    amount: Mapped[int | None]  # annual contract value, whole currency units
    stage: Mapped[str] = mapped_column(String(20), default="discovery")  # STAGES
    owner: Mapped[str | None] = mapped_column(String(200))  # the account executive
    opened_on: Mapped[date | None]
    closed_on: Mapped[date | None]
    result: Mapped[str] = mapped_column(String(10), default="open")  # RESULTS
    loss_reason: Mapped[str | None] = mapped_column(String(40))  # LOSS_REASONS, only when lost
    status: Mapped[str] = mapped_column(String(10), default="active")  # active | deleted
    # none | extracting | ready | failed: whether DealSignals exist (seeded deals are ready on arrival)
    signals_status: Mapped[str] = mapped_column(String(12), default="none")
    signals_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)
    # Outbox for the closed-deal summary document in the interactions bank (D-xxx).
    hindsight_status: Mapped[str] = mapped_column(String(15), default="pending")  # pending|retained|pending_delete|deleted
    hindsight_attempts: Mapped[int] = mapped_column(Integer, default=0)
    hindsight_error: Mapped[str | None] = mapped_column(Text)
    stakeholders: Mapped[list[Stakeholder]] = relationship(
        back_populates="deal", order_by="Stakeholder.id", cascade="all, delete-orphan"
    )
    interactions: Mapped[list[Interaction]] = relationship(
        back_populates="deal", order_by="(Interaction.occurred_on, Interaction.id)", cascade="all, delete-orphan"
    )
    signals: Mapped[DealSignals | None] = relationship(back_populates="deal", uselist=False, cascade="all, delete-orphan")

    @property
    def code(self) -> str:
        return deal_code(self.id)

    @property
    def is_open(self) -> bool:
        return self.result == "open"

    @property
    def live(self) -> bool:
        return self.status == "active"


class Stakeholder(Base):
    __tablename__ = "stakeholders"
    id: Mapped[int] = mapped_column(primary_key=True)
    deal_id: Mapped[int] = mapped_column(ForeignKey("deals.id"))
    name: Mapped[str] = mapped_column(String(200))
    title: Mapped[str | None] = mapped_column(String(200))
    stance: Mapped[str] = mapped_column(String(12), default="neutral")  # STANCES
    engaged: Mapped[bool] = mapped_column(Boolean, default=True)  # replied or met in the last 30 days
    economic_buyer: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str | None] = mapped_column(Text)
    deal: Mapped[Deal] = relationship(back_populates="stakeholders")


class Interaction(Base):
    """An email, call note, meeting note or CRM note. The evidence a brief cites as INT-xxxx."""

    __tablename__ = "interactions"
    __table_args__ = (UniqueConstraint("deal_id", "sha256", name="one_interaction_per_content"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    deal_id: Mapped[int] = mapped_column(ForeignKey("deals.id"))
    kind: Mapped[str] = mapped_column(String(12))  # INTERACTION_KINDS
    occurred_on: Mapped[date]
    author: Mapped[str | None] = mapped_column(String(200))
    subject: Mapped[str | None] = mapped_column(String(300))
    text: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)
    # Outbox for the interaction document in the interactions bank (INT-xxxx). Only open deals and
    # deals of an account with an open deal are retained, so unrelated closed history stays summary-only.
    hindsight_status: Mapped[str] = mapped_column(String(15), default="pending")  # pending|retained|pending_delete|deleted|skipped
    hindsight_attempts: Mapped[int] = mapped_column(Integer, default=0)
    hindsight_error: Mapped[str | None] = mapped_column(Text)
    deal: Mapped[Deal] = relationship(back_populates="interactions")

    @property
    def code(self) -> str:
        return interaction_code(self.id)


class DealSignals(Base):
    """The structured facts the gate and the lessons are built from. Seeded deals carry them already
    (so seeding makes no model call); uploaded deals get them from one extraction call."""

    __tablename__ = "deal_signals"
    deal_id: Mapped[int] = mapped_column(ForeignKey("deals.id"), primary_key=True)
    # [{"type": OBJECTION_TYPES, "text": str, "status": OBJECTION_STATUSES, "first_seen_on": "YYYY-MM-DD",
    #   "evidence": ["INT-0001", ...]}]
    objections: Mapped[list] = mapped_column(JSON, default=list)
    competitors: Mapped[list] = mapped_column(JSON, default=list)  # ["Brightline", ...]
    pricing: Mapped[dict] = mapped_column(JSON, default=dict)  # {"discount_requested": bool, "notes": str}
    # [{"text": str, "owner": str, "due_on": "YYYY-MM-DD"|None, "status": "open"|"kept"|"missed", "evidence": [...]}]
    promises: Mapped[list] = mapped_column(JSON, default=list)
    plays_used: Mapped[list] = mapped_column(JSON, default=list)  # ["PLAY-01", ...] what the team actually did
    source: Mapped[str] = mapped_column(String(10), default="seed")  # seed | extracted | manual
    input_hash: Mapped[str | None] = mapped_column(String(64))  # sha256 of the interactions it was read from
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)
    deal: Mapped[Deal] = relationship(back_populates="signals")


class Play(Base):
    """The catalogue of next steps a team can take (PLAY-xx). Seeded; not edited in the UI."""

    __tablename__ = "plays"
    code: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(20))  # PLAY_CATEGORIES
    addresses: Mapped[list] = mapped_column(JSON, default=list)  # OBJECTION_TYPES this play is meant to resolve


class PlayStats(Base):
    """Exact, auditable counts per play. Shown as counts; credit only breaks ties (n is small)."""

    __tablename__ = "play_stats"
    play_code: Mapped[str] = mapped_column(ForeignKey("plays.code"), primary_key=True)
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    won: Mapped[int] = mapped_column(Integer, default=0)
    lost_quality: Mapped[int] = mapped_column(Integer, default=0)
    lost_other: Mapped[int] = mapped_column(Integer, default=0)
    outcome_credit: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)


class Brief(Base):
    """A generated deal brief and the evidence behind it. `content` is the validated DealBrief JSON."""

    __tablename__ = "briefs"
    id: Mapped[int] = mapped_column(primary_key=True)
    deal_id: Mapped[int] = mapped_column(ForeignKey("deals.id"))
    mode: Mapped[str] = mapped_column(String(10))  # none | longctx | similar | hindsight (config.RETRIEVAL_MODES)
    status: Mapped[str] = mapped_column(String(10), default="ready")  # ready | failed
    content: Mapped[dict] = mapped_column(JSON, default=dict)
    flags: Mapped[list] = mapped_column(JSON, default=list)  # validation findings (unknown citation, uncited claim...)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)  # Recommendations as a dict: similar deals, plays, warnings
    memory_state: Mapped[str] = mapped_column(String(64), default="")  # hash of the memory the brief was made from
    prompt_version: Mapped[str] = mapped_column(String(20), default="")
    prompt_hash: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str | None] = mapped_column(String(100))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"))  # set for comparison arms
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class DealReflection(Base):
    """What the lessons bank's Reflect said about deals like this one, cached so reading it is free."""

    __tablename__ = "deal_reflections"
    deal_id: Mapped[int] = mapped_column(ForeignKey("deals.id"), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    based_on: Mapped[list] = mapped_column(JSON, default=list)
    backend: Mapped[str] = mapped_column(String(12), default="hindsight")  # hindsight | local
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class MemoryEvent(Base):
    """The learning journal: what the app remembered, in order."""

    __tablename__ = "memory_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(40))  # outcome_recorded | play_credit | deal_added | signals_extracted ...
    deal_id: Mapped[int | None] = mapped_column(ForeignKey("deals.id"))
    play_code: Mapped[str | None] = mapped_column(String(10))
    detail: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class ModelUsage(Base):
    """One row per model call made for this workspace: what ran, which model, how many tokens. Written by
    usage.MeteredLLM, read by GET /v1/usage. Counting starts when this table was added; earlier calls were not recorded."""

    __tablename__ = "model_usage"
    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(default=utcnow)
    purpose: Mapped[str] = mapped_column(String(60))  # the model method that was called (e.g. draft_answer, brief)
    model: Mapped[str | None] = mapped_column(String(100))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)  # False: the call failed (it used no tokens we know of)


class Lesson(Base):
    """A plain-language lesson for the Hindsight lessons bank, and its outbox state.

    SQLite keeps the exact record; Hindsight turns lessons into facts and observations for the Memory
    page and the playbook. `key` makes collection idempotent."""

    __tablename__ = "lessons"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(80), unique=True)  # e.g. outcome:D-004, play:D-004:PLAY-03
    deal_id: Mapped[int | None] = mapped_column(ForeignKey("deals.id"))
    play_code: Mapped[str | None] = mapped_column(String(10))
    signal: Mapped[str] = mapped_column(String(10))  # positive | negative | neutral
    text: Mapped[str] = mapped_column(Text)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    happened_at: Mapped[datetime] = mapped_column(default=utcnow)
    hindsight_status: Mapped[str] = mapped_column(String(15), default="pending")  # pending | retained
    hindsight_attempts: Mapped[int] = mapped_column(Integer, default=0)
    hindsight_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(30))  # extract_signals | brief | compare_memory
    target_id: Mapped[int] = mapped_column(Integer)  # the deal id
    status: Mapped[str] = mapped_column(String(15), default="queued")  # queued|running|completed|failed|cancelled|interrupted
    payload: Mapped[dict] = mapped_column(JSON, default=dict)  # survives a restart
    done: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    warning: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


class Database:
    def __init__(self, path: Path | str):
        url = "sqlite://" if str(path) == ":memory:" else f"sqlite:///{Path(path).as_posix()}"
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(self.engine, "connect")
        def _pragmas(dbapi_connection, _record):  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")  # readers don't block a running job's writes
            cursor.close()

        Base.metadata.create_all(self.engine)
        self.session: sessionmaker[Session] = sessionmaker(self.engine, expire_on_commit=False)
