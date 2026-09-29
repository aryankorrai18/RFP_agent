"""SQLite via SQLAlchemy: the source of truth for V1 (design §4, decision V1-D1).

Hindsight only ever holds a searchable copy of approved answers. Every recall result is joined
back to the `answers` table and filtered here before it can reach a draft (PRD D-10).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import Boolean, Float, JSON, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, event, inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))  # past_proposal | rfp
    filename: Mapped[str] = mapped_column(String(300))
    sha256: Mapped[str] = mapped_column(String(64))
    stored_path: Mapped[str] = mapped_column(String(500))
    uploaded_at: Mapped[datetime] = mapped_column(default=utcnow)


class PastProposal(Base):
    __tablename__ = "past_proposals"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    client: Mapped[str | None] = mapped_column(String(200))
    industry: Mapped[str | None] = mapped_column(String(100))
    submitted_on: Mapped[date | None]
    result: Mapped[str] = mapped_column(String(20), default="unknown")  # won|lost|no_decision|unknown (used from V3)
    loss_reason: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="uploaded")  # uploaded|extracting|extracted|confirmed|failed|discarded
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    document: Mapped[Document] = relationship()
    pairs: Mapped[list[Pair]] = relationship(back_populates="proposal", order_by="Pair.order", cascade="all, delete-orphan")


class Pair(Base):
    __tablename__ = "pairs"
    id: Mapped[int] = mapped_column(primary_key=True)
    past_proposal_id: Mapped[int] = mapped_column(ForeignKey("past_proposals.id"))
    order: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(300))
    reference: Mapped[str | None] = mapped_column(String(100))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(String(10), default="pending")  # pending|kept|edited|dropped
    proposal: Mapped[PastProposal] = relationship(back_populates="pairs")


class Answer(Base):
    """The library. A row exists only once a person has approved the answer."""

    __tablename__ = "answers"
    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(10))  # library | project
    past_proposal_id: Mapped[int | None] = mapped_column(ForeignKey("past_proposals.id"))
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"))
    requirement_id: Mapped[int | None] = mapped_column(ForeignKey("requirements.id"))
    client: Mapped[str | None] = mapped_column(String(200))
    industry: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)
    status: Mapped[str] = mapped_column(String(10), default="approved")  # approved | deleted
    superseded_by: Mapped[int | None]  # V3
    # Outbox for the Hindsight copy (design §12.2)
    hindsight_status: Mapped[str] = mapped_column(String(15), default="pending")  # pending|retained|pending_delete|deleted
    hindsight_attempts: Mapped[int] = mapped_column(Integer, default=0)
    hindsight_error: Mapped[str | None] = mapped_column(Text)

    @property
    def code(self) -> str:
        return answer_code(self.id)

    @property
    def live(self) -> bool:
        return self.status == "approved" and self.superseded_by is None


class AnswerStats(Base):
    """Exact, auditable outcome signals used by the V2 deterministic reranker."""

    __tablename__ = "answer_stats"
    answer_id: Mapped[int] = mapped_column(ForeignKey("answers.id"), primary_key=True)
    times_used: Mapped[int] = mapped_column(Integer, default=0)
    accepted: Mapped[int] = mapped_column(Integer, default=0)
    light_edits: Mapped[int] = mapped_column(Integer, default=0)
    length_edits: Mapped[int] = mapped_column(Integer, default=0)
    heavy_edits: Mapped[int] = mapped_column(Integer, default=0)
    rewritten: Mapped[int] = mapped_column(Integer, default=0)
    rejected: Mapped[int] = mapped_column(Integer, default=0)
    rating_total: Mapped[int] = mapped_column(Integer, default=0)
    rating_count: Mapped[int] = mapped_column(Integer, default=0)
    debrief_credit: Mapped[float] = mapped_column(Float, default=0.0)
    outcome_credit: Mapped[float] = mapped_column(Float, default=0.0)
    outdated_signals: Mapped[int] = mapped_column(Integer, default=0)
    suggested_supersede: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)


def answer_code(answer_id: int) -> str:
    return f"ANS-{answer_id:04d}"


def parse_answer_code(code: str) -> int | None:
    if code.startswith("ANS-") and code[4:].isdigit():
        return int(code[4:])
    return None


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    fact_sheet_document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"))
    name: Mapped[str] = mapped_column(String(300))
    client: Mapped[str | None] = mapped_column(String(200))
    industry: Mapped[str | None] = mapped_column(String(100))
    # uploaded|extracting|requirements_extracted|drafting|in_review|approved|exported|failed
    state: Mapped[str] = mapped_column(String(30), default="uploaded")
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    document: Mapped[Document] = relationship(foreign_keys=[document_id])
    fact_sheet_document: Mapped[Document | None] = relationship(foreign_keys=[fact_sheet_document_id])
    requirements: Mapped[list[RequirementRow]] = relationship(
        back_populates="project", order_by="RequirementRow.order", cascade="all, delete-orphan"
    )


class RequirementRow(Base):
    __tablename__ = "requirements"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    order: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(300))
    reference: Mapped[str | None] = mapped_column(String(100))
    question: Mapped[str] = mapped_column(Text)
    mandatory: Mapped[bool | None]
    word_limit: Mapped[int | None]
    project: Mapped[Project] = relationship(back_populates="requirements")
    drafts: Mapped[list[DraftRow]] = relationship(order_by="DraftRow.version", cascade="all, delete-orphan")

    @property
    def code(self) -> str:
        return f"REQ-{self.order:03d}"


class DraftRow(Base):
    __tablename__ = "drafts"
    __table_args__ = (UniqueConstraint("job_id", "requirement_id", name="one_draft_per_job"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    requirement_id: Mapped[int] = mapped_column(ForeignKey("requirements.id"))
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"))  # null for single regenerations
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(10))  # drafted | needs_sme | failed
    answer: Mapped[str] = mapped_column(Text, default="")
    claims: Mapped[list] = mapped_column(JSON, default=list)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    unsupported_claims: Mapped[list] = mapped_column(JSON, default=list)
    invalid_citations: Mapped[list] = mapped_column(JSON, default=list)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    sme_question: Mapped[str | None] = mapped_column(Text)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    retrieved: Mapped[list] = mapped_column(JSON, default=list)  # [{id, rank, final, semantic, reranker, keyword}]
    retrieval_warning: Mapped[str | None] = mapped_column(Text)
    instructions: Mapped[str | None] = mapped_column(Text)  # regenerate instructions, if any
    prompt_version: Mapped[str] = mapped_column(String(10))
    model: Mapped[str | None] = mapped_column(String(100))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    reviews: Mapped[list[Review]] = relationship(order_by="Review.id", cascade="all, delete-orphan")


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    draft_id: Mapped[int] = mapped_column(ForeignKey("drafts.id"))
    action: Mapped[str] = mapped_column(String(10))  # accepted|edited|rewritten|rejected
    final_text: Mapped[str | None] = mapped_column(Text)
    edit_distance: Mapped[float | None]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class ReviewFeedback(Base):
    __tablename__ = "review_feedback"
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id"), primary_key=True)
    reason_tags: Mapped[list] = mapped_column(JSON, default=list)
    rating: Mapped[int | None]


class ClientPreference(Base):
    __tablename__ = "client_preferences"
    __table_args__ = (UniqueConstraint("client", "key", name="one_preference_per_client_key"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    client: Mapped[str] = mapped_column(String(200))
    key: Mapped[str] = mapped_column(String(50))
    evidence_count: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)


class MemoryEvent(Base):
    __tablename__ = "memory_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(40))
    answer_id: Mapped[int | None] = mapped_column(ForeignKey("answers.id"))
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"))
    detail: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Lesson(Base):
    """A plain-language lesson for the Hindsight lessons bank, and its outbox state.

    SQLite keeps the exact record; Hindsight turns lessons into facts and observations that drive
    ranking, client briefs and the playbook. `key` makes collection idempotent."""

    __tablename__ = "lessons"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(80), unique=True)
    answer_id: Mapped[int | None] = mapped_column(ForeignKey("answers.id"))
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"))
    signal: Mapped[str] = mapped_column(String(10))  # positive | negative | neutral
    text: Mapped[str] = mapped_column(Text)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    happened_at: Mapped[datetime] = mapped_column(default=utcnow)
    hindsight_status: Mapped[str] = mapped_column(String(15), default="pending")  # pending | retained
    hindsight_attempts: Mapped[int] = mapped_column(Integer, default=0)
    hindsight_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class ProjectBrief(Base):
    """The latest Hindsight Reflect brief for a project's client, cached to save Hindsight credits."""

    __tablename__ = "project_briefs"
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    based_on: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class ProjectOutcome(Base):
    __tablename__ = "project_outcomes"
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), primary_key=True)
    result: Mapped[str] = mapped_column(String(20))
    loss_reason: Mapped[str | None] = mapped_column(String(200))
    decided_at: Mapped[date | None]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Debrief(Base):
    __tablename__ = "debriefs"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    section: Mapped[str | None] = mapped_column(String(300))
    score: Mapped[float | None] = mapped_column(Float)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(30))  # extract_pairs | extract_requirements | draft_all | compare_memory
    target_id: Mapped[int] = mapped_column(Integer)  # past_proposal_id or project_id
    status: Mapped[str] = mapped_column(String(15), default="queued")  # queued|running|completed|failed|interrupted
    payload: Mapped[dict] = mapped_column(JSON, default=dict)  # e.g. {"requirement_ids": [...]}: survives a restart
    done: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    warning: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


class ComparisonDraft(Base):
    """A draft made only to compare memory states. It is never shown for review, never exported and
    never feeds the learning signals; it keeps the retrieved answers and lesson evidence behind it."""

    __tablename__ = "comparison_drafts"
    __table_args__ = (UniqueConstraint("job_id", "requirement_id", "arm", name="one_comparison_draft"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    requirement_id: Mapped[int] = mapped_column(ForeignKey("requirements.id"))
    arm: Mapped[str] = mapped_column(String(12))  # none | plain | outcome | hindsight
    status: Mapped[str] = mapped_column(String(10))  # drafted | needs_sme | failed
    answer: Mapped[str] = mapped_column(Text, default="")
    sources: Mapped[list] = mapped_column(JSON, default=list)
    unsupported_claims: Mapped[list] = mapped_column(JSON, default=list)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    retrieved: Mapped[list] = mapped_column(JSON, default=list)
    warning: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(100))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class JudgeVerdict(Base):
    """V4: a judge model's blind verdict on one before/after pair, in one presentation order.
    `winner` is already un-blinded (plain | hindsight | tie). Only successful verdicts are stored."""

    __tablename__ = "judge_verdicts"
    __table_args__ = (UniqueConstraint("comparison_job_id", "requirement_id", "order", "judge_model", "prompt_version",
                                       name="one_judge_verdict"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))  # the judging run that made it
    comparison_job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    requirement_id: Mapped[int] = mapped_column(ForeignKey("requirements.id"))
    order: Mapped[str] = mapped_column(String(20))  # plain_first | hindsight_first: which arm was shown as A
    judge_model: Mapped[str] = mapped_column(String(100))  # the model asked for
    served_model: Mapped[str | None] = mapped_column(String(100))  # the model version that answered
    prompt_version: Mapped[str] = mapped_column(String(20))
    winner: Mapped[str] = mapped_column(String(10))
    reason: Mapped[str] = mapped_column(Text, default="")
    scores: Mapped[dict] = mapped_column(JSON, default=dict)  # {"plain": {...}, "hindsight": {...}}
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class HumanVerdict(Base):
    """V4: a person's blind spot-check of one before/after pair. `winner` is un-blinded."""

    __tablename__ = "human_verdicts"
    __table_args__ = (UniqueConstraint("comparison_job_id", "requirement_id", name="one_human_verdict"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    comparison_job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    requirement_id: Mapped[int] = mapped_column(ForeignKey("requirements.id"))
    shown_first: Mapped[str] = mapped_column(String(10))  # the arm shown as A
    winner: Mapped[str] = mapped_column(String(10))  # plain | hindsight | tie
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


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
            cursor.execute("PRAGMA journal_mode=WAL")  # readers don't block the drafting job's writes
            cursor.close()

        Base.metadata.create_all(self.engine)
        # create_all does not add columns to an existing SQLite table. This small additive migration
        # keeps hackathon databases usable after per-project fact sheets were introduced.
        columns = {column["name"] for column in inspect(self.engine).get_columns("projects")}
        if "fact_sheet_document_id" not in columns:
            with self.engine.begin() as connection:
                connection.exec_driver_sql(
                    "ALTER TABLE projects ADD COLUMN fact_sheet_document_id INTEGER REFERENCES documents(id)"
                )
        self.session: sessionmaker[Session] = sessionmaker(self.engine, expire_on_commit=False)
