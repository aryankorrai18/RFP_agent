"""Shared runtime context: database, memory, job runner, and model access for
the current settings and model client (settings are re-read so a key saved in .env takes effect)."""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...config import Settings, resolve_memory_backend
from ...providers.base import LLM
from .db import Database
from .jobs import JobRunner

if TYPE_CHECKING:
    from .lessons import LessonSyncReport, LessonsMemory
    from .memory import Memory
    from .sync import SyncReport

log = logging.getLogger(__name__)

SYNC_INTERVAL_SECONDS = 30
SYNC_MAX_INTERVAL_SECONDS = 300

# Job kinds and the module functions that run them. Each handler is `async def run(ctx, job_id)`.
JOB_HANDLERS = {
    "extract_signals": ("signals", "run_extract_signals"),
    "brief": ("briefs", "run_brief"),
    "compare_memory": ("experiment", "run_comparison"),
}


@dataclass
class V1Context:
    db: Database
    memory: Memory
    settings_provider: Callable[[], Settings]
    llm_provider: Callable[[Settings], LLM]
    sync_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lessons: LessonsMemory | None = None
    lessons_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    workspace_id: str | None = None  # set for a context opened by name; None for the active workspace's own
    memory_backend: str = "hindsight"
    last_lesson_sync: Any = field(default=None, init=False)
    jobs: JobRunner = field(init=False)
    _background: set[asyncio.Task] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.jobs = JobRunner(self)
        for kind, (module, attr) in JOB_HANDLERS.items():
            self.jobs.register(kind, getattr(importlib.import_module(f"{__package__}.{module}"), attr))

    @property
    def settings(self) -> Settings:
        return self.settings_provider()

    @property
    def llm(self) -> LLM:
        from .usage import MeteredLLM

        return MeteredLLM(self.llm_provider(self.settings), self.db)

    async def sync(self) -> SyncReport:
        from .sync import sync_outbox

        report = await sync_outbox(self.db, self.memory, self.sync_lock)
        await self.sync_lessons()
        return report

    async def sync_lessons(self) -> LessonSyncReport | None:
        """Describe new outcomes as lessons and send them to Hindsight."""
        if self.lessons is None:
            return None
        from .lessons import sync_lessons

        try:
            self.last_lesson_sync = await sync_lessons(self.db, self.lessons, self.lessons_lock)
        except Exception:  # lessons must never break the deal sync or the request that triggered it
            log.exception("Hindsight lessons sync failed")
        return self.last_lesson_sync

    def schedule_sync(self) -> None:
        """Push pending documents to Hindsight now, without making the request wait for it."""
        try:
            task = asyncio.get_running_loop().create_task(self.sync())
        except RuntimeError:  # no running loop (e.g. called from a sync test); the periodic sync catches up
            return
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def startup(self) -> None:
        await self.jobs.resume_interrupted()
        self.schedule_sync()
        task = asyncio.get_running_loop().create_task(self._periodic_sync())
        self._background.add(task)

    async def _periodic_sync(self) -> None:
        delay = SYNC_INTERVAL_SECONDS
        while True:
            await asyncio.sleep(delay)
            try:
                report = await self.sync()
                delay = min(delay * 2, SYNC_MAX_INTERVAL_SECONDS) if report.failed else SYNC_INTERVAL_SECONDS
            except Exception:  # a sync failure must never stop the loop
                log.exception("Periodic Hindsight sync failed")
                delay = min(delay * 2, SYNC_MAX_INTERVAL_SECONDS)

    async def shutdown(self) -> None:
        for task in list(self._background):
            task.cancel()
        await self.memory.close()
        if self.lessons is not None:
            await self.lessons.close()


def build_context(settings_provider: Callable[[], Settings], llm_provider: Callable[[Settings], LLM]) -> V1Context:
    settings = settings_provider()
    backend = resolve_memory_backend(settings)
    if backend == "local":
        from .memory_local import LocalLessons, LocalMemory

        path = settings.db_path.parent / "memory.db"
        memory: Memory = LocalMemory(path, settings.hindsight_bank)
        lessons: LessonsMemory | None = LocalLessons(path, settings.hindsight_lessons_bank) if settings.lessons_enabled else None
    else:
        from .lessons import HindsightLessons
        from .memory import HindsightMemory

        memory = HindsightMemory(settings.hindsight_url, settings.hindsight_bank, api_key=settings.hindsight_api_key,
                                 order=settings.recall_order)
        lessons = (HindsightLessons(settings.hindsight_url, settings.hindsight_lessons_bank,
                                    api_key=settings.hindsight_api_key) if settings.lessons_enabled else None)
    return V1Context(
        db=Database(settings.db_path), memory=memory, lessons=lessons, memory_backend=backend,
        settings_provider=settings_provider, llm_provider=llm_provider,
    )
