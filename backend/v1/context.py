"""Everything V1's routes and jobs share: the database, the memory, the job runner, and access to
the current settings and model client (settings are re-read so a key saved in .env takes effect)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import Settings
from ..llm import LLM
from .db import Database
from .jobs import JobRunner
from .lessons import HindsightLessons, LessonSyncReport, LessonsMemory, sync_lessons
from .memory import HindsightMemory, Memory
from .sync import SyncReport, sync_outbox

log = logging.getLogger(__name__)

SYNC_INTERVAL_SECONDS = 30
SYNC_MAX_INTERVAL_SECONDS = 300


@dataclass
class V1Context:
    db: Database
    memory: Memory
    settings_provider: Callable[[], Settings]
    llm_provider: Callable[[Settings], LLM]
    sync_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lessons: LessonsMemory | None = None
    lessons_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_lesson_sync: LessonSyncReport | None = field(default=None, init=False)
    jobs: JobRunner = field(init=False)
    _background: set[asyncio.Task] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.jobs = JobRunner(self)
        from . import experiment, judge, library, projects  # register handlers; imported here to avoid a cycle

        library.register(self.jobs)
        projects.register(self.jobs)
        experiment.register(self.jobs)
        judge.register(self.jobs)

    @property
    def settings(self) -> Settings:
        return self.settings_provider()

    @property
    def llm(self) -> LLM:
        return self.llm_provider(self.settings)

    async def sync(self) -> SyncReport:
        report = await sync_outbox(self.db, self.memory, self.sync_lock)
        await self.sync_lessons()
        return report

    async def sync_lessons(self) -> LessonSyncReport | None:
        """Describe new outcomes, reviews and debriefs as lessons and send them to Hindsight."""
        if self.lessons is None:
            return None
        try:
            self.last_lesson_sync = await sync_lessons(self.db, self.lessons, self.lessons_lock)
        except Exception:  # lessons must never break answer sync or the request that triggered it
            log.exception("Hindsight lessons sync failed")
        return self.last_lesson_sync

    def schedule_sync(self) -> None:
        """Push pending answers to Hindsight now, without making the request wait for it."""
        try:
            task = asyncio.get_running_loop().create_task(self.sync())
        except RuntimeError:  # no running loop (e.g. called from a sync test); the periodic sync catches up
            return
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def startup(self) -> None:
        from .library import backfill_historical_outcomes

        backfill_historical_outcomes(self.db)
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
                delay = (
                    min(delay * 2, SYNC_MAX_INTERVAL_SECONDS)
                    if report.failed
                    else SYNC_INTERVAL_SECONDS
                )
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
    return V1Context(
        db=Database(settings.db_path),
        memory=HindsightMemory(settings.hindsight_url, settings.hindsight_bank, api_key=settings.hindsight_api_key),
        lessons=(HindsightLessons(settings.hindsight_url, settings.hindsight_lessons_bank,
                                  api_key=settings.hindsight_api_key) if settings.lessons_enabled else None),
        settings_provider=settings_provider,
        llm_provider=llm_provider,
    )
