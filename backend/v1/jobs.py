"""Background jobs inside the app process (decision V1-D4; restart rules in design §12.1).

    QUEUED → RUNNING → COMPLETED | FAILED | CANCELLED (stopped by a person)
    RUNNING at startup (the process died) → INTERRUPTED → resumed automatically

Handlers must be safe to run again: extraction replaces its results in one transaction, and
draft_all skips requirements that already have a draft for the same job.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from sqlalchemy import select

from .db import Job, utcnow

if TYPE_CHECKING:
    from .context import V1Context

log = logging.getLogger(__name__)

Handler = Callable[["V1Context", int], Awaitable[None]]
# Called after a person stops a job, to move its project or proposal to a state they can act on.
CancelHandler = Callable[["V1Context", int], None]
FINAL_STATUSES = ("completed", "failed", "cancelled")
STOPPABLE_STATUSES = ("queued", "running", "interrupted")


class JobFailed(Exception):
    """An expected failure with a message the person can act on (bad file, model refusal…)."""


class JobRunner:
    def __init__(self, ctx: V1Context):
        self.ctx = ctx
        self.handlers: dict[str, Handler] = {}
        self.cancel_handlers: dict[str, CancelHandler] = {}
        self._tasks: dict[int, asyncio.Task] = {}
        self._stopped: set[int] = set()

    def register(self, kind: str, handler: Handler, on_cancel: CancelHandler | None = None) -> None:
        self.handlers[kind] = handler
        if on_cancel is not None:
            self.cancel_handlers[kind] = on_cancel

    def cancel(self, job_id: int) -> Job:
        """Stop a queued or running job at its next await (e.g. mid model call). Work already
        committed is kept; the job's cancel handler moves its target to an actionable state."""
        from ..core import PipelineError

        with self.ctx.db.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise PipelineError("not_found", f"No job {job_id}.", 404)
            if job.status not in STOPPABLE_STATUSES:
                raise PipelineError("invalid_state", f"This job has already {job.status}.", 409)
            kind = job.kind
        self._stopped.add(job_id)
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        self._finish(job_id, "cancelled", "Stopped by you.")
        on_cancel = self.cancel_handlers.get(kind)
        if on_cancel is not None:
            on_cancel(self.ctx, job_id)
        with self.ctx.db.session() as session:
            return session.get(Job, job_id)

    def submit(self, kind: str, target_id: int, payload: dict | None = None) -> Job:
        if kind not in self.handlers:
            raise ValueError(f"No handler registered for job kind {kind!r}")
        with self.ctx.db.session() as session:
            job = Job(kind=kind, target_id=target_id, payload=payload or {}, status="queued")
            session.add(job)
            session.commit()
        self.start(job.id)
        return job

    def busy(self) -> bool:
        """True while any job in this process is still queued or running."""
        return any(not task.done() for task in self._tasks.values())

    def start(self, job_id: int) -> None:
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            return
        self._tasks[job_id] = asyncio.get_running_loop().create_task(self._run(job_id))

    async def resume_interrupted(self) -> list[int]:
        """Run at startup: any job still marked running belonged to a dead process."""
        with self.ctx.db.session() as session:
            jobs = session.scalars(select(Job).where(Job.status.in_(("running", "queued", "interrupted")))).all()
            for job in jobs:
                if job.status == "running":
                    job.status = "interrupted"
            session.commit()
            ids = [job.id for job in jobs]
        for job_id in ids:
            self.start(job_id)
        return ids

    async def wait(self, job_id: int | None = None) -> None:
        """For tests and shutdown: wait for one job, or for all running jobs."""
        tasks = [self._tasks[job_id]] if job_id is not None and job_id in self._tasks else list(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, job_id: int) -> None:
        with self.ctx.db.session() as session:
            job = session.get(Job, job_id)
            if job is None or job.status in FINAL_STATUSES:
                return
            handler = self.handlers.get(job.kind)
            job.status = "running"
            job.started_at = job.started_at or utcnow()
            job.error = None
            session.commit()
        try:
            if handler is None:
                raise JobFailed(f"No handler for job kind {job.kind!r}")
            await handler(self.ctx, job_id)
            self._finish(job_id, "completed", None)
        except asyncio.CancelledError:
            if job_id in self._stopped:
                return  # stopped by a person: cancel() has already recorded the outcome
            raise  # shutdown: the job stays running and is resumed at the next startup
        except JobFailed as exc:
            self._finish(job_id, "failed", str(exc))
        except Exception as exc:  # an unexpected bug must still leave a visible, final state
            log.exception("Job %s failed unexpectedly", job_id)
            self._finish(job_id, "failed", f"Unexpected error: {exc}")

    def _finish(self, job_id: int, status: str, error: str | None) -> None:
        with self.ctx.db.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            job.status = status
            job.error = error
            job.finished_at = utcnow()
            session.commit()
