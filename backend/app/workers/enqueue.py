"""Helpers for enqueueing Celery jobs safely."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.job import Job, JobStatus

logger = logging.getLogger(__name__)


async def enqueue_video_job(
    db: AsyncSession,
    job: Job,
    max_attempts: int = 3,
    backoff_s: float = 0.5,
) -> Job:
    """Enqueue a video processing job and persist the task id or failure state."""
    from app.workers.tasks import process_video

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            task = process_video.apply_async(
                args=[str(job.video_id), str(job.id)],
                queue="video_processing",
            )
            job.celery_task_id = task.id
            job.status = JobStatus.QUEUED.value
            await db.commit()
            return job
        except Exception as exc:  # pragma: no cover - depends on broker state
            last_exc = exc
            logger.warning(
                "Failed to enqueue job %s (attempt %d/%d): %s",
                job.id,
                attempt,
                max_attempts,
                exc,
            )
            await asyncio.sleep(backoff_s * attempt)

    job.status = JobStatus.FAILED.value
    job.error_step = "enqueue"
    job.error_message = f"Failed to enqueue job: {last_exc}"
    job.completed_at = datetime.now(timezone.utc)
    await db.commit()
    return job


async def requeue_orphaned_jobs(db: AsyncSession) -> int:
    """Requeue queued jobs that never got a celery task id."""
    result = await db.execute(
        select(Job).where(
            Job.status == JobStatus.QUEUED.value,
            Job.celery_task_id.is_(None),
        )
    )
    jobs = result.scalars().all()
    if not jobs:
        return 0

    requeued = 0
    for job in jobs:
        await enqueue_video_job(db, job)
        if job.celery_task_id:
            requeued += 1

    return requeued


def _collect_active_task_ids() -> tuple[set[str], bool]:
    from app.workers.celery_app import celery_app

    inspector = celery_app.control.inspect(timeout=2.0)
    if not inspector:
        return set(), False

    workers_present = bool(inspector.ping() or {})

    task_ids: set[str] = set()
    for method_name in ("active", "reserved", "scheduled"):
        method = getattr(inspector, method_name, None)
        if not method:
            continue
        response = method() or {}
        for tasks in response.values():
            for task in tasks or []:
                task_id = task.get("id")
                if task_id:
                    task_ids.add(task_id)
    return task_ids, workers_present


async def recover_stale_processing_jobs(db: AsyncSession) -> int:
    """Requeue processing jobs that have no active Celery task and are stale."""
    settings = get_settings()
    active_task_ids, workers_present = _collect_active_task_ids()
    if not workers_present:
        logger.warning("No Celery workers found; skipping stale job recovery.")
        return 0
    return await recover_stale_processing_jobs_with_grace(
        db,
        stale_minutes=settings.job_stale_minutes,
        active_task_ids=active_task_ids,
    )


async def recover_stale_processing_jobs_with_grace(
    db: AsyncSession,
    stale_minutes: int,
    active_task_ids: set[str] | None = None,
) -> int:
    """Requeue stale processing jobs while preserving completed checkpoint state."""
    if active_task_ids is None:
        active_task_ids, _ = _collect_active_task_ids()

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)
    result = await db.execute(
        select(Job).where(
            Job.status == JobStatus.PROCESSING.value,
            Job.started_at.is_not(None),
            Job.started_at < cutoff,
        )
    )
    jobs = result.scalars().all()
    if not jobs:
        return 0

    recovered = 0
    for job in jobs:
        if job.celery_task_id and job.celery_task_id in active_task_ids:
            continue

        job.status = JobStatus.QUEUED.value
        job.error_message = None
        job.error_step = None
        job.started_at = None
        job.completed_at = None
        job.celery_task_id = None
        job.retry_count += 1
        await db.commit()

        await enqueue_video_job(db, job)
        if job.celery_task_id:
            recovered += 1

    return recovered
