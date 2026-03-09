"""Celery application configuration."""

import asyncio
import logging

from celery import Celery
from celery.signals import worker_ready

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

celery_app = Celery(
    "video_summarizer",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,
    task_time_limit=14400,  # 4 hours max per task
    task_soft_time_limit=14000,

    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time per worker
    worker_concurrency=settings.batch_concurrent_jobs,

    # Result backend
    result_expires=86400,  # Results expire after 24 hours

    # Task routing
    task_routes={
        "app.workers.tasks.process_video": {"queue": "video_processing"},
        "app.workers.tasks.process_batch": {"queue": "batch_processing"},
    },

    # Priority queues
    task_default_queue="default",
    task_queues={
        "video_processing": {
            "exchange": "video_processing",
            "routing_key": "video_processing",
        },
        "batch_processing": {
            "exchange": "batch_processing",
            "routing_key": "batch_processing",
        },
    },
)


@worker_ready.connect
def _recover_jobs_on_worker_ready(**_kwargs) -> None:
    """Requeue interrupted jobs as soon as a worker is available again."""
    from app.database import get_worker_async_session_maker
    from app.workers.enqueue import requeue_orphaned_jobs, recover_stale_processing_jobs

    async def _recover() -> None:
        worker_async_session_maker = get_worker_async_session_maker()
        async with worker_async_session_maker() as session:
            requeued = await requeue_orphaned_jobs(session)
            if requeued:
                logger.info("Worker startup requeued %d orphaned job(s)", requeued)
            recovered = await recover_stale_processing_jobs(session)
            if recovered:
                logger.info("Worker startup recovered %d stale processing job(s)", recovered)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_recover())
    except Exception as exc:  # pragma: no cover - startup best effort
        logger.warning("Worker startup recovery failed: %s", exc)
    finally:
        asyncio.set_event_loop(None)
        loop.close()
