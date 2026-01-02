"""Celery application configuration."""

from celery import Celery

from app.config import get_settings

settings = get_settings()

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
