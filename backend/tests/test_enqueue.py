import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.models.job import Job, JobStatus
from app.workers import enqueue as enqueue_module


class DummySession:
    def __init__(self, results=None):
        self.results = results or []
        self.commits = 0

    async def execute(self, _query):
        return DummyResult(self.results)

    async def commit(self):
        self.commits += 1


class DummyResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return self

    def all(self):
        return list(self._items)


@pytest.mark.asyncio
async def test_enqueue_video_job_success(monkeypatch):
    job = Job(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        status=JobStatus.QUEUED.value,
    )

    class DummyTask:
        def apply_async(self, args=None, queue=None):
            return type("Task", (), {"id": "task-123"})()

    monkeypatch.setattr("app.workers.tasks.process_video", DummyTask())

    db = DummySession()
    result = await enqueue_module.enqueue_video_job(db, job, max_attempts=1, backoff_s=0)

    assert result.celery_task_id == "task-123"
    assert result.status == JobStatus.QUEUED.value
    assert db.commits == 1


@pytest.mark.asyncio
async def test_enqueue_video_job_failure(monkeypatch):
    job = Job(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        status=JobStatus.QUEUED.value,
    )

    class DummyTask:
        def apply_async(self, args=None, queue=None):
            raise RuntimeError("broker down")

    monkeypatch.setattr("app.workers.tasks.process_video", DummyTask())

    db = DummySession()
    result = await enqueue_module.enqueue_video_job(db, job, max_attempts=1, backoff_s=0)

    assert result.status == JobStatus.FAILED.value
    assert result.error_step == "enqueue"
    assert result.completed_at is not None
    assert "broker down" in (result.error_message or "")


@pytest.mark.asyncio
async def test_requeue_orphaned_jobs(monkeypatch):
    job = Job(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        status=JobStatus.QUEUED.value,
        celery_task_id=None,
    )

    async def fake_enqueue(db, job_obj, **_kwargs):
        job_obj.celery_task_id = "task-abc"
        job_obj.status = JobStatus.QUEUED.value
        return job_obj

    monkeypatch.setattr(enqueue_module, "enqueue_video_job", fake_enqueue)

    db = DummySession(results=[job])
    requeued = await enqueue_module.requeue_orphaned_jobs(db)

    assert requeued == 1
    assert job.celery_task_id == "task-abc"


@pytest.mark.asyncio
async def test_recover_stale_processing_jobs(monkeypatch):
    step_progress = {
        "audio_extraction": {
            "progress": 100.0,
            "step_index": 1,
            "step_count": 8,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
        "transcription": {
            "progress": 25.0,
            "step_index": 2,
            "step_count": 8,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    stale_job = Job(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        status=JobStatus.PROCESSING.value,
        celery_task_id="old-task",
        started_at=datetime.now(timezone.utc) - timedelta(hours=1),
        current_step="transcription",
        progress=18.0,
        step_progress=step_progress,
        retry_count=0,
    )
    active_job = Job(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        status=JobStatus.PROCESSING.value,
        celery_task_id="active-task",
        started_at=datetime.now(timezone.utc) - timedelta(hours=1),
        retry_count=0,
    )

    class DummySettings:
        job_stale_minutes = 10

    async def fake_enqueue(db, job_obj, **_kwargs):
        job_obj.celery_task_id = "new-task"
        job_obj.status = JobStatus.QUEUED.value
        return job_obj

    monkeypatch.setattr(enqueue_module, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(enqueue_module, "enqueue_video_job", fake_enqueue)
    monkeypatch.setattr(enqueue_module, "_collect_active_task_ids", lambda: ({"active-task"}, True))

    db = DummySession(results=[stale_job, active_job])
    recovered = await enqueue_module.recover_stale_processing_jobs(db)

    assert recovered == 1
    assert stale_job.status == JobStatus.QUEUED.value
    assert stale_job.celery_task_id == "new-task"
    assert stale_job.retry_count == 1
    assert stale_job.current_step == "transcription"
    assert stale_job.progress == 18.0
    assert stale_job.step_progress == step_progress
    assert active_job.status == JobStatus.PROCESSING.value


@pytest.mark.asyncio
async def test_recover_stale_processing_jobs_skips_without_workers(monkeypatch):
    job = Job(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        status=JobStatus.PROCESSING.value,
        celery_task_id="old-task",
        started_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )

    monkeypatch.setattr(enqueue_module, "_collect_active_task_ids", lambda: (set(), False))

    db = DummySession(results=[job])
    recovered = await enqueue_module.recover_stale_processing_jobs(db)

    assert recovered == 0
    assert job.status == JobStatus.PROCESSING.value


@pytest.mark.asyncio
async def test_recover_stale_processing_jobs_with_zero_grace_recovers_recent_job(monkeypatch):
    recent_job = Job(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        status=JobStatus.PROCESSING.value,
        celery_task_id="recent-task",
        started_at=datetime.now(timezone.utc),
        current_step="transcription",
        progress=10.0,
        step_progress={"transcription": {"progress": 10.0}},
        retry_count=0,
    )

    async def fake_enqueue(db, job_obj, **_kwargs):
        job_obj.celery_task_id = "new-task"
        job_obj.status = JobStatus.QUEUED.value
        return job_obj

    monkeypatch.setattr(enqueue_module, "enqueue_video_job", fake_enqueue)

    db = DummySession(results=[recent_job])
    recovered = await enqueue_module.recover_stale_processing_jobs_with_grace(
        db,
        stale_minutes=0,
        active_task_ids=set(),
    )

    assert recovered == 1
    assert recent_job.status == JobStatus.QUEUED.value
    assert recent_job.celery_task_id == "new-task"
    assert recent_job.current_step == "transcription"
    assert recent_job.progress == 10.0
    assert recent_job.step_progress == {"transcription": {"progress": 10.0}}
