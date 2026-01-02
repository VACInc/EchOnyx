import uuid
from datetime import datetime, timezone

import pytest

from app.api.routes.videos import retry_video
from app.models.job import Job, JobStatus
from app.models.video import Video


class DummyResult:
    def __init__(self, items):
        self._items = items

    def scalar_one_or_none(self):
        return self._items[0] if self._items else None


class DummySession:
    def __init__(self, results):
        self._results = list(results)
        self.commits = 0

    async def execute(self, _query):
        return self._results.pop(0)

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_retry_video_preserves_step_progress(monkeypatch):
    video_id = uuid.uuid4()
    job_id = uuid.uuid4()

    video = Video(
        id=video_id,
        filename=f"{video_id}.mp4",
        original_filename="sample.mp4",
        file_path="/tmp/sample.mp4",
        file_size=123,
        mime_type="video/mp4",
        created_at=datetime.now(timezone.utc),
    )
    step_progress = {"transcription": {"progress": 100, "completed_at": "2025-01-01T00:00:00Z"}}
    job = Job(
        id=job_id,
        video_id=video_id,
        status=JobStatus.FAILED.value,
        step_progress=step_progress,
        retry_count=0,
    )

    async def fake_enqueue(db, job_obj, **_kwargs):
        job_obj.celery_task_id = "task-xyz"
        job_obj.status = JobStatus.QUEUED.value
        return job_obj

    monkeypatch.setattr("app.workers.enqueue.enqueue_video_job", fake_enqueue)

    db = DummySession([
        DummyResult([video]),
        DummyResult([job]),
    ])

    response = await retry_video(str(video_id), db)

    assert response.status == JobStatus.QUEUED.value
    assert job.step_progress == step_progress
    assert job.error_message is None
    assert job.error_step is None
    assert job.retry_count == 1
