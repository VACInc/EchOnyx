import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.api.routes.batch import create_batch, list_batches, cancel_batch
from app.api.routes.jobs import cancel_job, list_jobs
from app.models.job import Batch, Job, JobStatus
from app.models.video import Video
from app.workers.tasks import _process_batch_async
from tests.helpers import SequenceResult, StreamingUpload, create_sample_video, ensure_timestamp_defaults


class QueueSession:
    def __init__(self, results):
        self._results = list(results)
        self.commits = 0
        self.flushes = 0
        self.added = []

    async def execute(self, _query):
        return self._results.pop(0)

    def add(self, obj):
        ensure_timestamp_defaults(obj)
        self.added.append(obj)

    async def flush(self):
        self.flushes += 1

    async def commit(self):
        self.commits += 1


class CancelSession:
    def __init__(self, results):
        self._results = list(results)
        self.commits = 0

    async def execute(self, _query):
        return self._results.pop(0)

    async def commit(self):
        self.commits += 1


def _install_celery_stub(monkeypatch, revoked_calls: list[tuple[str, bool]]) -> None:
    def revoke(task_id: str, terminate: bool = False):
        revoked_calls.append((task_id, terminate))

    celery_module = types.SimpleNamespace(
        celery_app=types.SimpleNamespace(
            control=types.SimpleNamespace(revoke=revoke),
        )
    )
    monkeypatch.setitem(sys.modules, "app.workers.celery_app", celery_module)


@pytest.mark.asyncio
async def test_list_jobs_uses_count_value():
    job = Job(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        status=JobStatus.QUEUED.value,
        progress=0.0,
        created_at=datetime.now(timezone.utc),
    )

    class CountResult:
        def scalar_one(self):
            return 42

        def all(self):
            return [1, 2, 3]

    session = QueueSession([
        CountResult(),
        SequenceResult(items=[job]),
    ])

    response = await list_jobs(page=1, page_size=20, db=session)

    assert response.total == 42
    assert response.jobs[0].id == str(job.id)


@pytest.mark.asyncio
async def test_list_batches_uses_count_value():
    batch = Batch(
        id=uuid.uuid4(),
        name="import",
        total_videos=2,
        completed_videos=1,
        failed_videos=0,
        status=JobStatus.PROCESSING.value,
        created_at=datetime.now(timezone.utc),
    )

    class CountResult:
        def scalar_one(self):
            return 7

        def all(self):
            return [1, 2]

    session = QueueSession([
        CountResult(),
        SequenceResult(items=[batch]),
    ])

    response = await list_batches(page=1, page_size=20, db=session)

    assert response.total == 7
    assert response.batches[0].id == str(batch.id)


@pytest.mark.asyncio
async def test_cancel_job_revokes_celery_task(monkeypatch):
    job = Job(
        id=uuid.uuid4(),
        video_id=uuid.uuid4(),
        status=JobStatus.PROCESSING.value,
        celery_task_id="task-123",
        created_at=datetime.now(timezone.utc),
    )
    revoked_calls: list[tuple[str, bool]] = []
    _install_celery_stub(monkeypatch, revoked_calls)

    session = CancelSession([SequenceResult(scalar=job)])

    response = await cancel_job(str(job.id), session)

    assert response["message"] == "Job cancelled successfully"
    assert job.status == JobStatus.CANCELLED.value
    assert revoked_calls == [("task-123", True)]


@pytest.mark.asyncio
async def test_cancel_batch_revokes_all_celery_tasks(monkeypatch):
    batch = Batch(
        id=uuid.uuid4(),
        name="batch",
        total_videos=2,
        status=JobStatus.PROCESSING.value,
        created_at=datetime.now(timezone.utc),
    )
    jobs = [
        Job(
            id=uuid.uuid4(),
            video_id=uuid.uuid4(),
            batch_id=batch.id,
            status=JobStatus.QUEUED.value,
            celery_task_id="task-a",
            created_at=datetime.now(timezone.utc),
        ),
        Job(
            id=uuid.uuid4(),
            video_id=uuid.uuid4(),
            batch_id=batch.id,
            status=JobStatus.PROCESSING.value,
            celery_task_id="task-b",
            created_at=datetime.now(timezone.utc),
        ),
    ]
    revoked_calls: list[tuple[str, bool]] = []
    _install_celery_stub(monkeypatch, revoked_calls)

    session = CancelSession([
        SequenceResult(scalar=batch),
        SequenceResult(items=jobs),
    ])

    response = await cancel_batch(str(batch.id), session)

    assert response["cancelled_jobs"] == 2
    assert revoked_calls == [("task-a", True), ("task-b", True)]
    assert all(job.status == JobStatus.CANCELLED.value for job in jobs)
    assert batch.status == JobStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_create_batch_streams_uploads_and_enqueues_processing(monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.routes.batch.settings.upload_dir", tmp_path)
    monkeypatch.setattr("app.api.routes.batch.settings.max_upload_size_gb", 1)

    video_one = create_sample_video(tmp_path / "one.mp4", color="black", frequency=440)
    video_two = create_sample_video(tmp_path / "two.mp4", color="blue", frequency=660)
    upload_one = StreamingUpload("one.mp4", video_one.read_bytes())
    upload_two = StreamingUpload("two.mp4", video_two.read_bytes())

    queued_batch_ids: list[str] = []

    class DummyTask:
        def delay(self, batch_id: str):
            queued_batch_ids.append(batch_id)
            return types.SimpleNamespace(id="batch-task-1")

    monkeypatch.setattr("app.workers.tasks.process_batch", DummyTask())

    session = QueueSession([])

    response = await create_batch(
        files=[upload_one, upload_two],
        name="fixtures",
        priority=5,
        db=session,
    )

    assert response.total_videos == 2
    assert response.status == JobStatus.QUEUED.value
    assert len(queued_batch_ids) == 1
    assert session.flushes == 1
    assert session.commits == 1
    assert all(size != -1 for size in upload_one.read_sizes + upload_two.read_sizes)
    saved_videos = [item for item in session.added if isinstance(item, Video)]
    saved_jobs = [item for item in session.added if isinstance(item, Job)]
    assert len(saved_videos) == 2
    assert len(saved_jobs) == 2
    assert all(Path(video.file_path).exists() for video in saved_videos)


@pytest.mark.asyncio
async def test_create_batch_reports_rejected_corrupt_file(monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.routes.batch.settings.upload_dir", tmp_path)
    monkeypatch.setattr("app.api.routes.batch.settings.max_upload_size_gb", 1)

    good_video = create_sample_video(tmp_path / "good.mp4", color="black", frequency=440)
    good_upload = StreamingUpload("good.mp4", good_video.read_bytes())
    corrupt_upload = StreamingUpload("corrupt.mp4", b"not a video")

    queued_batch_ids: list[str] = []

    class DummyTask:
        def delay(self, batch_id: str):
            queued_batch_ids.append(batch_id)
            return types.SimpleNamespace(id="batch-task-1")

    monkeypatch.setattr("app.workers.tasks.process_batch", DummyTask())

    session = QueueSession([])

    response = await create_batch(
        files=[good_upload, corrupt_upload],
        name="mixed",
        priority=0,
        db=session,
    )

    assert response.total_videos == 1
    assert len(response.rejected) == 1
    assert response.rejected[0].filename == "corrupt.mp4"
    assert "not valid video media" in response.rejected[0].reason
    assert len(queued_batch_ids) == 1


@pytest.mark.asyncio
async def test_process_batch_enqueues_each_job(monkeypatch):
    batch = Batch(
        id=uuid.uuid4(),
        name="fixtures",
        total_videos=2,
        status=JobStatus.QUEUED.value,
        created_at=datetime.now(timezone.utc),
    )
    jobs = [
        Job(
            id=uuid.uuid4(),
            video_id=uuid.uuid4(),
            batch_id=batch.id,
            status=JobStatus.QUEUED.value,
            created_at=datetime.now(timezone.utc),
        ),
        Job(
            id=uuid.uuid4(),
            video_id=uuid.uuid4(),
            batch_id=batch.id,
            status=JobStatus.QUEUED.value,
            created_at=datetime.now(timezone.utc),
        ),
    ]

    queued_job_ids: list[uuid.UUID] = []

    async def fake_enqueue(_db, job_obj, **_kwargs):
        queued_job_ids.append(job_obj.id)
        job_obj.celery_task_id = f"task-{job_obj.id}"
        return job_obj

    monkeypatch.setattr("app.workers.enqueue.enqueue_video_job", fake_enqueue)

    session = QueueSession([
        SequenceResult(scalar=batch),
        SequenceResult(items=jobs),
        SequenceResult(scalar=batch),
        SequenceResult(items=jobs),
    ])

    class SessionFactory:
        def __init__(self, inner):
            self.inner = inner

        async def __aenter__(self):
            return self.inner

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("app.workers.tasks.get_worker_async_session_maker", lambda: lambda: SessionFactory(session))

    result = await _process_batch_async(object(), str(batch.id))

    assert result["jobs_queued"] == 2
    assert queued_job_ids == [jobs[0].id, jobs[1].id]
    assert batch.status == JobStatus.PROCESSING.value
