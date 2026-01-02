import uuid

import pytest

from app.api.routes.jobs import cancel_orphaned_jobs
from app.models.job import Job, JobStatus


class DummyResult:
    def __init__(self, items=None):
        self._items = items or []

    def scalars(self):
        return self

    def all(self):
        return list(self._items)


class DummySession:
    def __init__(self, results):
        self._results = list(results)
        self.commits = 0

    async def execute(self, _query):
        return self._results.pop(0)

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_cancel_orphaned_jobs_marks_cancelled():
    job_one = Job(video_id=uuid.uuid4(), status=JobStatus.QUEUED.value, progress=0.0)
    job_two = Job(video_id=uuid.uuid4(), status=JobStatus.PROCESSING.value, progress=12.5)

    db = DummySession([DummyResult(items=[job_one, job_two])])

    result = await cancel_orphaned_jobs(db)

    assert result["cancelled"] == 2
    assert job_one.status == JobStatus.CANCELLED.value
    assert job_two.status == JobStatus.CANCELLED.value
    assert db.commits == 1


@pytest.mark.asyncio
async def test_cancel_orphaned_jobs_no_matches():
    db = DummySession([DummyResult(items=[])])

    result = await cancel_orphaned_jobs(db)

    assert result["cancelled"] == 0
    assert db.commits == 0
