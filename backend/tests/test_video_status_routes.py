import uuid
from datetime import datetime, timezone

import pytest

from app.api.routes.videos import get_video, list_videos
from app.models.job import Job, JobStatus
from app.models.video import Video
from tests.helpers import SequenceResult


class DummySession:
    def __init__(self, results):
        self._results = list(results)
        self.commits = 0

    async def execute(self, _query):
        return self._results.pop(0)

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_get_video_prefers_active_job_over_older_completed_job(tmp_path):
    video_id = uuid.uuid4()
    file_path = tmp_path / "sample.mp4"
    file_path.write_text("data", encoding="utf-8")
    video = Video(
        id=video_id,
        filename=file_path.name,
        original_filename="sample.mp4",
        file_path=str(file_path),
        file_size=123,
        mime_type="video/mp4",
        created_at=datetime.now(timezone.utc),
    )
    active_job = Job(video_id=video_id, status=JobStatus.PROCESSING.value)

    db = DummySession([
        SequenceResult(scalar=video),
        SequenceResult(scalar=active_job),
    ])

    response = await get_video(str(video_id), db)

    assert response.status == JobStatus.PROCESSING.value


@pytest.mark.asyncio
async def test_list_videos_prefers_active_job_over_completed_job(tmp_path):
    video_id = uuid.uuid4()
    file_path = tmp_path / "sample.mp4"
    file_path.write_text("data", encoding="utf-8")
    video = Video(
        id=video_id,
        filename=file_path.name,
        original_filename="sample.mp4",
        file_path=str(file_path),
        file_size=123,
        mime_type="video/mp4",
        created_at=datetime.now(timezone.utc),
    )
    active_job = Job(video_id=video_id, status=JobStatus.PROCESSING.value)

    db = DummySession([
        SequenceResult(scalar=1),
        SequenceResult(items=[video]),
        SequenceResult(scalar=active_job),
    ])

    response = await list_videos(page=1, page_size=20, db=db)

    assert response.total == 1
    assert response.videos[0].status == JobStatus.PROCESSING.value
