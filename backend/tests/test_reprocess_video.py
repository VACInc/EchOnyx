import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.api.routes.videos import reprocess_video
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
        self.added = []
        self.flushed = 0
        self.commits = 0

    async def execute(self, _query):
        return self._results.pop(0)

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self.flushed += 1

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_reprocess_video_returns_active_job():
    video_id = uuid.uuid4()
    active_job = Job(video_id=video_id, status=JobStatus.PROCESSING.value)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / f"{video_id}.mp4"
        file_path.write_text("data")

        video = Video(
            id=video_id,
            filename=file_path.name,
            original_filename="sample.mp4",
            file_path=str(file_path),
            file_size=123,
            mime_type="video/mp4",
            created_at=datetime.now(timezone.utc),
        )

        db = DummySession([
            DummyResult([video]),
            DummyResult([active_job]),
        ])

        response = await reprocess_video(str(video_id), db)

        assert response.status == JobStatus.PROCESSING.value
        assert db.added == []
        assert db.flushed == 0
