import uuid
from datetime import datetime, timezone

import pytest

from app.api.routes.videos import VideoTagsUpdate, update_video_tags
from app.models.video import Video


class DummyResult:
    def __init__(self, item):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


class DummySession:
    def __init__(self, results):
        self._results = list(results)
        self.commits = 0

    async def execute(self, _query):
        return self._results.pop(0)

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_update_video_tags_normalizes(monkeypatch):
    video_id = uuid.uuid4()
    video = Video(
        id=video_id,
        filename=f"{video_id}.mp4",
        original_filename="sample.mp4",
        file_path="/tmp/sample.mp4",
        file_size=123,
        mime_type="video/mp4",
        tags=None,
        created_at=datetime.now(timezone.utc),
    )

    payload = VideoTagsUpdate(tags=["  Alpha ", "alpha", "", "Beta"])

    db = DummySession([
        DummyResult(video),
        DummyResult(None),
    ])

    response = await update_video_tags(str(video_id), payload, db)

    assert response.tags == ["Alpha", "Beta"]
    assert video.tags == ["Alpha", "Beta"]
    assert db.commits == 1
