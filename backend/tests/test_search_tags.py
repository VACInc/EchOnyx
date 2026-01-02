import uuid

import pytest

from app.api.routes.search import search
from app.models.video import Video


class DummyResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return self

    def all(self):
        return list(self._items)


class DummySession:
    def __init__(self, results):
        self._results = list(results)

    async def execute(self, _query):
        return DummyResult(self._results.pop(0))


@pytest.mark.asyncio
async def test_search_filters_by_tags():
    video_ok = Video(
        id=uuid.uuid4(),
        filename="a.mp4",
        original_filename="a.mp4",
        file_path="/tmp/a.mp4",
        file_size=1,
        mime_type="video/mp4",
        tags=["Team", "Review"],
        transcript={"segments": [{"text": "hello world", "start": 0}]},
    )
    video_skip = Video(
        id=uuid.uuid4(),
        filename="b.mp4",
        original_filename="b.mp4",
        file_path="/tmp/b.mp4",
        file_size=1,
        mime_type="video/mp4",
        tags=["Other"],
        transcript={"segments": [{"text": "hello world", "start": 0}]},
    )

    db = DummySession([[video_ok, video_skip]])

    response = await search(
        q="hello",
        tags=["team"],
        limit=10,
        db=db,
    )

    assert response.total == 1
    assert response.results[0].video_id == str(video_ok.id)
