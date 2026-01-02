import pytest

from app.api.routes.videos import get_video_stats


class DummyResult:
    def __init__(self, row):
        self._row = row

    def one(self):
        return self._row


class DummySession:
    def __init__(self, results):
        self._results = list(results)

    async def execute(self, _query):
        return self._results.pop(0)


@pytest.mark.asyncio
async def test_get_video_stats_returns_counts():
    class Row:
        total = 5
        completed = 2
        workload = 3

    db = DummySession([DummyResult(Row())])

    stats = await get_video_stats(db)

    assert stats.total == 5
    assert stats.completed == 2
    assert stats.workload == 3
