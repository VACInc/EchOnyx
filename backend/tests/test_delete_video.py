import sys
import tempfile
import types
import uuid
from pathlib import Path

import pytest

from app.api.routes.videos import delete_video
from app.models.video import Video


class DummyResult:
    def __init__(self, scalar=None, items=None):
        self._scalar = scalar
        self._items = items or []

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return list(self._items)


class DummySession:
    def __init__(self, results):
        self._results = list(results)
        self.deleted = []
        self.commits = 0

    async def execute(self, _query):
        return self._results.pop(0)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_delete_video_removes_artifacts(monkeypatch):
    video_id = uuid.uuid4()

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        video_path = base / f"{video_id}.mp4"
        video_path.write_text("data")
        work_dir = base / f"work_{video_id}"
        work_dir.mkdir()
        (work_dir / "audio.wav").write_text("audio")

        video = Video(
            id=video_id,
            filename=video_path.name,
            original_filename="sample.mp4",
            file_path=str(video_path),
            file_size=123,
            mime_type="video/mp4",
        )

        deleted_ids = []
        def fake_delete(video_id_str: str):
            deleted_ids.append(video_id_str)

        dummy_embeddings = types.SimpleNamespace(delete_video_content=fake_delete)
        monkeypatch.setitem(sys.modules, "app.core.embeddings", dummy_embeddings)

        db = DummySession([
            DummyResult(scalar=video),
            DummyResult(items=[]),
        ])

        result = await delete_video(str(video_id), db)

        assert result["message"] == "Video deleted successfully"
        assert not video_path.exists()
        assert not work_dir.exists()
        assert deleted_ids == [str(video_id)]
        assert video in db.deleted
        assert db.commits == 1
