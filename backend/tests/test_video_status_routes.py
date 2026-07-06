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


@pytest.mark.asyncio
async def test_get_video_exposes_safe_duplicate_info(tmp_path):
    video_id = uuid.uuid4()
    representative_id = uuid.uuid4()
    file_path = tmp_path / "duplicate.mp4"
    file_path.write_text("data", encoding="utf-8")
    video = Video(
        id=video_id,
        filename=file_path.name,
        original_filename="duplicate.mp4",
        file_path=str(file_path),
        file_size=123,
        mime_type="video/mp4",
        created_at=datetime.now(timezone.utc),
        duplicate_info={
            "classification": "exact_duplicate",
            "score": 0.9912,
            "suppressed": True,
            "representative_video_id": str(representative_id),
            "representative_title": "Stored title",
            "source_path": "/tmp/private/source.mp4",
        },
    )
    representative = Video(
        id=representative_id,
        filename="kept.mp4",
        original_filename="kept.mp4",
        file_path=str(tmp_path / "kept.mp4"),
        file_size=456,
        mime_type="video/mp4",
        title="Kept copy",
        created_at=datetime.now(timezone.utc),
    )
    active_job = Job(video_id=video_id, status=JobStatus.COMPLETED.value)

    db = DummySession([
        SequenceResult(scalar=video),
        SequenceResult(scalar=active_job),
        SequenceResult(scalar=representative),
    ])

    response = await get_video(str(video_id), db)

    assert response.duplicate_info is not None
    assert response.duplicate_info.classification == "exact_duplicate"
    assert response.duplicate_info.score == 0.9912
    assert response.duplicate_info.suppressed is True
    assert response.duplicate_info.duplicate_of is not None
    assert response.duplicate_info.duplicate_of.id == str(representative_id)
    assert response.duplicate_info.duplicate_of.title == "Kept copy"
    assert "source_path" not in response.duplicate_info.model_dump_json()


@pytest.mark.asyncio
async def test_list_videos_exposes_safe_duplicate_info(tmp_path):
    video_id = uuid.uuid4()
    representative_id = uuid.uuid4()
    file_path = tmp_path / "duplicate.mp4"
    file_path.write_text("data", encoding="utf-8")
    video = Video(
        id=video_id,
        filename=file_path.name,
        original_filename="duplicate.mp4",
        file_path=str(file_path),
        file_size=123,
        mime_type="video/mp4",
        created_at=datetime.now(timezone.utc),
        duplicate_info={
            "classification": "probable_duplicate",
            "score": "0.9123",
            "suppressed": False,
            "representative_video_id": str(representative_id),
            "representative_title": "Stored title",
            "filesystem_path": "/tmp/private/source.mp4",
        },
    )
    active_job = Job(video_id=video_id, status=JobStatus.COMPLETED.value)

    db = DummySession([
        SequenceResult(scalar=1),
        SequenceResult(items=[video]),
        SequenceResult(scalar=active_job),
        SequenceResult(scalar=None),
    ])

    response = await list_videos(page=1, page_size=20, db=db)

    duplicate_info = response.videos[0].duplicate_info
    assert duplicate_info is not None
    assert duplicate_info.classification == "probable_duplicate"
    assert duplicate_info.score == 0.9123
    assert duplicate_info.suppressed is False
    assert duplicate_info.duplicate_of is not None
    assert duplicate_info.duplicate_of.id == str(representative_id)
    assert duplicate_info.duplicate_of.title == "Stored title"
    assert "filesystem_path" not in duplicate_info.model_dump_json()
