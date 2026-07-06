from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.routes.summaries import get_slide_image, get_summary
from app.security import (
    DEFAULT_LOCAL_ORIGIN_REGEX,
    cors_configuration,
    is_origin_allowed,
    validate_endpoint_url,
    validate_model_name,
)
from app.models.video import Video
from tests.helpers import SequenceResult


class DummySession:
    def __init__(self, results):
        self._results = list(results)

    async def execute(self, _query):
        return self._results.pop(0)


def test_is_origin_allowed_accepts_private_network_hosts():
    assert is_origin_allowed(
        "http://192.168.1.147:3000",
        allowed_origins=[],
        allow_origin_regex=DEFAULT_LOCAL_ORIGIN_REGEX,
    )
    assert is_origin_allowed(
        "http://localhost:3000",
        allowed_origins=[],
        allow_origin_regex=DEFAULT_LOCAL_ORIGIN_REGEX,
    )


def test_is_origin_allowed_rejects_public_origin():
    assert not is_origin_allowed(
        "https://evil.example.com",
        allowed_origins=[],
        allow_origin_regex=DEFAULT_LOCAL_ORIGIN_REGEX,
    )


def test_cors_configuration_keeps_explicit_origins():
    settings = SimpleNamespace(
        cors_allowed_origins="https://app.example.com, http://192.168.1.20:3000 ",
        cors_allow_origin_regex="",
    )

    origins, regex = cors_configuration(settings)

    assert origins == ["https://app.example.com", "http://192.168.1.20:3000"]
    assert regex == DEFAULT_LOCAL_ORIGIN_REGEX


def test_validate_endpoint_url_rejects_public_http():
    with pytest.raises(ValueError):
        validate_endpoint_url("http://example.com:8000")


def test_validate_endpoint_url_accepts_private_http():
    assert validate_endpoint_url("http://192.168.1.50:8000") == "http://192.168.1.50:8000"


def test_validate_model_name_rejects_path_like_values():
    with pytest.raises(ValueError):
        validate_model_name("/tmp/model.gguf")


def test_validate_model_name_accepts_hf_repo_id():
    assert validate_model_name("Qwen/Qwen3-Embedding-8B") == "Qwen/Qwen3-Embedding-8B"


@pytest.mark.asyncio
async def test_get_summary_strips_absolute_slide_paths():
    video_id = uuid4()
    video = Video(
        id=video_id,
        filename="sample.mp4",
        original_filename="sample.mp4",
        file_path="/tmp/sample.mp4",
        file_size=123,
        mime_type="video/mp4",
        created_at=datetime.now(timezone.utc),
        slides=[
            {
                "timestamp": 1.0,
                "image_path": "/Users/vac/EchOnyx/data/uploads/work_123/frames/frame-001.jpg",
                "ocr_text": "Budget",
                "description": "Budget slide",
            }
        ],
    )

    response = await get_summary(str(video_id), db=DummySession([SequenceResult(scalar=video)]))

    assert response.slides[0].image_path == Path("frame-001.jpg").name


@pytest.mark.asyncio
async def test_get_slide_image_serves_only_matching_stored_slide(tmp_path):
    video_id = uuid4()
    slide_path = tmp_path / "frame-001.jpg"
    slide_path.write_bytes(b"jpeg-data")
    video = Video(
        id=video_id,
        filename="sample.mp4",
        original_filename="sample.mp4",
        file_path=str(tmp_path / "sample.mp4"),
        file_size=123,
        mime_type="video/mp4",
        created_at=datetime.now(timezone.utc),
        slides=[
            {
                "timestamp": 1.0,
                "image_path": str(slide_path),
                "ocr_text": "Budget",
                "description": "Budget slide",
            }
        ],
    )

    response = await get_slide_image(
        str(video_id),
        "frame-001.jpg",
        db=DummySession([SequenceResult(scalar=video)]),
    )

    assert Path(response.path) == slide_path
    assert response.media_type == "image/jpeg"


@pytest.mark.asyncio
async def test_get_slide_image_rejects_path_traversal():
    with pytest.raises(HTTPException) as exc:
        await get_slide_image(
            str(uuid4()),
            "../frame-001.jpg",
            db=DummySession([]),
        )

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_get_slide_image_404s_unknown_slide(tmp_path):
    video_id = uuid4()
    slide_path = tmp_path / "frame-001.jpg"
    slide_path.write_bytes(b"jpeg-data")
    video = Video(
        id=video_id,
        filename="sample.mp4",
        original_filename="sample.mp4",
        file_path=str(tmp_path / "sample.mp4"),
        file_size=123,
        mime_type="video/mp4",
        created_at=datetime.now(timezone.utc),
        slides=[{"timestamp": 1.0, "image_path": str(slide_path)}],
    )

    with pytest.raises(HTTPException) as exc:
        await get_slide_image(
            str(video_id),
            "frame-002.jpg",
            db=DummySession([SequenceResult(scalar=video)]),
        )

    assert exc.value.status_code == 404
