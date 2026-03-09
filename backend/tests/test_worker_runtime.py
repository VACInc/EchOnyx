import sys
import types
import uuid

import pytest
from sqlalchemy.pool import NullPool

from app.database import get_worker_async_session_maker
from app.workers import tasks as tasks_module


def test_get_worker_async_session_maker_uses_null_pool():
    session_maker = get_worker_async_session_maker()
    engine = session_maker.kw["bind"]

    assert isinstance(engine.sync_engine.pool, NullPool)


@pytest.mark.asyncio
async def test_process_video_marks_failed_when_session_bootstrap_fails(monkeypatch):
    job_id = uuid.uuid4()
    captured: dict[str, object] = {}

    class FailingSessionFactory:
        def __call__(self):
            return self

        async def __aenter__(self):
            raise RuntimeError("db bootstrap failed")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_mark_job_failed(job_id_arg, error_message: str, error_step: str | None = None):
        captured["job_id"] = job_id_arg
        captured["error_message"] = error_message
        captured["error_step"] = error_step

    async def fake_notify_job_error(*args, **kwargs):
        captured["notify_job_error"] = {"args": args, "kwargs": kwargs}

    async def _unused_async(*_args, **_kwargs):
        raise AssertionError("pipeline step should not run when session setup fails")

    monkeypatch.setattr(tasks_module, "get_worker_async_session_maker", lambda: FailingSessionFactory())
    monkeypatch.setattr(tasks_module, "_mark_job_failed", fake_mark_job_failed)
    monkeypatch.setitem(
        sys.modules,
        "app.api.websocket",
        types.SimpleNamespace(
            notify_job_error=fake_notify_job_error,
            notify_job_progress=_unused_async,
            notify_job_step=_unused_async,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "app.core.diarization",
        types.SimpleNamespace(
            diarize_audio=_unused_async,
            merge_transcript_with_diarization=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "app.core.transcription",
        types.SimpleNamespace(transcribe_audio=_unused_async),
    )
    monkeypatch.setitem(
        sys.modules,
        "app.utils.ffmpeg",
        types.SimpleNamespace(extract_audio=_unused_async),
    )

    task = types.SimpleNamespace(
        request=types.SimpleNamespace(retries=0, id="celery-task"),
        max_retries=0,
    )

    result = await tasks_module._process_video_async(task, str(uuid.uuid4()), str(job_id))

    assert result["status"] == "error"
    assert captured["job_id"] == job_id
    assert "db bootstrap failed" in str(captured["error_message"])
    assert captured["error_step"] is None
