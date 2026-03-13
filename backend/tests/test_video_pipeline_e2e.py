import sys
import types
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.api.routes.videos import upload_video
from app.models.job import Job, JobStatus
from app.models.video import Video
from app.workers.tasks import _process_video_async
from tests.helpers import SequenceResult, StreamingUpload, create_sample_video, ensure_timestamp_defaults


class UploadSession:
    def __init__(self):
        self.added = []
        self.flushes = 0
        self.commits = 0

    def add(self, obj):
        ensure_timestamp_defaults(obj)
        self.added.append(obj)

    async def flush(self):
        self.flushes += 1

    async def commit(self):
        self.commits += 1


class ProcessingSession:
    def __init__(self, results):
        self._results = list(results)
        self.commits = 0

    async def execute(self, _query):
        return self._results.pop(0)

    async def commit(self):
        self.commits += 1


class SessionFactory:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def _noop_async(*_args, **_kwargs):
    return None


@pytest.mark.asyncio
async def test_upload_video_streams_sample_mp4(monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.routes.videos.settings.upload_dir", tmp_path)
    monkeypatch.setattr("app.api.routes.videos.settings.max_upload_size_gb", 1)

    source_video = create_sample_video(tmp_path / "upload-source.mp4", color="black", frequency=440)
    upload = StreamingUpload("upload-source.mp4", source_video.read_bytes())
    session = UploadSession()

    async def fake_enqueue(_db, job, **_kwargs):
        job.celery_task_id = "task-upload"
        return job

    monkeypatch.setattr("app.workers.enqueue.enqueue_video_job", fake_enqueue)

    response = await upload_video(file=upload, title="Fixture", db=session)

    assert response.title == "Fixture"
    assert response.status == JobStatus.QUEUED.value
    assert session.flushes == 2
    assert session.commits == 1
    assert all(size != -1 for size in upload.read_sizes)
    saved_video = next(item for item in session.added if isinstance(item, Video))
    assert Path(saved_video.file_path).exists()
    assert Path(saved_video.file_path).read_bytes() == source_video.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("color", "frequency", "transcript_text"),
    [
        ("black", 440, "Discussed roadmap milestones."),
        ("blue", 660, "Reviewed budget updates."),
    ],
)
async def test_process_video_pipeline_with_real_sample_video(
    monkeypatch,
    tmp_path,
    color: str,
    frequency: int,
    transcript_text: str,
):
    video_path = create_sample_video(tmp_path / f"{color}-{frequency}.mp4", color=color, frequency=frequency)
    video_id = uuid.uuid4()
    job_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)

    video = Video(
        id=video_id,
        filename=video_path.name,
        original_filename=video_path.name,
        file_path=str(video_path),
        file_size=video_path.stat().st_size,
        mime_type="video/mp4",
        title=f"{color} fixture",
        created_at=created_at,
    )
    job = Job(
        id=job_id,
        video_id=video_id,
        status=JobStatus.QUEUED.value,
        created_at=created_at,
    )

    captured_embeddings: list[dict] = []
    captured_summary_calls: list[dict] = []

    async def fake_transcribe_audio(_audio_path, **_kwargs):
        return {
            "text": transcript_text,
            "segments": [
                {"start": 0.0, "end": 1.0, "text": transcript_text},
            ],
            "language": "en",
            "duration": 1.0,
        }

    async def fake_diarize_audio(_audio_path, **_kwargs):
        return {
            "speakers": [{"id": "SPEAKER_00", "name": "Speaker 1", "total_time": 1.0}],
            "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
            "num_speakers": 1,
        }

    async def fake_analyze_frames(frames, **_kwargs):
        return [
            {
                **frame,
                "description": f"{color} slide",
                "ocr_text": transcript_text,
                "is_slide": True,
                "slide_title": "Slide 1",
                "key_points": [transcript_text],
            }
            for frame in frames
        ]

    async def fake_extract_slide_content(frames):
        return [
            {
                "timestamp": frames[0]["timestamp"],
                "title": "Slide 1",
                "content": transcript_text,
                "ocr_text": transcript_text,
                "key_points": [transcript_text],
                "description": f"{color} slide",
                "image_path": frames[0]["path"],
            }
        ]

    async def fake_generate_summary(transcript, **_kwargs):
        captured_summary_calls.append(_kwargs)
        return {
            "executive_summary": transcript["text"],
            "key_points": [transcript["text"]],
            "action_items": [],
            "decisions": [],
            "topics": [
                {
                    "timestamp": "00:00:00",
                    "topic": "Overview",
                    "summary": transcript["text"],
                    "speakers": ["Speaker 1"],
                }
            ],
        }

    async def fake_classify_audio_events(_audio_path):
        return {
            "hints": ["Audio most likely sounds like direct meeting-room or office speech."],
            "primary_context": {
                "key": "meeting_room_speech",
                "label": "direct meeting or office speech",
                "confidence": "high",
                "score": 0.71,
                "hint": "Audio most likely sounds like direct meeting-room or office speech.",
            },
            "supporting_contexts": [],
            "summary_context": "Primary audio context: direct meeting or office speech (high confidence).",
        }

    async def fake_index_video_content(**kwargs):
        captured_embeddings.append(kwargs)
        return 1

    monkeypatch.setattr("app.workers.tasks.get_worker_async_session_maker", lambda: lambda: SessionFactory(
        ProcessingSession([
            SequenceResult(scalar=video),
            SequenceResult(scalar=job),
            SequenceResult(items=[]),
        ])
    ))
    monkeypatch.setattr("app.workers.tasks.log_gpu_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.api.websocket.notify_job_error", _noop_async)
    monkeypatch.setattr("app.api.websocket.notify_job_progress", _noop_async)
    monkeypatch.setattr("app.api.websocket.notify_job_step", _noop_async)
    monkeypatch.setattr("app.core.transcription.transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr("app.core.diarization.diarize_audio", fake_diarize_audio)
    monkeypatch.setattr("app.core.vision.analyze_frames", fake_analyze_frames)
    monkeypatch.setattr("app.core.vision.annotate_frame_relevance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.core.vision.extract_slide_content", fake_extract_slide_content)
    monkeypatch.setattr("app.core.summarizer.generate_summary", fake_generate_summary)
    monkeypatch.setattr("app.core.embeddings.index_video_content", fake_index_video_content)
    monkeypatch.setitem(
        sys.modules,
        "app.core.audio_classification",
        types.SimpleNamespace(classify_audio_events=fake_classify_audio_events),
    )

    task = types.SimpleNamespace(
        request=types.SimpleNamespace(id=f"task-{color}", retries=0),
        max_retries=0,
    )

    result = await _process_video_async(task, str(video_id), str(job_id))

    work_dir = tmp_path / f"work_{video_id}"
    assert result["status"] == "success"
    assert job.status == JobStatus.COMPLETED.value
    assert video.duration_seconds == pytest.approx(1.0)
    assert video.transcript is not None
    assert video.summary is not None
    assert video.slides is not None
    assert (work_dir / "audio.wav").exists()
    assert (work_dir / "transcript.json").exists()
    assert (work_dir / "merged_transcript.json").exists()
    assert (work_dir / "frames.json").exists()
    assert (work_dir / "frames_analyzed.json").exists()
    assert (work_dir / "slides.json").exists()
    assert (work_dir / "audio_event.json").exists()
    audio_event = json.loads((work_dir / "audio_event.json").read_text(encoding="utf-8"))
    assert audio_event["primary_context"]["label"] == "direct meeting or office speech"
    assert captured_summary_calls[0]["audio_context"]["primary_context"]["label"] == "direct meeting or office speech"
    assert captured_embeddings and captured_embeddings[0]["video_id"] == str(video_id)


@pytest.mark.asyncio
async def test_process_video_pipeline_continues_when_diarization_is_skipped(monkeypatch, tmp_path):
    video_path = create_sample_video(tmp_path / "no-diarization.mp4", color="green", frequency=550)
    video_id = uuid.uuid4()
    job_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)

    video = Video(
        id=video_id,
        filename=video_path.name,
        original_filename=video_path.name,
        file_path=str(video_path),
        file_size=video_path.stat().st_size,
        mime_type="video/mp4",
        title="skip diarization fixture",
        created_at=created_at,
    )
    job = Job(
        id=job_id,
        video_id=video_id,
        status=JobStatus.QUEUED.value,
        created_at=created_at,
    )

    async def fake_transcribe_audio(_audio_path, **_kwargs):
        return {
            "text": "Budget review is due Friday.",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Budget review is due Friday."},
            ],
            "language": "en",
            "duration": 1.0,
        }

    async def fake_diarize_audio(_audio_path, **_kwargs):
        return {
            "speakers": [],
            "segments": [],
            "num_speakers": 0,
            "skipped": True,
            "reason": "missing_hf_token",
        }

    async def fake_analyze_frames(frames, **_kwargs):
        return [
            {
                **frame,
                "description": "green slide",
                "ocr_text": "Budget review is due Friday.",
                "is_slide": True,
                "slide_title": "Slide 1",
                "key_points": ["Budget review is due Friday."],
            }
            for frame in frames
        ]

    async def fake_extract_slide_content(frames):
        return [
            {
                "timestamp": frames[0]["timestamp"],
                "title": "Slide 1",
                "content": "Budget review is due Friday.",
                "ocr_text": "Budget review is due Friday.",
                "key_points": ["Budget review is due Friday."],
                "description": "green slide",
                "image_path": frames[0]["path"],
            }
        ]

    async def fake_generate_summary(transcript, **_kwargs):
        return {
            "executive_summary": transcript["text"],
            "key_points": [transcript["text"]],
            "action_items": [],
            "decisions": [],
            "topics": [],
        }

    async def fake_classify_audio_events(_audio_path):
        return {
            "hints": ["Audio most likely sounds like direct meeting-room or office speech."],
            "primary_context": {
                "key": "meeting_room_speech",
                "label": "direct meeting or office speech",
                "confidence": "high",
                "score": 0.71,
                "hint": "Audio most likely sounds like direct meeting-room or office speech.",
            },
            "supporting_contexts": [],
            "summary_context": "Primary audio context: direct meeting or office speech (high confidence).",
        }

    async def fake_index_video_content(**_kwargs):
        return 1

    monkeypatch.setattr("app.workers.tasks.get_worker_async_session_maker", lambda: lambda: SessionFactory(
        ProcessingSession([
            SequenceResult(scalar=video),
            SequenceResult(scalar=job),
            SequenceResult(items=[]),
        ])
    ))
    monkeypatch.setattr("app.workers.tasks.log_gpu_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.api.websocket.notify_job_error", _noop_async)
    monkeypatch.setattr("app.api.websocket.notify_job_progress", _noop_async)
    monkeypatch.setattr("app.api.websocket.notify_job_step", _noop_async)
    monkeypatch.setattr("app.core.transcription.transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr("app.core.diarization.diarize_audio", fake_diarize_audio)
    monkeypatch.setattr("app.core.vision.analyze_frames", fake_analyze_frames)
    monkeypatch.setattr("app.core.vision.annotate_frame_relevance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.core.vision.extract_slide_content", fake_extract_slide_content)
    monkeypatch.setattr("app.core.summarizer.generate_summary", fake_generate_summary)
    monkeypatch.setattr("app.core.embeddings.index_video_content", fake_index_video_content)
    monkeypatch.setitem(
        sys.modules,
        "app.core.audio_classification",
        types.SimpleNamespace(classify_audio_events=fake_classify_audio_events),
    )

    task = types.SimpleNamespace(
        request=types.SimpleNamespace(id="task-skip-diarization", retries=0),
        max_retries=0,
    )

    result = await _process_video_async(task, str(video_id), str(job_id))

    work_dir = tmp_path / f"work_{video_id}"
    merged = json.loads((work_dir / "merged_transcript.json").read_text(encoding="utf-8"))
    diarization = json.loads((work_dir / "diarization.json").read_text(encoding="utf-8"))

    assert result["status"] == "success"
    assert job.status == JobStatus.COMPLETED.value
    assert video.speakers == []
    assert merged["segments"][0]["speaker"] is None
    assert diarization["skipped"] is True
    assert diarization["reason"] == "missing_hf_token"


@pytest.mark.asyncio
async def test_process_video_pipeline_continues_when_audio_event_classification_fails(monkeypatch, tmp_path):
    video_path = create_sample_video(tmp_path / "audio-event-failure.mp4", color="red", frequency=500)
    video_id = uuid.uuid4()
    job_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)

    video = Video(
        id=video_id,
        filename=video_path.name,
        original_filename=video_path.name,
        file_path=str(video_path),
        file_size=video_path.stat().st_size,
        mime_type="video/mp4",
        title="audio event fallback fixture",
        created_at=created_at,
    )
    job = Job(
        id=job_id,
        video_id=video_id,
        status=JobStatus.QUEUED.value,
        created_at=created_at,
    )
    captured_summary_calls: list[dict] = []

    async def fake_transcribe_audio(_audio_path, **_kwargs):
        return {
            "text": "Budget review is due Friday.",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Budget review is due Friday."}],
            "language": "en",
            "duration": 1.0,
        }

    async def fake_diarize_audio(_audio_path, **_kwargs):
        return {
            "speakers": [{"id": "SPEAKER_00", "name": "Speaker 1", "total_time": 1.0}],
            "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
            "num_speakers": 1,
        }

    async def fake_analyze_frames(frames, **_kwargs):
        return [
            {
                **frame,
                "description": "red slide",
                "ocr_text": "Budget review is due Friday.",
                "is_slide": True,
                "slide_title": "Slide 1",
                "key_points": ["Budget review is due Friday."],
            }
            for frame in frames
        ]

    async def fake_extract_slide_content(frames):
        return [
            {
                "timestamp": frames[0]["timestamp"],
                "title": "Slide 1",
                "content": "Budget review is due Friday.",
                "ocr_text": "Budget review is due Friday.",
                "key_points": ["Budget review is due Friday."],
                "description": "red slide",
                "image_path": frames[0]["path"],
            }
        ]

    async def fake_generate_summary(transcript, **kwargs):
        captured_summary_calls.append(kwargs)
        return {
            "executive_summary": transcript["text"],
            "key_points": [transcript["text"]],
            "action_items": [],
            "decisions": [],
            "topics": [],
        }

    async def fake_index_video_content(**_kwargs):
        return 1

    async def failing_classify_audio_events(_audio_path):
        raise RuntimeError("torchcodec blew up")

    monkeypatch.setattr("app.workers.tasks.get_worker_async_session_maker", lambda: lambda: SessionFactory(
        ProcessingSession([
            SequenceResult(scalar=video),
            SequenceResult(scalar=job),
            SequenceResult(items=[]),
        ])
    ))
    monkeypatch.setattr("app.workers.tasks.log_gpu_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.api.websocket.notify_job_error", _noop_async)
    monkeypatch.setattr("app.api.websocket.notify_job_progress", _noop_async)
    monkeypatch.setattr("app.api.websocket.notify_job_step", _noop_async)
    monkeypatch.setattr("app.core.transcription.transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr("app.core.diarization.diarize_audio", fake_diarize_audio)
    monkeypatch.setattr("app.core.vision.analyze_frames", fake_analyze_frames)
    monkeypatch.setattr("app.core.vision.annotate_frame_relevance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.core.vision.extract_slide_content", fake_extract_slide_content)
    monkeypatch.setattr("app.core.summarizer.generate_summary", fake_generate_summary)
    monkeypatch.setattr("app.core.embeddings.index_video_content", fake_index_video_content)
    monkeypatch.setitem(
        sys.modules,
        "app.core.audio_classification",
        types.SimpleNamespace(classify_audio_events=failing_classify_audio_events),
    )

    task = types.SimpleNamespace(
        request=types.SimpleNamespace(id="task-audio-event-failure", retries=0),
        max_retries=0,
    )

    result = await _process_video_async(task, str(video_id), str(job_id))

    work_dir = tmp_path / f"work_{video_id}"
    audio_event = json.loads((work_dir / "audio_event.json").read_text(encoding="utf-8"))

    assert result["status"] == "success"
    assert job.status == JobStatus.COMPLETED.value
    assert audio_event["error"] == "torchcodec blew up"
    assert captured_summary_calls[0]["audio_context"]["hints"] == []
