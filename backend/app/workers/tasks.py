"""Celery tasks for video processing."""

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from celery import shared_task
from sqlalchemy import exists, select

from app.database import get_worker_async_session_maker
from app.models.job import Batch, Job, JobStatus, JobStep, JOB_STEP_ORDER
from app.models.video import Video
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        ).strip()
    except Exception as exc:
        return f"command failed: {exc}"


def log_gpu_memory(tag: str) -> None:
    """Log GPU memory usage (best-effort)."""
    logger.info("GPU memory snapshot (%s)", tag)

    cmds = [
        ["rocm-smi", "--showmemuse", "--showmeminfo", "vram", "--showmeminfo", "gtt", "--showpids"],
        ["/opt/rocm/bin/rocm-smi", "--showmemuse", "--showmeminfo", "vram", "--showmeminfo", "gtt", "--showpids"],
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
    ]

    for cmd in cmds:
        output = _run_cmd(cmd)
        if not output.startswith("command failed:"):
            logger.info("GPU memory command output (%s):\n%s", " ".join(cmd), output)
            break
    else:
        logger.info("GPU memory command output: unavailable")

    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            logger.info("torch.cuda.mem_get_info: free=%s total=%s", free, total)
            logger.info("torch.cuda.memory_allocated: %s", torch.cuda.memory_allocated())
            logger.info("torch.cuda.memory_reserved: %s", torch.cuda.memory_reserved())
    except Exception as exc:
        logger.debug("torch memory query failed: %s", exc)


def run_async(coro):
    """Helper to run async code in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        asyncio.set_event_loop(None)
        loop.close()


async def _mark_job_failed(
    job_id: uuid.UUID,
    error_message: str,
    error_step: str | None = None,
) -> None:
    """Persist a failed job state using a fresh worker-safe DB session."""
    worker_async_session_maker = get_worker_async_session_maker()
    async with worker_async_session_maker() as session:
        job_result = await session.execute(select(Job).where(Job.id == job_id))
        job = job_result.scalar_one_or_none()
        if not job:
            return

        job.status = JobStatus.FAILED.value
        job.error_message = error_message
        job.error_step = error_step or job.current_step
        job.completed_at = datetime.now(timezone.utc)
        await session.commit()
        await _sync_batch_status(session, job.batch_id)


async def _sync_batch_status(session, batch_id: uuid.UUID | None) -> None:
    """Refresh aggregate batch status from the current job states."""
    if batch_id is None:
        return

    batch_result = await session.execute(select(Batch).where(Batch.id == batch_id))
    batch = batch_result.scalar_one_or_none()
    if not batch:
        return

    job_result = await session.execute(select(Job).where(Job.batch_id == batch_id))
    jobs = job_result.scalars().all()
    if not jobs:
        batch.total_videos = 0
        batch.completed_videos = 0
        batch.failed_videos = 0
        batch.status = JobStatus.COMPLETED.value
        await session.commit()
        return

    completed = sum(1 for item in jobs if item.status == JobStatus.COMPLETED.value)
    failed = sum(1 for item in jobs if item.status == JobStatus.FAILED.value)
    cancelled = sum(1 for item in jobs if item.status == JobStatus.CANCELLED.value)
    active = sum(
        1 for item in jobs
        if item.status in {
            JobStatus.PENDING.value,
            JobStatus.QUEUED.value,
            JobStatus.PROCESSING.value,
        }
    )

    batch.total_videos = len(jobs)
    batch.completed_videos = completed
    batch.failed_videos = failed

    if active > 0:
        batch.status = JobStatus.PROCESSING.value
    elif cancelled == len(jobs):
        batch.status = JobStatus.CANCELLED.value
    elif failed > 0 or cancelled > 0:
        batch.status = JobStatus.FAILED.value
    else:
        batch.status = JobStatus.COMPLETED.value

    await session.commit()


@celery_app.task(bind=True, max_retries=3)
def process_video(self, video_id: str, job_id: str):
    """
    Main video processing task.

    Orchestrates the full pipeline:
    1. Audio extraction
    2. Transcription
    3. Diarization
    4. Transcript merging
    5. Frame extraction
    6. Vision analysis
    7. Summarization
    8. Embedding generation
    """
    return run_async(_process_video_async(self, video_id, job_id))


async def _process_video_async(task, video_id: str, job_id: str):
    """Async implementation of video processing."""
    from app.api.websocket import notify_job_error, notify_job_progress, notify_job_step
    from app.core.diarization import diarize_audio, merge_transcript_with_diarization
    from app.core.transcription import transcribe_audio
    from app.utils.ffmpeg import extract_audio

    vid = uuid.UUID(video_id)
    jid = uuid.UUID(job_id)
    job = None
    worker_async_session_maker = get_worker_async_session_maker()

    try:
        async with worker_async_session_maker() as session:
            # Get video and job
            video_result = await session.execute(select(Video).where(Video.id == vid))
            video = video_result.scalar_one_or_none()

            job_result = await session.execute(select(Job).where(Job.id == jid))
            job = job_result.scalar_one_or_none()

            if not video or not job:
                logger.error(f"Video or job not found: video={video_id}, job={job_id}")
                return {"status": "error", "message": "Video or job not found"}

            loop = asyncio.get_running_loop()
            progress_lock = asyncio.Lock()
            step_count = len(JOB_STEP_ORDER)
            step_index_map = {
                step: index + 1
                for index, step in enumerate(JOB_STEP_ORDER)
            }
            step_started_at: dict[JobStep, datetime] = {}

            async def update_step_state(
                step: JobStep,
                progress: float,
                eta_seconds: int | None = None,
                mark_start: bool = False,
                mark_complete: bool = False,
            ):
                now = datetime.now(timezone.utc)
                started_at = None
                completed_at = None
                duration_seconds = None

                if mark_start and step not in step_started_at:
                    step_started_at[step] = now

                if mark_complete:
                    if step not in step_started_at:
                        step_started_at[step] = now
                    started = step_started_at.get(step)
                    started_at = started
                    completed_at = now
                    if started:
                        duration_seconds = (now - started).total_seconds()
                elif mark_start:
                    started_at = step_started_at.get(step)

                async with progress_lock:
                    job.update_step(
                        step,
                        progress,
                        eta_seconds=eta_seconds,
                        step_index=step_index_map.get(step),
                        step_count=step_count,
                        started_at=started_at.isoformat() if started_at else None,
                        completed_at=completed_at.isoformat() if completed_at else None,
                        duration_seconds=duration_seconds,
                    )
                    await session.commit()

            # Progress callbacks may run in executor threads; schedule updates on the task loop.
            def make_progress_callback(
                step: JobStep,
                min_interval: float = 1.0,
                min_delta: float = 1.0,
            ):
                last_emit = 0.0
                last_progress = -1.0

                async def _update(progress: float):
                    await update_step_state(step, progress)
                    await notify_job_step(job_id, step.value, progress)

                def callback(progress: float):
                    nonlocal last_emit, last_progress
                    now = time.monotonic()
                    if (
                        progress < 100
                        and (now - last_emit) < min_interval
                        and (progress - last_progress) < min_delta
                    ):
                        return
                    last_emit = now
                    last_progress = progress
                    try:
                        future = asyncio.run_coroutine_threadsafe(_update(progress), loop)
                    except RuntimeError as exc:
                        logger.warning(
                            "Progress update skipped for %s: %s",
                            step.value,
                            exc,
                        )
                        return

                    def _log_failure(fut):
                        exc = fut.exception()
                        if exc:
                            logger.warning(
                                "Progress update failed for %s: %s",
                                step.value,
                                exc,
                            )

                    future.add_done_callback(_log_failure)

                return callback

            video_path = Path(video.file_path)
            if not video_path.exists():
                async with progress_lock:
                    job.status = JobStatus.FAILED.value
                    job.error_message = "Video file missing from disk"
                    job.error_step = "preflight"
                    job.completed_at = datetime.now(timezone.utc)
                    job.progress = 0.0
                    await session.commit()
                await notify_job_error(job_id, job.error_message or "Video file missing")
                return {"status": "error", "message": "Video file missing"}

            # Update job status
            async with progress_lock:
                job.status = JobStatus.PROCESSING.value
                job.started_at = datetime.now(timezone.utc)
                job.celery_task_id = task.request.id
                await session.commit()

            work_dir = video_path.parent / f"work_{video.id}"
            work_dir.mkdir(exist_ok=True)

            audio_path = work_dir / "audio.wav"
            transcript_path = work_dir / "transcript.json"
            diarization_path = work_dir / "diarization.json"
            merged_path = work_dir / "merged_transcript.json"
            frames_path = work_dir / "frames.json"
            analyzed_frames_path = work_dir / "frames_analyzed.json"
            slides_path = work_dir / "slides.json"
            audio_event_path = work_dir / "audio_event.json"

            def step_completed(step: JobStep) -> bool:
                if not job.step_progress:
                    return False
                step_data = job.step_progress.get(step.value, {})
                return (step_data.get("progress", 0) >= 100) or bool(step_data.get("completed_at"))

            def load_json(path: Path):
                if not path.exists():
                    return None
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("Failed to read %s: %s", path, exc)
                    return None

            def save_json(path: Path, data: object) -> None:
                try:
                    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as exc:
                    logger.warning("Failed to write %s: %s", path, exc)

            log_gpu_memory("start")

            # Step 1: Extract audio
            if step_completed(JobStep.AUDIO_EXTRACTION) and audio_path.exists():
                logger.info("Skipping audio extraction (already completed)")
            else:
                await update_step_state(JobStep.AUDIO_EXTRACTION, 0, mark_start=True)
                await notify_job_step(job_id, JobStep.AUDIO_EXTRACTION.value, 0)
                await extract_audio(video_path, audio_path)
                await update_step_state(JobStep.AUDIO_EXTRACTION, 100, mark_complete=True)
                await notify_job_progress(job_id, job.status, job.progress, job.current_step)
                log_gpu_memory("after audio_extraction")

            # Step 2: Transcription
            transcript = None
            if step_completed(JobStep.TRANSCRIPTION) and transcript_path.exists():
                transcript = load_json(transcript_path)
                logger.info("Skipping transcription (already completed)")
            if transcript is None:
                await update_step_state(JobStep.TRANSCRIPTION, 0, mark_start=True)
                await notify_job_step(job_id, JobStep.TRANSCRIPTION.value, 0)
                transcription_progress = make_progress_callback(JobStep.TRANSCRIPTION)
                transcript = await transcribe_audio(audio_path, progress_callback=transcription_progress)
                save_json(transcript_path, transcript)
                async with progress_lock:
                    video.duration_seconds = transcript.get("duration", 0)
                await update_step_state(JobStep.TRANSCRIPTION, 100, mark_complete=True)
                log_gpu_memory("after transcription")

            # Step 3: Diarization
            diarization = None
            if step_completed(JobStep.DIARIZATION) and diarization_path.exists():
                diarization = load_json(diarization_path)
                logger.info("Skipping diarization (already completed)")
            if diarization is None:
                await update_step_state(JobStep.DIARIZATION, 0, mark_start=True)
                await notify_job_step(job_id, JobStep.DIARIZATION.value, 0)
                diarization_progress = make_progress_callback(JobStep.DIARIZATION)
                diarization = await diarize_audio(audio_path, progress_callback=diarization_progress)
                save_json(diarization_path, diarization)
                await update_step_state(JobStep.DIARIZATION, 100, mark_complete=True)
                log_gpu_memory("after diarization")

            # Step 4: Merge transcript with diarization
            merged_transcript = None
            if step_completed(JobStep.TRANSCRIPT_MERGE):
                merged_transcript = load_json(merged_path) or video.transcript
                if merged_transcript:
                    logger.info("Skipping transcript merge (already completed)")
            if merged_transcript is None:
                await update_step_state(JobStep.TRANSCRIPT_MERGE, 0, mark_start=True)
                merged_transcript = merge_transcript_with_diarization(transcript, diarization)
                save_json(merged_path, merged_transcript)
                async with progress_lock:
                    video.transcript = merged_transcript
                    video.speakers = merged_transcript.get("speakers", [])
                await update_step_state(JobStep.TRANSCRIPT_MERGE, 100, mark_complete=True)
                await notify_job_progress(job_id, job.status, job.progress, job.current_step)
                log_gpu_memory("after transcript_merge")

            # Step 5: Frame extraction
            frames = None
            if step_completed(JobStep.FRAME_EXTRACTION) and frames_path.exists():
                frames = load_json(frames_path)
                logger.info("Skipping frame extraction (already completed)")
            if frames is None:
                await update_step_state(JobStep.FRAME_EXTRACTION, 0, mark_start=True)
                await notify_job_step(job_id, JobStep.FRAME_EXTRACTION.value, 0)
                from app.utils.scene_detect import extract_keyframes

                frames_dir = work_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                frames = await extract_keyframes(video_path, frames_dir)
                if not frames:
                    logger.warning("Frame extraction returned 0 frames for %s", video_id)
                else:
                    logger.info(
                        "Frame extraction complete: %d frames (first=%s last=%s)",
                        len(frames),
                        frames[0].get("path"),
                        frames[-1].get("path"),
                    )
                save_json(frames_path, frames)
                await update_step_state(JobStep.FRAME_EXTRACTION, 100, mark_complete=True)
                log_gpu_memory("after frame_extraction")

            # Step 6: Vision analysis
            analyzed_frames = None
            slides = None
            if step_completed(JobStep.VISION_ANALYSIS) and analyzed_frames_path.exists():
                analyzed_frames = load_json(analyzed_frames_path)
                slides = load_json(slides_path) or video.slides
                if analyzed_frames is not None:
                    logger.info("Skipping vision analysis (already completed)")
            if analyzed_frames is None:
                await update_step_state(JobStep.VISION_ANALYSIS, 0, mark_start=True)
                await notify_job_step(job_id, JobStep.VISION_ANALYSIS.value, 0)

                from app.core.vision import (
                    analyze_frames,
                    annotate_frame_relevance,
                    extract_slide_content,
                )

                vision_progress = make_progress_callback(JobStep.VISION_ANALYSIS)
                analyzed_frames = await analyze_frames(
                    frames,
                    context=video.title,
                    progress_callback=vision_progress,
                )
                logger.info(
                    "Vision analysis complete: %d analyzed frame(s)",
                    len(analyzed_frames),
                )
                annotate_frame_relevance(analyzed_frames, merged_transcript)
                slides = await extract_slide_content(analyzed_frames)
                logger.info("Slides extracted: %d", len(slides))
                save_json(analyzed_frames_path, analyzed_frames)
                save_json(slides_path, slides)
                async with progress_lock:
                    video.slides = slides
                await update_step_state(JobStep.VISION_ANALYSIS, 100, mark_complete=True)
                log_gpu_memory("after vision_analysis")
            else:
                if slides is None:
                    slides = []

            # Step 7: Summarization
            summary = video.summary
            if step_completed(JobStep.SUMMARIZATION) and summary:
                logger.info("Skipping summarization (already completed)")
            else:
                await update_step_state(JobStep.SUMMARIZATION, 0, mark_start=True)
                await notify_job_step(job_id, JobStep.SUMMARIZATION.value, 0)

                from app.core.audio_classification import classify_audio_events
                from app.core.summarizer import generate_summary

                summary_progress = make_progress_callback(JobStep.SUMMARIZATION)
                if not audio_path.exists():
                    await extract_audio(video_path, audio_path)
                audio_event = load_json(audio_event_path)
                if not audio_event:
                    try:
                        audio_event = await classify_audio_events(audio_path)
                    except Exception as exc:
                        logger.warning(
                            "Audio event classification failed for %s; continuing without audio context: %s",
                            video_id,
                            exc,
                            exc_info=True,
                        )
                        audio_event = {
                            "hints": [],
                            "top_labels": [],
                            "tv_score": 0.0,
                            "speech_score": 0.0,
                            "primary_context": None,
                            "supporting_contexts": [],
                            "summary_context": "",
                            "error": str(exc),
                        }
                    save_json(audio_event_path, audio_event)
                audio_hints = audio_event.get("hints", [])
                summary_audio_context = str(audio_event.get("summary_context") or "").strip()
                if summary_audio_context:
                    logger.info("Audio event context: %s", summary_audio_context)
                elif audio_hints:
                    logger.info("Audio event hints: %s", audio_hints)
                summary = await generate_summary(
                    merged_transcript,
                    slides=slides,
                    frames=analyzed_frames,
                    audio_context=audio_event,
                    audio_hints=audio_hints,
                    title=video.title,
                    progress_callback=summary_progress,
                )
                async with progress_lock:
                    video.summary = summary
                await update_step_state(JobStep.SUMMARIZATION, 100, mark_complete=True)
                log_gpu_memory("after summarization")

            from app.config import get_settings
            from app.core.duplicates import best_duplicate_match

            duplicate_match = None
            if merged_transcript or summary:
                candidate_result = await session.execute(
                    select(Video).where(
                        Video.id != video.id,
                        exists(
                            select(1).where(
                                Job.video_id == Video.id,
                                Job.status == JobStatus.COMPLETED.value,
                            )
                        ),
                    )
                )
                duplicate_match = best_duplicate_match(
                    source_video=video,
                    candidate_videos=candidate_result.scalars().all(),
                    settings=get_settings(),
                )
            async with progress_lock:
                video.duplicate_info = duplicate_match
                await session.commit()

            # Step 8: Generate embeddings for search
            if step_completed(JobStep.EMBEDDING):
                logger.info("Skipping embedding (already completed)")
            else:
                await update_step_state(JobStep.EMBEDDING, 0, mark_start=True)
                await notify_job_step(job_id, JobStep.EMBEDDING.value, 0)
                log_gpu_memory("before embedding")

                from app.core.embeddings import delete_video_content, index_video_content

                if duplicate_match and duplicate_match.get("suppressed"):
                    delete_video_content(str(video.id))
                    logger.info(
                        "Skipping embedding for duplicate video %s matched to %s at score %.4f",
                        video.id,
                        duplicate_match.get("representative_video_id"),
                        float(duplicate_match.get("score") or 0.0),
                    )
                else:
                    await index_video_content(
                        video_id=str(video.id),
                        transcript=merged_transcript,
                        summary=summary,
                        slides=slides,
                    )

                await update_step_state(JobStep.EMBEDDING, 100, mark_complete=True)
                log_gpu_memory("after embedding")

            # Mark job as completed
            async with progress_lock:
                job.status = JobStatus.COMPLETED.value
                job.progress = 100.0
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()
                await _sync_batch_status(session, job.batch_id)

            await notify_job_progress(job_id, job.status, 100.0, None)

            logger.info(f"Video processing complete: {video_id}")
            return {"status": "success", "video_id": video_id}

    except Exception as e:
        logger.exception(f"Error processing video {video_id}: {e}")
        log_gpu_memory("error")

        await _mark_job_failed(jid, str(e), job.current_step if job else None)
        await notify_job_error(job_id, str(e), job.current_step if job else None)

        # Retry if applicable
        if task.request.retries < task.max_retries:
            raise task.retry(exc=e, countdown=60 * (task.request.retries + 1))

        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True)
def process_batch(self, batch_id: str):
    """Process all videos in a batch."""
    return run_async(_process_batch_async(self, batch_id))


async def _process_batch_async(task, batch_id: str):
    """Async implementation of batch processing."""
    from app.workers.enqueue import enqueue_video_job

    bid = uuid.UUID(batch_id)
    worker_async_session_maker = get_worker_async_session_maker()

    async with worker_async_session_maker() as session:
        # Get batch
        batch_result = await session.execute(select(Batch).where(Batch.id == bid))
        batch = batch_result.scalar_one_or_none()

        if not batch:
            logger.error(f"Batch not found: {batch_id}")
            return {"status": "error", "message": "Batch not found"}

        # Get all jobs in batch
        jobs_result = await session.execute(
            select(Job)
            .where(Job.batch_id == bid)
            .where(Job.status == JobStatus.QUEUED.value)
            .order_by(Job.created_at)
        )
        jobs = jobs_result.scalars().all()

        batch.status = JobStatus.PROCESSING.value
        await session.commit()

        # Queue each job
        for job in jobs:
            await enqueue_video_job(session, job)

        await _sync_batch_status(session, batch.id)

        logger.info(f"Batch processing started: {batch_id}, {len(jobs)} jobs queued")
        return {"status": "started", "jobs_queued": len(jobs)}
