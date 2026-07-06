"""Video upload and management endpoints."""

import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy import case, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.job import Job, JobStatus
from app.models.video import Video
from app.utils.ffmpeg import get_video_info

router = APIRouter()
settings = get_settings()

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
}


async def _save_upload_file(upload: UploadFile, file_path: Path, max_size_bytes: int) -> int:
    """Persist an upload to disk while enforcing the size limit."""
    file_size = 0
    async with aiofiles.open(file_path, "wb") as output:
        while chunk := await upload.read(1024 * 1024):
            file_size += len(chunk)
            if file_size > max_size_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Max size: {settings.max_upload_size_gb}GB",
                )
            await output.write(chunk)
    return file_size


async def _probe_uploaded_video(file_path: Path) -> dict:
    """Ensure the uploaded file is probeable as real video media."""
    try:
        info = await get_video_info(file_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Uploaded file is not valid video media: {exc}") from exc
    if not info.get("duration") or not info.get("width") or not info.get("height"):
        raise HTTPException(status_code=400, detail="Uploaded file is missing required video streams")
    return info


class DuplicateOfVideoResponse(BaseModel):
    """Safe duplicate target metadata."""

    id: str
    title: str | None = None


class VideoDuplicateInfoResponse(BaseModel):
    """Safe duplicate detection metadata."""

    classification: str | None = None
    score: float | None = None
    suppressed: bool | None = None
    duplicate_of: DuplicateOfVideoResponse | None = None


class VideoResponse(BaseModel):
    """Video response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    original_filename: str
    file_size: int
    duration_seconds: float | None
    duration_formatted: str
    title: str | None
    tags: list[str] | None
    duplicate_info: VideoDuplicateInfoResponse | None = None
    status: str
    created_at: str


class VideoListResponse(BaseModel):
    """Paginated video list response."""

    videos: list[VideoResponse]
    total: int
    page: int
    page_size: int


class VideoStatsResponse(BaseModel):
    """Aggregated video stats for dashboard."""

    total: int
    completed: int
    workload: int


class VideoLabelResponse(BaseModel):
    """One user label suggestion with its video count."""

    name: str
    count: int


class VideoLabelsResponse(BaseModel):
    """Distinct user labels across videos."""

    labels: list[VideoLabelResponse]


class VideoTagsUpdate(BaseModel):
    """Update tags for a video."""

    tags: list[str] | None = None


def _normalize_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for tag in tags:
        clean = tag.strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(clean)
    return normalized


async def _get_active_job(db: AsyncSession, video_id: uuid.UUID) -> Job | None:
    result = await db.execute(
        select(Job)
        .where(
            Job.video_id == video_id,
            Job.status.in_([
                JobStatus.PENDING.value,
                JobStatus.QUEUED.value,
                JobStatus.PROCESSING.value,
            ]),
        )
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _get_latest_completed_job(db: AsyncSession, video_id: uuid.UUID) -> Job | None:
    result = await db.execute(
        select(Job)
        .where(
            Job.video_id == video_id,
            Job.status == JobStatus.COMPLETED.value,
        )
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _get_display_job(db: AsyncSession, video_id: uuid.UUID) -> Job | None:
    active_job = await _get_active_job(db, video_id)
    if active_job:
        return active_job

    completed_job = await _get_latest_completed_job(db, video_id)
    if completed_job:
        return completed_job

    job_result = await db.execute(
        select(Job)
        .where(Job.video_id == video_id)
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    return job_result.scalar_one_or_none()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _safe_duplicate_info(video: Video, db: AsyncSession) -> VideoDuplicateInfoResponse | None:
    duplicate_info = video.duplicate_info or {}
    if not isinstance(duplicate_info, dict) or not duplicate_info:
        return None

    duplicate_of = None
    representative_id = duplicate_info.get("representative_video_id")
    if representative_id:
        representative_id_text = str(representative_id)
        representative_title = None
        try:
            representative_uuid = uuid.UUID(representative_id_text)
        except ValueError:
            representative_uuid = None

        if representative_uuid:
            result = await db.execute(select(Video).where(Video.id == representative_uuid))
            representative = result.scalar_one_or_none()
            if representative:
                representative_title = representative.title or representative.original_filename

        representative_title = representative_title or duplicate_info.get("representative_title")
        duplicate_of = DuplicateOfVideoResponse(
            id=representative_id_text,
            title=str(representative_title) if representative_title else None,
        )

    response = VideoDuplicateInfoResponse(
        classification=(
            str(duplicate_info["classification"])
            if duplicate_info.get("classification") is not None
            else None
        ),
        score=_safe_float(duplicate_info.get("score")),
        suppressed=(
            bool(duplicate_info["suppressed"])
            if duplicate_info.get("suppressed") is not None
            else None
        ),
        duplicate_of=duplicate_of,
    )
    if (
        response.classification is None
        and response.score is None
        and response.suppressed is None
        and response.duplicate_of is None
    ):
        return None
    return response


async def _video_response(video: Video, *, status: str, db: AsyncSession) -> VideoResponse:
    return VideoResponse(
        id=str(video.id),
        filename=video.filename,
        original_filename=video.original_filename,
        file_size=video.file_size,
        duration_seconds=video.duration_seconds,
        duration_formatted=video.duration_formatted,
        title=video.title,
        tags=video.tags,
        duplicate_info=await _safe_duplicate_info(video, db),
        status=status,
        created_at=video.created_at.isoformat(),
    )


@router.post("/upload", response_model=VideoResponse)
async def upload_video(
    file: Annotated[UploadFile, File(description="Video file to upload")],
    title: Annotated[str | None, Form()] = None,
    description: Annotated[str | None, Form()] = None,
    auto_process: Annotated[bool, Form()] = True,
    db: AsyncSession = Depends(get_db),
) -> VideoResponse:
    """Upload a single video file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate file type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}",
        )

    # Generate unique filename
    video_id = uuid.uuid4()
    file_ext = Path(file.filename).suffix
    unique_filename = f"{video_id}{file_ext}"
    file_path = settings.upload_dir / unique_filename

    max_size = settings.max_upload_size_gb * 1024 * 1024 * 1024

    # Save file
    try:
        file_size = await _save_upload_file(file, file_path, max_size)
        video_info = await _probe_uploaded_video(file_path)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Create database record
    video = Video(
        id=video_id,
        filename=unique_filename,
        original_filename=file.filename,
        file_path=str(file_path),
        file_size=file_size,
        mime_type=content_type,
        title=title,
        description=description,
        duration_seconds=video_info.get("duration"),
        width=video_info.get("width"),
        height=video_info.get("height"),
        fps=video_info.get("fps"),
    )
    db.add(video)
    await db.flush()

    # Create processing job if auto_process is enabled
    if auto_process:
        job = Job(
            video_id=video.id,
            status=JobStatus.QUEUED.value,
        )
        db.add(job)
        await db.flush()  # Ensure job ID is generated
        await db.commit()

        # Queue Celery task (with retry + failure handling)
        from app.workers.enqueue import enqueue_video_job
        job = await enqueue_video_job(db, job)
    else:
        await db.commit()

    return await _video_response(video, status=job.status if auto_process else "uploaded", db=db)


@router.get("", response_model=VideoListResponse)
async def list_videos(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = None,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> VideoListResponse:
    """List all videos with pagination and filters."""
    query = select(Video).order_by(Video.created_at.desc())
    filters = []

    if search:
        filters.append(
            Video.original_filename.ilike(f"%{search}%")
            | Video.title.ilike(f"%{search}%")
        )
        query = query.where(filters[-1])

    # Get total count
    count_query = select(func.count()).select_from(Video)
    if filters:
        count_query = count_query.where(*filters)
    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    # Get paginated results
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    videos = result.scalars().all()

    # Get latest job status for each video
    video_responses = []
    dirty_jobs = False
    for video in videos:
        job = await _get_display_job(db, video.id)
        file_missing = not Path(video.file_path).exists()
        status = job.status if job else "uploaded"

        if file_missing:
            if job and job.status in {
                JobStatus.PENDING.value,
                JobStatus.QUEUED.value,
                JobStatus.PROCESSING.value,
            }:
                job.status = JobStatus.FAILED.value
                job.error_message = "Video file missing from disk"
                job.error_step = "preflight"
                job.completed_at = datetime.now(timezone.utc)
                job.progress = 0.0
                dirty_jobs = True
            status = JobStatus.FAILED.value if job else "missing"

        video_responses.append(await _video_response(video, status=status, db=db))

    if dirty_jobs:
        await db.commit()

    return VideoListResponse(
        videos=video_responses,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=VideoStatsResponse)
async def get_video_stats(
    db: AsyncSession = Depends(get_db),
) -> VideoStatsResponse:
    """Get aggregated counts for completed vs workload videos."""
    has_completed_job = exists(
        select(1).where(
            Job.video_id == Video.id,
            Job.status == JobStatus.COMPLETED.value,
        )
    )
    completed_condition = has_completed_job

    workload_condition = ~has_completed_job

    stats_query = (
        select(
            func.count(Video.id).label("total"),
            func.coalesce(func.sum(case((completed_condition, 1), else_=0)), 0).label("completed"),
            func.coalesce(func.sum(case((workload_condition, 1), else_=0)), 0).label("workload"),
        )
        .select_from(Video)
    )

    result = await db.execute(stats_query)
    row = result.one()

    return VideoStatsResponse(
        total=int(row.total or 0),
        completed=int(row.completed or 0),
        workload=int(row.workload or 0),
    )


@router.get("/labels", response_model=VideoLabelsResponse)
async def list_video_labels(
    db: AsyncSession = Depends(get_db),
) -> VideoLabelsResponse:
    """List distinct user labels with per-label video counts."""
    result = await db.execute(select(Video.tags))
    tag_lists = result.scalars().all()

    counts: Counter[str] = Counter()
    display_names: dict[str, str] = {}
    for tags in tag_lists:
        if not tags:
            continue
        video_keys: set[str] = set()
        for tag in tags:
            clean = str(tag).strip()
            if not clean:
                continue
            key = clean.lower()
            if key in video_keys:
                continue
            video_keys.add(key)
            counts[key] += 1
            display_names.setdefault(key, clean)

    labels = [
        VideoLabelResponse(name=display_names[key], count=count)
        for key, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], display_names[item[0]].lower(), display_names[item[0]]),
        )
    ]
    return VideoLabelsResponse(labels=labels)


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
) -> VideoResponse:
    """Get a single video by ID."""
    try:
        vid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID")

    result = await db.execute(select(Video).where(Video.id == vid))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    job = await _get_display_job(db, video.id)

    file_missing = not Path(video.file_path).exists()
    status = job.status if job else "uploaded"
    if file_missing:
        if job and job.status in {
            JobStatus.PENDING.value,
            JobStatus.QUEUED.value,
            JobStatus.PROCESSING.value,
        }:
            job.status = JobStatus.FAILED.value
            job.error_message = "Video file missing from disk"
            job.error_step = "preflight"
            job.completed_at = datetime.now(timezone.utc)
            job.progress = 0.0
            await db.commit()
        status = JobStatus.FAILED.value if job else "missing"

    return await _video_response(video, status=status, db=db)


@router.put("/{video_id}/tags", response_model=VideoResponse)
async def update_video_tags(
    video_id: str,
    payload: VideoTagsUpdate,
    db: AsyncSession = Depends(get_db),
) -> VideoResponse:
    """Update user-defined tags for a video."""
    try:
        vid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID")

    result = await db.execute(select(Video).where(Video.id == vid))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    normalized_tags = _normalize_tags(payload.tags)
    video.tags = normalized_tags or None
    await db.commit()

    job_query = (
        select(Job)
        .where(Job.video_id == video.id)
        .order_by(Job.started_at.desc().nulls_last(), Job.created_at.desc())
        .limit(1)
    )
    job_result = await db.execute(job_query)
    job = job_result.scalar_one_or_none()

    status = job.status if job else "uploaded"

    return await _video_response(video, status=status, db=db)


@router.post("/{video_id}/reprocess", response_model=VideoResponse)
async def reprocess_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
    force: Annotated[bool, Query(description="Force a rerun even if the video already completed")] = False,
) -> VideoResponse:
    """Reprocess an existing video (creates a new job)."""
    try:
        vid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID")

    result = await db.execute(select(Video).where(Video.id == vid))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Check if video file still exists
    file_path = Path(video.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="Video file no longer exists")

    active_job = await _get_active_job(db, video.id)
    if active_job:
        return await _video_response(video, status=active_job.status, db=db)

    completed_job = await _get_latest_completed_job(db, video.id)
    if completed_job and not force:
        raise HTTPException(
            status_code=409,
            detail="Video is already completed. Use force=true to rerun it explicitly.",
        )

    # Create new processing job
    job = Job(
        video_id=video.id,
        status=JobStatus.QUEUED.value,
    )
    db.add(job)
    await db.flush()

    await db.commit()

    # Queue Celery task (with retry + failure handling)
    from app.workers.enqueue import enqueue_video_job
    job = await enqueue_video_job(db, job)

    return await _video_response(video, status=job.status, db=db)


@router.post("/{video_id}/reset", response_model=VideoResponse)
async def reset_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
    force: Annotated[bool, Query(description="Force a reset even if the video already completed")] = False,
) -> VideoResponse:
    """Reset processing by creating a fresh job for the video."""
    return await reprocess_video(video_id=video_id, force=force, db=db)


@router.post("/{video_id}/retry", response_model=VideoResponse)
async def retry_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
) -> VideoResponse:
    """Retry the latest failed job for a video, resuming from completed steps."""
    try:
        vid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID")

    result = await db.execute(select(Video).where(Video.id == vid))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    active_job = await _get_active_job(db, video.id)
    if active_job:
        return await _video_response(video, status=active_job.status, db=db)

    job_query = (
        select(Job)
        .where(Job.video_id == video.id)
        .where(Job.status == JobStatus.FAILED.value)
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    job_result = await db.execute(job_query)
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=400, detail="No failed job to retry")

    job.status = JobStatus.QUEUED.value
    job.current_step = None
    job.error_message = None
    job.error_step = None
    job.started_at = None
    job.completed_at = None
    job.retry_count += 1
    await db.commit()

    from app.workers.enqueue import enqueue_video_job
    job = await enqueue_video_job(db, job)

    return await _video_response(video, status=job.status, db=db)


@router.delete("/{video_id}")
async def delete_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a video and its associated data."""
    try:
        vid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID")

    result = await db.execute(select(Video).where(Video.id == vid))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete file from disk
    file_path = Path(video.file_path)
    if file_path.exists():
        file_path.unlink()

    # Delete associated work directory (frames, audio, artifacts)
    work_dir = file_path.parent / f"work_{video.id}"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    # Revoke any queued/active Celery tasks
    try:
        from app.workers.celery_app import celery_app

        jobs_result = await db.execute(select(Job).where(Job.video_id == video.id))
        jobs = jobs_result.scalars().all()
        for job in jobs:
            if job.celery_task_id:
                celery_app.control.revoke(job.celery_task_id, terminate=True)
    except Exception:
        pass

    # Remove embeddings for this video
    try:
        from app.core.embeddings import delete_video_content

        delete_video_content(str(video.id))
    except Exception:
        pass

    # Delete from database (cascade will delete jobs)
    await db.delete(video)
    await db.commit()

    return {"message": "Video deleted successfully"}
