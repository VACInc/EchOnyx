"""Video upload and management endpoints."""

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import and_, case, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.job import Job, JobStatus
from app.models.video import Video

router = APIRouter()
settings = get_settings()


class VideoResponse(BaseModel):
    """Video response schema."""

    id: str
    filename: str
    original_filename: str
    file_size: int
    duration_seconds: float | None
    duration_formatted: str
    title: str | None
    tags: list[str] | None
    status: str
    created_at: str

    class Config:
        from_attributes = True


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
    allowed_types = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo", "video/x-matroska"}
    content_type = file.content_type or "application/octet-stream"
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Allowed: {', '.join(allowed_types)}",
        )

    # Generate unique filename
    video_id = uuid.uuid4()
    file_ext = Path(file.filename).suffix
    unique_filename = f"{video_id}{file_ext}"
    file_path = settings.upload_dir / unique_filename

    # Save file
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Get file size
    file_size = file_path.stat().st_size

    # Check size limit
    max_size = settings.max_upload_size_gb * 1024 * 1024 * 1024
    if file_size > max_size:
        file_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_upload_size_gb}GB",
        )

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

    return VideoResponse(
        id=str(video.id),
        filename=video.filename,
        original_filename=video.original_filename,
        file_size=video.file_size,
        duration_seconds=video.duration_seconds,
        duration_formatted=video.duration_formatted,
        title=video.title,
        tags=video.tags,
        status=job.status if auto_process else "uploaded",
        created_at=video.created_at.isoformat(),
    )


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
        active_job = await _get_active_job(db, video.id)
        if active_job:
            job = active_job
        else:
            job_query = (
                select(Job)
                .where(Job.video_id == video.id)
                .order_by(Job.created_at.desc())
                .limit(1)
            )
            job_result = await db.execute(job_query)
            job = job_result.scalar_one_or_none()
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

        video_responses.append(
            VideoResponse(
                id=str(video.id),
                filename=video.filename,
                original_filename=video.original_filename,
                file_size=video.file_size,
                duration_seconds=video.duration_seconds,
                duration_formatted=video.duration_formatted,
                title=video.title,
                tags=video.tags,
                status=status,
                created_at=video.created_at.isoformat(),
            )
        )

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
    active_statuses = [
        JobStatus.PENDING.value,
        JobStatus.QUEUED.value,
        JobStatus.PROCESSING.value,
    ]

    latest_job_subquery = (
        select(
            Job.video_id.label("video_id"),
            Job.status.label("status"),
            func.row_number()
            .over(partition_by=Job.video_id, order_by=Job.created_at.desc())
            .label("rn"),
        )
        .subquery()
    )

    has_active_job = exists(
        select(1).where(
            Job.video_id == Video.id,
            Job.status.in_(active_statuses),
        )
    )

    latest_status = latest_job_subquery.c.status

    completed_condition = and_(
        ~has_active_job,
        latest_status == JobStatus.COMPLETED.value,
    )

    workload_condition = or_(
        has_active_job,
        latest_status.is_(None),
        latest_status != JobStatus.COMPLETED.value,
    )

    stats_query = (
        select(
            func.count(Video.id).label("total"),
            func.coalesce(func.sum(case((completed_condition, 1), else_=0)), 0).label("completed"),
            func.coalesce(func.sum(case((workload_condition, 1), else_=0)), 0).label("workload"),
        )
        .select_from(Video)
        .outerjoin(
            latest_job_subquery,
            and_(
                latest_job_subquery.c.video_id == Video.id,
                latest_job_subquery.c.rn == 1,
            ),
        )
    )

    result = await db.execute(stats_query)
    row = result.one()

    return VideoStatsResponse(
        total=int(row.total or 0),
        completed=int(row.completed or 0),
        workload=int(row.workload or 0),
    )


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

    active_job = await _get_active_job(db, video.id)
    if active_job:
        job = active_job
    else:
        job_query = (
            select(Job)
            .where(Job.video_id == video.id)
            .order_by(Job.created_at.desc())
            .limit(1)
        )
        job_result = await db.execute(job_query)
        job = job_result.scalar_one_or_none()

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

    return VideoResponse(
        id=str(video.id),
        filename=video.filename,
        original_filename=video.original_filename,
        file_size=video.file_size,
        duration_seconds=video.duration_seconds,
        duration_formatted=video.duration_formatted,
        title=video.title,
        tags=video.tags,
        status=status,
        created_at=video.created_at.isoformat(),
    )


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

    return VideoResponse(
        id=str(video.id),
        filename=video.filename,
        original_filename=video.original_filename,
        file_size=video.file_size,
        duration_seconds=video.duration_seconds,
        duration_formatted=video.duration_formatted,
        title=video.title,
        tags=video.tags,
        status=status,
        created_at=video.created_at.isoformat(),
    )


@router.post("/{video_id}/reprocess", response_model=VideoResponse)
async def reprocess_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
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
        return VideoResponse(
            id=str(video.id),
            filename=video.filename,
            original_filename=video.original_filename,
            file_size=video.file_size,
            duration_seconds=video.duration_seconds,
            duration_formatted=video.duration_formatted,
            title=video.title,
            tags=video.tags,
            status=active_job.status,
            created_at=video.created_at.isoformat(),
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

    return VideoResponse(
        id=str(video.id),
        filename=video.filename,
        original_filename=video.original_filename,
        file_size=video.file_size,
        duration_seconds=video.duration_seconds,
        duration_formatted=video.duration_formatted,
        title=video.title,
        tags=video.tags,
        status=job.status,
        created_at=video.created_at.isoformat(),
    )


@router.post("/{video_id}/reset", response_model=VideoResponse)
async def reset_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
) -> VideoResponse:
    """Reset processing by creating a fresh job for the video."""
    return await reprocess_video(video_id, db)


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
        return VideoResponse(
            id=str(video.id),
            filename=video.filename,
            original_filename=video.original_filename,
            file_size=video.file_size,
            duration_seconds=video.duration_seconds,
            duration_formatted=video.duration_formatted,
            title=video.title,
            tags=video.tags,
            status=active_job.status,
            created_at=video.created_at.isoformat(),
        )

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

    return VideoResponse(
        id=str(video.id),
        filename=video.filename,
        original_filename=video.original_filename,
        file_size=video.file_size,
        duration_seconds=video.duration_seconds,
        duration_formatted=video.duration_formatted,
        title=video.title,
        tags=video.tags,
        status=job.status,
        created_at=video.created_at.isoformat(),
    )


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
