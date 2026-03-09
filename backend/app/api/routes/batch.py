"""Batch processing endpoints."""

import uuid
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.job import Batch, Job, JobStatus
from app.models.video import Video

router = APIRouter()
settings = get_settings()


class BatchResponse(BaseModel):
    """Batch response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str | None
    status: str
    total_videos: int
    completed_videos: int
    failed_videos: int
    progress: float
    created_at: str


ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
}


async def _save_upload_file(upload: UploadFile, file_path: Path, max_size_bytes: int) -> int:
    """Persist an upload to disk without reading the entire file into memory."""
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


class BatchListResponse(BaseModel):
    """Paginated batch list response."""

    batches: list[BatchResponse]
    total: int
    page: int
    page_size: int


def batch_to_response(batch: Batch) -> BatchResponse:
    """Convert Batch model to response schema."""
    completed_videos = int(batch.completed_videos or 0)
    failed_videos = int(batch.failed_videos or 0)
    progress = 0.0
    if batch.total_videos > 0:
        progress = ((completed_videos + failed_videos) / batch.total_videos) * 100

    return BatchResponse(
        id=str(batch.id),
        name=batch.name,
        status=batch.status,
        total_videos=batch.total_videos,
        completed_videos=completed_videos,
        failed_videos=failed_videos,
        progress=progress,
        created_at=batch.created_at.isoformat(),
    )


@router.get("", response_model=BatchListResponse)
async def list_batches(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> BatchListResponse:
    """List all batches with pagination."""
    query = select(Batch).order_by(Batch.created_at.desc())

    if status:
        query = query.where(Batch.status == status)

    # Get total count
    count_query = select(func.count()).select_from(query.order_by(None).subquery())
    count_result = await db.execute(count_query)
    total = int(count_result.scalar_one())

    # Get paginated results
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    batches = result.scalars().all()

    return BatchListResponse(
        batches=[batch_to_response(batch) for batch in batches],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=BatchResponse)
async def create_batch(
    files: Annotated[list[UploadFile], File(description="Video files to upload")],
    name: Annotated[str | None, Form()] = None,
    priority: Annotated[int, Form()] = 0,
    db: AsyncSession = Depends(get_db),
) -> BatchResponse:
    """Create a batch job with multiple videos."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    max_size_bytes = settings.max_upload_size_gb * 1024 * 1024 * 1024
    accepted_videos: list[Video] = []
    accepted_jobs: list[Job] = []
    batch_id = uuid.uuid4()

    # Process each file
    for file in files:
        if not file.filename:
            continue

        content_type = file.content_type or "application/octet-stream"
        if content_type not in ALLOWED_VIDEO_TYPES:
            continue

        # Generate unique filename
        video_id = uuid.uuid4()
        file_ext = Path(file.filename).suffix or ".mp4"
        unique_filename = f"{video_id}{file_ext}"
        file_path = settings.upload_dir / unique_filename

        # Save file
        try:
            file_size = await _save_upload_file(file, file_path, max_size_bytes)
        except HTTPException:
            if file_path.exists():
                file_path.unlink(missing_ok=True)
            continue
        except Exception:
            if file_path.exists():
                file_path.unlink(missing_ok=True)
            continue

        # Create video record
        video = Video(
            id=video_id,
            filename=unique_filename,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size=file_size,
            mime_type=content_type,
        )
        accepted_videos.append(video)

        # Create job for video
        job = Job(
            video_id=video.id,
            batch_id=batch_id,
            status=JobStatus.QUEUED.value,
        )
        accepted_jobs.append(job)

    if not accepted_videos:
        raise HTTPException(status_code=400, detail="No valid video files provided")

    batch = Batch(
        id=batch_id,
        name=name,
        total_videos=len(accepted_videos),
        completed_videos=0,
        failed_videos=0,
        priority=priority,
        status=JobStatus.QUEUED.value,
    )
    db.add(batch)
    for video in accepted_videos:
        db.add(video)
    for job in accepted_jobs:
        db.add(job)

    await db.flush()
    await db.commit()

    try:
        from app.workers.tasks import process_batch

        process_batch.delay(str(batch.id))
    except Exception as exc:
        batch.status = JobStatus.FAILED.value
        await db.commit()
        raise HTTPException(status_code=502, detail=f"Failed to enqueue batch processing: {exc}")

    return batch_to_response(batch)


@router.get("/{batch_id}", response_model=BatchResponse)
async def get_batch(
    batch_id: str,
    db: AsyncSession = Depends(get_db),
) -> BatchResponse:
    """Get a single batch by ID."""
    try:
        bid = uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid batch ID")

    result = await db.execute(select(Batch).where(Batch.id == bid))
    batch = result.scalar_one_or_none()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    return batch_to_response(batch)


@router.delete("/{batch_id}")
async def cancel_batch(
    batch_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Cancel a batch and all its pending jobs."""
    try:
        bid = uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid batch ID")

    result = await db.execute(select(Batch).where(Batch.id == bid))
    batch = result.scalar_one_or_none()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Cancel all pending/queued/processing jobs in this batch
    job_result = await db.execute(
        select(Job).where(
            Job.batch_id == bid,
            Job.status.in_([
                JobStatus.PENDING.value,
                JobStatus.QUEUED.value,
                JobStatus.PROCESSING.value,
            ]),
        )
    )
    jobs = job_result.scalars().all()

    for job in jobs:
        if job.celery_task_id:
            try:
                from app.workers.celery_app import celery_app

                celery_app.control.revoke(job.celery_task_id, terminate=True)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Failed to revoke Celery task: {exc}")
        job.status = JobStatus.CANCELLED.value

    batch.status = JobStatus.CANCELLED.value
    await db.commit()

    return {"message": "Batch cancelled successfully", "cancelled_jobs": len(jobs)}
