"""Batch processing endpoints."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.job import Batch, Job, JobStatus
from app.models.video import Video

router = APIRouter()
settings = get_settings()


class BatchResponse(BaseModel):
    """Batch response schema."""

    id: str
    name: str | None
    status: str
    total_videos: int
    completed_videos: int
    failed_videos: int
    progress: float
    created_at: str

    class Config:
        from_attributes = True


class BatchListResponse(BaseModel):
    """Paginated batch list response."""

    batches: list[BatchResponse]
    total: int
    page: int
    page_size: int


def batch_to_response(batch: Batch) -> BatchResponse:
    """Convert Batch model to response schema."""
    progress = 0.0
    if batch.total_videos > 0:
        progress = ((batch.completed_videos + batch.failed_videos) / batch.total_videos) * 100

    return BatchResponse(
        id=str(batch.id),
        name=batch.name,
        status=batch.status,
        total_videos=batch.total_videos,
        completed_videos=batch.completed_videos,
        failed_videos=batch.failed_videos,
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
    count_result = await db.execute(select(Batch.id).select_from(query.subquery()))
    total = len(count_result.all())

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

    # Create batch
    batch = Batch(
        name=name,
        total_videos=len(files),
        priority=priority,
        status=JobStatus.QUEUED.value,
    )
    db.add(batch)
    await db.flush()

    # Allowed video types
    allowed_types = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo", "video/x-matroska"}

    # Process each file
    for file in files:
        if not file.filename:
            continue

        content_type = file.content_type or "application/octet-stream"
        if content_type not in allowed_types:
            continue

        # Generate unique filename
        video_id = uuid.uuid4()
        file_ext = file.filename.split(".")[-1] if "." in file.filename else "mp4"
        unique_filename = f"{video_id}.{file_ext}"
        file_path = settings.upload_dir / unique_filename

        # Save file
        try:
            content = await file.read()
            file_path.write_bytes(content)
        except Exception:
            continue

        file_size = file_path.stat().st_size

        # Create video record
        video = Video(
            id=video_id,
            filename=unique_filename,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size=file_size,
            mime_type=content_type,
        )
        db.add(video)
        await db.flush()

        # Create job for video
        job = Job(
            video_id=video.id,
            batch_id=batch.id,
            status=JobStatus.QUEUED.value,
        )
        db.add(job)

    await db.commit()

    # TODO: Queue batch processing
    # from app.workers.tasks import process_batch
    # process_batch.delay(str(batch.id))

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
        job.status = JobStatus.CANCELLED.value
        # TODO: Revoke Celery tasks

    batch.status = JobStatus.CANCELLED.value
    await db.commit()

    return {"message": "Batch cancelled successfully", "cancelled_jobs": len(jobs)}
