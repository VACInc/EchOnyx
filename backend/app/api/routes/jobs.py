"""Job status and progress endpoints."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.job import Job, JobStatus

router = APIRouter()


class JobProgress(BaseModel):
    """Progress for a single processing step."""

    progress: float
    eta_seconds: int | None = None
    step_index: int | None = None
    step_count: int | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None


class JobResponse(BaseModel):
    """Job response schema."""

    id: str
    video_id: str
    status: str
    current_step: str | None
    progress: float
    step_progress: dict[str, JobProgress] | None
    error_message: str | None
    error_step: str | None
    started_at: str | None
    completed_at: str | None
    created_at: str

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Paginated job list response."""

    jobs: list[JobResponse]
    total: int
    page: int
    page_size: int


def job_to_response(job: Job) -> JobResponse:
    """Convert Job model to response schema."""
    step_progress = None
    if job.step_progress:
        step_progress = {
            k: JobProgress(**v) for k, v in job.step_progress.items()
        }

    return JobResponse(
        id=str(job.id),
        video_id=str(job.video_id),
        status=job.status,
        current_step=job.current_step,
        progress=job.progress,
        step_progress=step_progress,
        error_message=job.error_message,
        error_step=job.error_step,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        created_at=job.created_at.isoformat(),
    )


@router.get("", response_model=JobListResponse)
async def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = None,
    video_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> JobListResponse:
    """List all jobs with pagination and filters."""
    query = select(Job).order_by(Job.created_at.desc())

    if status:
        query = query.where(Job.status == status)

    if video_id:
        try:
            vid = uuid.UUID(video_id)
            query = query.where(Job.video_id == vid)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid video ID")

    # Get total count
    count_result = await db.execute(select(Job.id).select_from(query.subquery()))
    total = len(count_result.all())

    # Get paginated results
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    jobs = result.scalars().all()

    return JobListResponse(
        jobs=[job_to_response(job) for job in jobs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """Get a single job by ID."""
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID")

    result = await db.execute(select(Job).where(Job.id == jid))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job_to_response(job)


@router.get("/{job_id}/progress", response_model=JobResponse)
async def get_job_progress(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """Get detailed progress for a job (alias for get_job)."""
    return await get_job(job_id, db)


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Cancel a pending or processing job."""
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID")

    result = await db.execute(select(Job).where(Job.id == jid))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in [JobStatus.PENDING.value, JobStatus.QUEUED.value, JobStatus.PROCESSING.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}",
        )

    # Cancel Celery task if running
    if job.celery_task_id:
        # TODO: Import and revoke Celery task
        # from app.workers.celery_app import celery_app
        # celery_app.control.revoke(job.celery_task_id, terminate=True)
        pass

    job.status = JobStatus.CANCELLED.value
    await db.commit()

    return {"message": "Job cancelled successfully"}


@router.post("/{job_id}/retry")
async def retry_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """Retry a failed job."""
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID")

    result = await db.execute(select(Job).where(Job.id == jid))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.FAILED.value:
        raise HTTPException(
            status_code=400,
            detail="Can only retry failed jobs",
        )

    # Reset job state but keep completed step progress for resume
    job.status = JobStatus.QUEUED.value
    job.current_step = None
    job.error_message = None
    job.error_step = None
    job.started_at = None
    job.completed_at = None
    job.retry_count += 1

    await db.flush()

    await db.commit()

    # Queue Celery task (with retry + failure handling)
    from app.workers.enqueue import enqueue_video_job
    job = await enqueue_video_job(db, job)

    return job_to_response(job)
