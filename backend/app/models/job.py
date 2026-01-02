"""Job database model for tracking processing tasks."""

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.video import Video


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStep(str, Enum):
    """Processing step enumeration."""

    AUDIO_EXTRACTION = "audio_extraction"
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"
    TRANSCRIPT_MERGE = "transcript_merge"
    FRAME_EXTRACTION = "frame_extraction"
    VISION_ANALYSIS = "vision_analysis"
    SUMMARIZATION = "summarization"
    EMBEDDING = "embedding"


JOB_STEP_ORDER = [
    JobStep.AUDIO_EXTRACTION,
    JobStep.TRANSCRIPTION,
    JobStep.DIARIZATION,
    JobStep.TRANSCRIPT_MERGE,
    JobStep.FRAME_EXTRACTION,
    JobStep.VISION_ANALYSIS,
    JobStep.SUMMARIZATION,
    JobStep.EMBEDDING,
]


class Job(Base, TimestampMixin):
    """Job entity representing a video processing task."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Foreign keys
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
    )
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("batches.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Status tracking
    status: Mapped[str] = mapped_column(
        String(50),
        default=JobStatus.PENDING.value,
        nullable=False,
    )
    current_step: Mapped[str | None] = mapped_column(String(50), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # 0-100

    # Step progress
    step_progress: Mapped[dict | None] = mapped_column(MutableDict.as_mutable(JSON), nullable=True)
    # Example: {"transcription": {"progress": 45, "eta_seconds": 120}}

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Error handling
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_step: Mapped[str | None] = mapped_column(String(50), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Celery task tracking
    celery_task_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Relationships
    video: Mapped["Video"] = relationship("Video", back_populates="jobs")

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, video_id={self.video_id}, status={self.status})>"

    def update_step(
        self,
        step: JobStep,
        progress: float,
        eta_seconds: int | None = None,
        step_index: int | None = None,
        step_count: int | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """Update the current step progress."""
        self.current_step = step.value
        if self.step_progress is None:
            self.step_progress = {}
        step_data = dict(self.step_progress.get(step.value, {}))
        step_data["progress"] = progress

        if step_index is None or step_count is None:
            try:
                step_position = JOB_STEP_ORDER.index(step) + 1
                step_total = len(JOB_STEP_ORDER)
            except ValueError:
                step_position = None
                step_total = None
        else:
            step_position = step_index
            step_total = step_count

        if step_position is not None:
            step_data["step_index"] = step_position
        if step_total is not None:
            step_data["step_count"] = step_total

        if eta_seconds is not None or "eta_seconds" not in step_data:
            step_data["eta_seconds"] = eta_seconds
        if started_at is not None:
            step_data["started_at"] = started_at
        if completed_at is not None:
            step_data["completed_at"] = completed_at
        if duration_seconds is not None:
            step_data["duration_seconds"] = duration_seconds

        self.step_progress[step.value] = step_data
        # Calculate overall progress based on step weights
        self._calculate_overall_progress()

    def _calculate_overall_progress(self) -> None:
        """Calculate overall progress from step progress."""
        step_weights = {
            JobStep.AUDIO_EXTRACTION.value: 5,
            JobStep.TRANSCRIPTION.value: 30,
            JobStep.DIARIZATION.value: 15,
            JobStep.TRANSCRIPT_MERGE.value: 5,
            JobStep.FRAME_EXTRACTION.value: 10,
            JobStep.VISION_ANALYSIS.value: 15,
            JobStep.SUMMARIZATION.value: 15,
            JobStep.EMBEDDING.value: 5,
        }
        total_weight = sum(step_weights.values())

        if not self.step_progress:
            self.progress = 0.0
            return

        weighted_progress = 0.0
        for step, weight in step_weights.items():
            step_data = self.step_progress.get(step, {})
            step_progress = step_data.get("progress", 0)
            weighted_progress += (step_progress / 100) * weight

        self.progress = (weighted_progress / total_weight) * 100


class Batch(Base, TimestampMixin):
    """Batch entity for grouping multiple video processing jobs."""

    __tablename__ = "batches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Batch info
    name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    total_videos: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_videos: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_videos: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Status
    status: Mapped[str] = mapped_column(
        String(50),
        default=JobStatus.PENDING.value,
        nullable=False,
    )

    # Priority (higher = processed first)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    def __repr__(self) -> str:
        return f"<Batch(id={self.id}, total={self.total_videos}, completed={self.completed_videos})>"
