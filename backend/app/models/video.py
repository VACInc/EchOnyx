"""Video database model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, BigInteger, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.job import Job


class Video(Base, TimestampMixin):
    """Video entity representing an uploaded video file."""

    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # File info
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)  # bytes
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Video metadata
    duration_seconds: Mapped[float | None] = mapped_column(nullable=True)
    width: Mapped[int | None] = mapped_column(nullable=True)
    height: Mapped[int | None] = mapped_column(nullable=True)
    fps: Mapped[float | None] = mapped_column(nullable=True)

    # User-provided metadata
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # Processing results
    transcript: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    speakers: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    slides: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    duplicate_info: Mapped[dict | None] = mapped_column(MutableDict.as_mutable(JSON), nullable=True)

    # Relationships
    jobs: Mapped[list["Job"]] = relationship("Job", back_populates="video", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Video(id={self.id}, filename={self.original_filename})>"

    @property
    def duration_formatted(self) -> str:
        """Return duration as HH:MM:SS format."""
        if self.duration_seconds is None:
            return "00:00:00"
        hours = int(self.duration_seconds // 3600)
        minutes = int((self.duration_seconds % 3600) // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
