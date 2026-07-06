"""Summary retrieval and export endpoints."""

import mimetypes
import uuid
from io import BytesIO
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.video import Video

router = APIRouter()


class TopicSummary(BaseModel):
    """Topic-based summary segment."""

    timestamp: str
    topic: str
    summary: str
    speakers: list[str] | None = None


class SummaryContent(BaseModel):
    """Structured summary content."""

    executive_summary: str
    key_points: list[str]
    action_items: list[str]
    decisions: list[str]
    topics: list[TopicSummary]


class TranscriptSegment(BaseModel):
    """Transcript segment with speaker and timing."""

    start: float
    end: float
    speaker: str | None
    text: str


class SlideInfo(BaseModel):
    """Extracted slide information."""

    timestamp: float
    image_path: str
    ocr_text: str | None
    description: str | None


class FullSummaryResponse(BaseModel):
    """Complete summary response including all data."""

    video_id: str
    title: str | None
    duration_formatted: str
    speakers: list[str]
    summary: SummaryContent | None
    transcript: list[TranscriptSegment]
    slides: list[SlideInfo]


async def _get_video_or_404(video_id: str, db: AsyncSession) -> Video:
    try:
        vid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID")

    result = await db.execute(select(Video).where(Video.id == vid))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


def _reject_unsafe_slide_filename(filename: str) -> None:
    if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid slide filename")


def _slide_frames_dir(video: Video) -> Path:
    video_path = Path(video.file_path)
    return video_path.parent / f"work_{video.id}" / "frames"


def _resolve_slide_image_path(video: Video, filename: str) -> Path:
    frames_dir = _slide_frames_dir(video)
    try:
        frames_root = frames_dir.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="Slide image not found")

    for slide in video.slides or []:
        raw_path = str(slide.get("image_path", "")).strip()
        if not raw_path:
            continue
        image_path = Path(raw_path)
        if image_path.name == filename:
            candidate = image_path if image_path.is_absolute() else frames_dir / image_path
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(frames_root)
            except (FileNotFoundError, OSError, RuntimeError, ValueError):
                raise HTTPException(status_code=404, detail="Slide image not found")
            if not resolved.is_file():
                raise HTTPException(status_code=404, detail="Slide image not found")
            return resolved
    raise HTTPException(status_code=404, detail="Slide image not found")


@router.get("/{video_id}", response_model=FullSummaryResponse)
async def get_summary(
    video_id: str,
    db: AsyncSession = Depends(get_db),
) -> FullSummaryResponse:
    """Get the full summary for a video."""
    video = await _get_video_or_404(video_id, db)

    # Extract speaker names from speaker data
    speakers = []
    if video.speakers:
        speakers = [s.get("name", f"Speaker {i+1}") for i, s in enumerate(video.speakers)]

    # Parse transcript
    transcript = []
    if video.transcript:
        for segment in video.transcript.get("segments", []):
            transcript.append(
                TranscriptSegment(
                    start=segment.get("start", 0),
                    end=segment.get("end", 0),
                    speaker=segment.get("speaker"),
                    text=segment.get("text", ""),
                )
            )

    # Parse slides
    slides = []
    if video.slides:
        for slide in video.slides:
            slides.append(
                SlideInfo(
                    timestamp=slide.get("timestamp", 0),
                    image_path=Path(str(slide.get("image_path", ""))).name,
                    ocr_text=slide.get("ocr_text"),
                    description=slide.get("description"),
                )
            )

    # Parse summary
    summary = None
    if video.summary:
        summary = SummaryContent(
            executive_summary=video.summary.get("executive_summary", ""),
            key_points=video.summary.get("key_points", []),
            action_items=video.summary.get("action_items", []),
            decisions=video.summary.get("decisions", []),
            topics=[
                TopicSummary(**t) for t in video.summary.get("topics", [])
            ],
        )

    return FullSummaryResponse(
        video_id=str(video.id),
        title=video.title or video.original_filename,
        duration_formatted=video.duration_formatted,
        speakers=speakers,
        summary=summary,
        transcript=transcript,
        slides=slides,
    )


@router.get("/{video_id}/slides/{filename}")
async def get_slide_image(
    video_id: str,
    filename: str,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """Serve a stored slide image for a video."""
    _reject_unsafe_slide_filename(filename)
    video = await _get_video_or_404(video_id, db)
    image_path = _resolve_slide_image_path(video, filename)

    media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    return FileResponse(image_path, media_type=media_type)


@router.get("/{video_id}/export")
async def export_summary(
    video_id: str,
    format: Literal["md", "pdf", "json"] = Query("md"),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Export summary in various formats."""
    # Get the summary data first
    summary_data = await get_summary(video_id, db)

    if format == "json":
        return Response(
            content=summary_data.model_dump_json(indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{video_id}_summary.json"'
            },
        )

    elif format == "md":
        md_content = generate_markdown(summary_data)
        return Response(
            content=md_content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{video_id}_summary.md"'
            },
        )

    elif format == "pdf":
        pdf_content = generate_pdf(summary_data)
        return StreamingResponse(
            BytesIO(pdf_content),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{video_id}_summary.pdf"'
            },
        )


def generate_markdown(data: FullSummaryResponse) -> str:
    """Generate markdown from summary data."""
    lines = [
        f"# {data.title or 'Video Summary'}",
        "",
        f"**Duration:** {data.duration_formatted}",
        f"**Speakers:** {', '.join(data.speakers) if data.speakers else 'N/A'}",
        "",
    ]

    if data.summary:
        lines.extend([
            "## Executive Summary",
            "",
            data.summary.executive_summary,
            "",
            "## Key Points",
            "",
        ])
        for point in data.summary.key_points:
            lines.append(f"- {point}")

        if data.summary.action_items:
            lines.extend([
                "",
                "## Action Items",
                "",
            ])
            for item in data.summary.action_items:
                lines.append(f"- [ ] {item}")

        if data.summary.decisions:
            lines.extend([
                "",
                "## Decisions",
                "",
            ])
            for decision in data.summary.decisions:
                lines.append(f"- {decision}")

        if data.summary.topics:
            lines.extend([
                "",
                "## Topic Breakdown",
                "",
            ])
            for topic in data.summary.topics:
                lines.extend([
                    f"### {topic.topic} ({topic.timestamp})",
                    "",
                    topic.summary,
                    "",
                ])

    if data.transcript:
        lines.extend([
            "## Full Transcript",
            "",
        ])
        for segment in data.transcript:
            speaker = segment.speaker or "Unknown"
            lines.append(f"**[{format_timestamp(segment.start)}] {speaker}:** {segment.text}")
        lines.append("")

    return "\n".join(lines)


def generate_pdf(data: FullSummaryResponse) -> bytes:
    """Generate PDF from summary data."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5 * inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=8,
    )

    story = []

    # Title
    story.append(Paragraph(data.title or "Video Summary", title_style))
    story.append(Spacer(1, 12))

    # Metadata
    story.append(Paragraph(f"<b>Duration:</b> {data.duration_formatted}", styles["Normal"]))
    speakers_str = ", ".join(data.speakers) if data.speakers else "N/A"
    story.append(Paragraph(f"<b>Speakers:</b> {speakers_str}", styles["Normal"]))
    story.append(Spacer(1, 12))

    if data.summary:
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Paragraph(data.summary.executive_summary, styles["Normal"]))
        story.append(Spacer(1, 12))

        # Key Points
        story.append(Paragraph("Key Points", heading_style))
        for point in data.summary.key_points:
            story.append(Paragraph(f"• {point}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Action Items
        if data.summary.action_items:
            story.append(Paragraph("Action Items", heading_style))
            for item in data.summary.action_items:
                story.append(Paragraph(f"☐ {item}", styles["Normal"]))
            story.append(Spacer(1, 12))

        # Decisions
        if data.summary.decisions:
            story.append(Paragraph("Decisions", heading_style))
            for decision in data.summary.decisions:
                story.append(Paragraph(f"• {decision}", styles["Normal"]))
            story.append(Spacer(1, 12))

    doc.build(story)
    return buffer.getvalue()


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
