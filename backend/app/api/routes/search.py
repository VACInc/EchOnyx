"""Search and RAG endpoints."""

import uuid
from typing import Annotated, Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.embeddings import find_similar_content, search_content
from app.core.summarizer import complete_with_summarization_model
from app.database import get_db
from app.models.video import Video

router = APIRouter()


class SearchResult(BaseModel):
    """Single search result."""

    video_id: str
    video_title: str
    timestamp: float | None
    timestamp_formatted: str | None
    speaker: str | None
    text: str
    context: str | None
    relevance_score: float


class SearchResponse(BaseModel):
    """Search response with results."""

    query: str
    results: list[SearchResult]
    total: int


class RAGQuestion(BaseModel):
    """RAG question input."""

    question: str
    video_ids: list[str] | None = None  # Optional: limit to specific videos
    tags: list[str] | None = None  # Optional: limit to videos with tags


class RAGAnswer(BaseModel):
    """RAG answer response."""

    question: str
    answer: str
    sources: list[SearchResult]
    confidence: float


QUESTION_ANSWERING_SYSTEM_PROMPT = """You answer questions about processed video content.
Use only the supplied context.
If the context is incomplete, say what is missing instead of inventing facts.
Keep the answer direct and concise."""


@router.get("", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(min_length=1, description="Search query")],
    video_id: str | None = None,
    tags: Sequence[str] | None = Query(None),
    speaker: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    Semantic search across all video transcripts and summaries.

    Uses vector embeddings for semantic matching.
    """
    videos = await _load_candidate_videos(
        db,
        video_id=video_id,
        tags=tags,
    )
    if not videos:
        return SearchResponse(query=q, results=[], total=0)

    semantic_results = await _semantic_search(
        q=q,
        videos=videos,
        speaker=speaker,
        limit=limit,
    )
    if semantic_results:
        return SearchResponse(query=q, results=semantic_results, total=len(semantic_results))

    # Fall back to transcript substring matching for older videos that were never indexed.
    results = _fallback_text_search(videos=videos, q=q, speaker=speaker, limit=limit)

    return SearchResponse(
        query=q,
        results=results[:limit],
        total=len(results),
    )


@router.post("/ask", response_model=RAGAnswer)
async def ask_question(
    question: RAGQuestion,
    db: AsyncSession = Depends(get_db),
) -> RAGAnswer:
    """
    Ask a question about the video content using RAG.

    Example: "What did Bob say about the budget in the Q3 review?"
    """
    videos = await _load_candidate_videos(
        db,
        video_ids=question.video_ids,
        tags=question.tags,
    )
    if not videos:
        return RAGAnswer(
            question=question.question,
            answer="I couldn't find relevant information to answer this question.",
            sources=[],
            confidence=0.0,
        )

    sources = await _semantic_search(
        q=question.question,
        videos=videos,
        speaker=None,
        limit=5,
    )
    if not sources:
        sources = _fallback_text_search(videos=videos, q=question.question, speaker=None, limit=5)

    if not sources:
        return RAGAnswer(
            question=question.question,
            answer="I couldn't find relevant information to answer this question.",
            sources=[],
            confidence=0.0,
        )

    context_blocks = []
    for idx, source in enumerate(sources, start=1):
        prefix = f"Source {idx}: {source.video_title}"
        if source.timestamp_formatted:
            prefix += f" @ {source.timestamp_formatted}"
        context_blocks.append(f"{prefix}\n{source.text}")
        if source.context:
            context_blocks.append(f"Supporting context: {source.context}")

    answer = await complete_with_summarization_model(
        messages=[
            {"role": "system", "content": QUESTION_ANSWERING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question.question}\n\n"
                    "Context:\n"
                    f"{'\n'.join(context_blocks)}"
                ),
            },
        ],
        max_tokens=768,
        temperature=0.1,
    )
    confidence = min(0.95, max((source.relevance_score for source in sources), default=0.0))

    return RAGAnswer(
        question=question.question,
        answer=answer or "I couldn't generate an answer from the available context.",
        sources=sources,
        confidence=confidence,
    )


@router.get("/similar/{video_id}", response_model=SearchResponse)
async def find_similar(
    video_id: str,
    limit: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """Find videos with similar content."""
    try:
        vid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID")

    result = await db.execute(select(Video).where(Video.id == vid))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    similar_videos = await find_similar_content(str(video.id), n_results=limit)
    if not similar_videos:
        return SearchResponse(
            query=f"similar to: {video.title or video.original_filename}",
            results=[],
            total=0,
        )

    similar_ids = [item["video_id"] for item in similar_videos]
    result = await db.execute(select(Video).where(Video.id.in_([uuid.UUID(item) for item in similar_ids])))
    matched_videos = {
        str(item.id): item
        for item in result.scalars().all()
    }

    results = []
    for item in similar_videos:
        matched = matched_videos.get(item["video_id"])
        if not matched:
            continue
        summary = matched.summary or {}
        text = summary.get("executive_summary") or matched.title or matched.original_filename
        results.append(
            SearchResult(
                video_id=str(matched.id),
                video_title=matched.title or matched.original_filename,
                timestamp=None,
                timestamp_formatted=None,
                speaker=None,
                text=text,
                context="similarity",
                relevance_score=float(item["score"]),
            )
        )

    return SearchResponse(
        query=f"similar to: {video.title or video.original_filename}",
        results=results,
        total=len(results),
    )


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_context(transcript: dict, segment: dict, context_segments: int = 1) -> str:
    """Get surrounding context for a segment."""
    segments = transcript.get("segments", [])
    current_idx = None

    for i, s in enumerate(segments):
        if s.get("start") == segment.get("start") and s.get("text") == segment.get("text"):
            current_idx = i
            break

    if current_idx is None:
        return ""

    # Get surrounding segments
    start_idx = max(0, current_idx - context_segments)
    end_idx = min(len(segments), current_idx + context_segments + 1)

    context_parts = []
    for i in range(start_idx, end_idx):
        if i != current_idx:
            context_parts.append(segments[i].get("text", ""))

    return " [...] ".join(context_parts) if context_parts else ""


async def _load_candidate_videos(
    db: AsyncSession,
    video_id: str | None = None,
    video_ids: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
) -> list[Video]:
    query = select(Video)

    id_filters: list[uuid.UUID] = []
    if video_id:
        try:
            id_filters = [uuid.UUID(video_id)]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid video ID")
    elif video_ids:
        try:
            id_filters = [uuid.UUID(value) for value in video_ids]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid video ID")

    if id_filters:
        query = query.where(Video.id.in_(id_filters))

    result = await db.execute(query)
    videos = result.scalars().all()

    tag_filter = _normalize_tag_filter(tags)
    if tag_filter:
        videos = [video for video in videos if _video_has_tags(video.tags, tag_filter)]

    return [
        video for video in videos
        if video.transcript is not None or video.summary is not None or video.slides is not None
    ]


async def _semantic_search(
    q: str,
    videos: Sequence[Video],
    speaker: str | None,
    limit: int,
) -> list[SearchResult]:
    if not videos:
        return []

    video_map = {str(video.id): video for video in videos}
    try:
        matches = await search_content(
            query=q,
            video_ids=list(video_map.keys()),
            n_results=max(limit * 5, limit),
        )
    except Exception:
        return []

    results: list[SearchResult] = []
    for match in matches:
        metadata = match.get("metadata") or {}
        matched_video_id = str(metadata.get("video_id") or "")
        video = video_map.get(matched_video_id)
        if not video:
            continue

        matched_speaker = metadata.get("speaker")
        if speaker and matched_speaker != speaker:
            continue

        timestamp = _coerce_timestamp(metadata.get("timestamp"))
        context = _build_semantic_context(video=video, text=match.get("text", ""), metadata=metadata)
        results.append(
            SearchResult(
                video_id=str(video.id),
                video_title=video.title or video.original_filename,
                timestamp=timestamp,
                timestamp_formatted=format_timestamp(timestamp) if timestamp is not None else None,
                speaker=matched_speaker,
                text=match.get("text", ""),
                context=context,
                relevance_score=float(match.get("score") or 0.0),
            )
        )
        if len(results) >= limit:
            break

    return results


def _fallback_text_search(
    videos: Sequence[Video],
    q: str,
    speaker: str | None,
    limit: int,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    search_lower = q.lower()

    for video in videos:
        if not video.transcript:
            continue

        for segment in video.transcript.get("segments", []):
            text = segment.get("text", "")
            if search_lower not in text.lower():
                continue

            segment_speaker = segment.get("speaker")
            if speaker and segment_speaker != speaker:
                continue

            start_time = segment.get("start", 0.0)
            results.append(
                SearchResult(
                    video_id=str(video.id),
                    video_title=video.title or video.original_filename,
                    timestamp=start_time,
                    timestamp_formatted=format_timestamp(start_time),
                    speaker=segment_speaker,
                    text=text,
                    context=get_context(video.transcript, segment),
                    relevance_score=1.0,
                )
            )
            if len(results) >= limit:
                return results

    return results


def _build_semantic_context(video: Video, text: str, metadata: dict) -> str:
    content_type = str(metadata.get("type") or "")
    if content_type == "transcript" and video.transcript:
        timestamp = _coerce_timestamp(metadata.get("timestamp"))
        if timestamp is not None:
            segment = _find_segment_by_timestamp(video.transcript, timestamp, text)
            if segment:
                return get_context(video.transcript, segment)

    if content_type == "summary":
        section = metadata.get("section")
        if section:
            return f"summary section: {section}"
    if content_type == "topic":
        topic_name = metadata.get("topic_name")
        if topic_name:
            return f"topic: {topic_name}"
    if content_type == "slide":
        slide_title = metadata.get("slide_title")
        if slide_title:
            return f"slide: {slide_title}"

    return content_type or ""


def _find_segment_by_timestamp(transcript: dict, timestamp: float, text: str) -> dict | None:
    segments = transcript.get("segments", [])
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if start <= timestamp <= end:
            return segment
    for segment in segments:
        if text and text in segment.get("text", ""):
            return segment
    return None


def _coerce_timestamp(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            pass
        parts = stripped.split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            try:
                return (int(minutes) * 60) + float(seconds)
            except ValueError:
                return None
        if len(parts) == 3:
            hours, minutes, seconds = parts
            try:
                return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)
            except ValueError:
                return None
    return None


def _normalize_tag_filter(tags: Sequence[str] | None) -> list[str]:
    if not tags:
        return []
    cleaned: list[str] = []
    for tag in tags:
        parts = tag.split(",")
        for part in parts:
            value = part.strip()
            if value:
                cleaned.append(value)
    seen: set[str] = set()
    normalized: list[str] = []
    for tag in cleaned:
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(tag)
    return normalized


def _video_has_tags(video_tags: list[str] | None, filter_tags: Sequence[str]) -> bool:
    if not filter_tags:
        return True
    if not video_tags:
        return False
    video_tag_set = {tag.lower() for tag in video_tags}
    return all(tag.lower() in video_tag_set for tag in filter_tags)
