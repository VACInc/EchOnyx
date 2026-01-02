"""Search and RAG endpoints."""

import uuid
from typing import Annotated, Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.video import Video

router = APIRouter()
settings = get_settings()


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
    # TODO: Implement ChromaDB vector search
    # For now, return a basic text search implementation

    results = []

    # Build query
    query = select(Video).where(Video.transcript.isnot(None))

    if video_id:
        try:
            vid = uuid.UUID(video_id)
            query = query.where(Video.id == vid)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid video ID")

    result = await db.execute(query)
    videos = result.scalars().all()

    tag_filter = _normalize_tag_filter(tags)
    search_lower = q.lower()

    for video in videos:
        if not video.transcript:
            continue
        if tag_filter and not _video_has_tags(video.tags, tag_filter):
            continue

        for segment in video.transcript.get("segments", []):
            text = segment.get("text", "")
            if search_lower in text.lower():
                segment_speaker = segment.get("speaker")

                # Skip if speaker filter is set and doesn't match
                if speaker and segment_speaker != speaker:
                    continue

                start_time = segment.get("start", 0)

                results.append(
                    SearchResult(
                        video_id=str(video.id),
                        video_title=video.title or video.original_filename,
                        timestamp=start_time,
                        timestamp_formatted=format_timestamp(start_time),
                        speaker=segment_speaker,
                        text=text,
                        context=get_context(video.transcript, segment),
                        relevance_score=1.0,  # Placeholder
                    )
                )

                if len(results) >= limit:
                    break

        if len(results) >= limit:
            break

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
    # TODO: Implement full RAG pipeline:
    # 1. Embed the question
    # 2. Search ChromaDB for relevant chunks
    # 3. Build context from top results
    # 4. Send to LLM for answer generation

    # For now, return a placeholder
    # Search for relevant content first
    search_results = await search(
        q=question.question,
        video_id=question.video_ids[0] if question.video_ids else None,
        tags=question.tags,
        limit=5,
        db=db,
    )

    if not search_results.results:
        return RAGAnswer(
            question=question.question,
            answer="I couldn't find relevant information to answer this question.",
            sources=[],
            confidence=0.0,
        )

    # TODO: Actually call the LLM here
    # For now, return a placeholder based on search results
    context_text = "\n".join([r.text for r in search_results.results[:3]])

    return RAGAnswer(
        question=question.question,
        answer=f"Based on the video content, I found the following relevant information:\n\n{context_text}",
        sources=search_results.results[:3],
        confidence=0.7,  # Placeholder
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

    # TODO: Implement similarity search using embeddings
    # For now, return empty results

    return SearchResponse(
        query=f"similar to: {video.title or video.original_filename}",
        results=[],
        total=0,
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
