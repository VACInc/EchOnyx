"""Embeddings and vector search using ChromaDB."""

import asyncio
import logging
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.core.model_manager import ModelType, get_model_manager

logger = logging.getLogger(__name__)

# Global ChromaDB client
_chroma_client: chromadb.ClientAPI | None = None


def _has_embedding_rows(result: dict) -> bool:
    embeddings = result.get("embeddings")
    return embeddings is not None and len(embeddings) > 0


def get_chroma_client() -> chromadb.ClientAPI:
    """Get or create ChromaDB client."""
    global _chroma_client

    if _chroma_client is None:
        settings = get_settings()
        _chroma_client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

    return _chroma_client


def get_collection(name: str = "video_content") -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def delete_video_content(video_id: str) -> None:
    """Remove all indexed content for a video from ChromaDB."""
    try:
        collection = get_collection()
        collection.delete(where={"video_id": video_id})
    except Exception as exc:
        logger.warning("Failed to delete embeddings for %s: %s", video_id, exc)


async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors
    """
    manager = get_model_manager()
    model = await manager.get_model(ModelType.EMBEDDING)

    try:
        loop = asyncio.get_event_loop()

        def do_embed():
            embeddings = model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

        return await loop.run_in_executor(None, do_embed)

    finally:
        await manager.release_model(ModelType.EMBEDDING)


async def index_video_content(
    video_id: str,
    transcript: dict,
    summary: dict | None = None,
    slides: list[dict] | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> int:
    """
    Index video content for semantic search.

    Args:
        video_id: UUID of the video
        transcript: Merged transcript
        summary: Optional summary dict
        slides: Optional slide content
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Number of chunks indexed
    """
    collection = get_collection()

    # Remove existing entries for this video
    try:
        collection.delete(where={"video_id": video_id})
    except Exception:
        pass

    chunks = []
    metadatas = []
    ids = []

    # Chunk transcript segments
    segments = transcript.get("segments", [])
    current_chunk = []
    current_length = 0
    chunk_start_time = 0

    for segment in segments:
        text = segment.get("text", "").strip()
        speaker = segment.get("speaker")
        start_time = segment.get("start", 0)

        if not text:
            continue

        if current_length == 0:
            chunk_start_time = start_time

        # Add segment to current chunk
        current_chunk.append(f"{speaker}: {text}" if speaker else text)
        current_length += len(text)

        # Check if chunk is full
        if current_length >= chunk_size:
            chunk_text = " ".join(current_chunk)
            chunk_id = f"{video_id}_transcript_{len(chunks)}"

            chunks.append(chunk_text)
            metadatas.append({
                "video_id": video_id,
                "type": "transcript",
                "timestamp": chunk_start_time,
                "speaker": segment.get("speaker"),
            })
            ids.append(chunk_id)

            # Start new chunk with overlap
            overlap_segments = []
            overlap_length = 0
            for seg in reversed(current_chunk):
                if overlap_length + len(seg) <= chunk_overlap:
                    overlap_segments.insert(0, seg)
                    overlap_length += len(seg)
                else:
                    break

            current_chunk = overlap_segments
            current_length = overlap_length

    # Add final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_id = f"{video_id}_transcript_{len(chunks)}"
        chunks.append(chunk_text)
        metadatas.append({
            "video_id": video_id,
            "type": "transcript",
            "timestamp": chunk_start_time,
        })
        ids.append(chunk_id)

    # Add summary content
    if summary:
        # Executive summary
        exec_summary = summary.get("executive_summary", "")
        if exec_summary:
            chunks.append(exec_summary)
            metadatas.append({
                "video_id": video_id,
                "type": "summary",
                "section": "executive_summary",
            })
            ids.append(f"{video_id}_summary_exec")

        # Key points
        key_points = summary.get("key_points", [])
        if key_points:
            chunks.append(" ".join(key_points))
            metadatas.append({
                "video_id": video_id,
                "type": "summary",
                "section": "key_points",
            })
            ids.append(f"{video_id}_summary_keypoints")

        # Topics
        for idx, topic in enumerate(summary.get("topics", [])):
            topic_text = f"{topic.get('topic', '')}: {topic.get('summary', '')}"
            chunks.append(topic_text)
            metadatas.append({
                "video_id": video_id,
                "type": "topic",
                "timestamp": topic.get("timestamp"),
                "topic_name": topic.get("topic"),
            })
            ids.append(f"{video_id}_topic_{idx}")

    # Add slide content
    if slides:
        for idx, slide in enumerate(slides):
            slide_text = f"{slide.get('title', '')}: {slide.get('content', '')}"
            if slide_text.strip():
                chunks.append(slide_text)
                metadatas.append({
                    "video_id": video_id,
                    "type": "slide",
                    "timestamp": slide.get("timestamp"),
                    "slide_title": slide.get("title"),
                })
                ids.append(f"{video_id}_slide_{idx}")

    if not chunks:
        logger.warning(f"No content to index for video {video_id}")
        return 0

    # Generate embeddings
    embeddings = await generate_embeddings(chunks)

    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    logger.info(f"Indexed {len(chunks)} chunks for video {video_id}")
    return len(chunks)


async def search_content(
    query: str,
    video_id: str | None = None,
    video_ids: list[str] | None = None,
    content_type: str | None = None,
    n_results: int = 10,
) -> list[dict]:
    """
    Search indexed content using semantic similarity.

    Args:
        query: Search query
        video_id: Optional filter by video
        content_type: Optional filter by type (transcript, summary, slide, topic)
        n_results: Number of results to return

    Returns:
        List of search results with scores
    """
    collection = get_collection()

    # Generate query embedding
    query_embedding = await generate_embeddings([query])

    # Build where filter
    filter_clauses: list[dict[str, Any]] = []
    if video_id:
        filter_clauses.append({"video_id": video_id})
    elif video_ids:
        unique_video_ids = sorted(set(video_ids))
        if len(unique_video_ids) == 1:
            filter_clauses.append({"video_id": unique_video_ids[0]})
        elif unique_video_ids:
            filter_clauses.append({"video_id": {"$in": unique_video_ids}})
    if content_type:
        filter_clauses.append({"type": content_type})

    where: dict[str, Any] | None = None
    if len(filter_clauses) == 1:
        where = filter_clauses[0]
    elif filter_clauses:
        where = {"$and": filter_clauses}

    # Search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    formatted = []
    for idx in range(len(results["ids"][0])):
        formatted.append({
            "id": results["ids"][0][idx],
            "text": results["documents"][0][idx],
            "metadata": results["metadatas"][0][idx],
            "score": 1 - results["distances"][0][idx],  # Convert distance to similarity
        })

    return formatted


async def find_similar_content(
    video_id: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Find videos with similar content.

    Args:
        video_id: Video to find similar content for
        n_results: Number of similar videos to return

    Returns:
        List of similar video IDs with scores
    """
    collection = get_collection()

    # Get embeddings for this video's summary
    results = collection.get(
        where={"$and": [{"video_id": video_id}, {"type": "summary"}]},
        include=["embeddings"],
    )

    if not _has_embedding_rows(results):
        results = collection.get(
            where={"$and": [{"video_id": video_id}, {"type": "transcript"}]},
            include=["embeddings"],
        )

    if not _has_embedding_rows(results):
        return []

    # Use the first summary embedding as query
    query_embedding = results["embeddings"][0]

    # Search for similar content in other videos
    similar = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 3,  # Get more to filter
        where={"type": "summary"},
        include=["metadatas", "distances"],
    )

    # Aggregate by video_id
    video_scores: dict[str, float] = {}
    for idx in range(len(similar["ids"][0])):
        vid = similar["metadatas"][0][idx].get("video_id")
        if vid and vid != video_id:
            score = 1 - similar["distances"][0][idx]
            if vid not in video_scores or score > video_scores[vid]:
                video_scores[vid] = score

    # Sort and return top results
    sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"video_id": vid, "score": score} for vid, score in sorted_videos[:n_results]]
