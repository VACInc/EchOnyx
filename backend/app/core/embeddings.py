"""Embeddings and vector search using ChromaDB."""

import asyncio
import json
import logging
import math
import re
import uuid
from collections.abc import Sequence
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.core.model_manager import ModelType, get_model_manager

logger = logging.getLogger(__name__)
CHUNK_TOKEN_REGEX = re.compile(r"[a-z0-9]{2,}")

# Global ChromaDB client
_chroma_client: chromadb.ClientAPI | None = None
_embedding_inference_lock = asyncio.Lock()


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


_LEGACY_COLLECTION_NAME = "video_content"
_LEGACY_OWNER_FILENAME = "legacy_collection_owner.json"


def _collection_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return (slug or "default")[:40].rstrip("-")


def _collection_name_for_model(model_name: str) -> str:
    return f"{_LEGACY_COLLECTION_NAME}--{_collection_slug(model_name)}"


def _legacy_owner_path():
    from pathlib import Path

    return Path(get_settings().chroma_persist_dir) / _LEGACY_OWNER_FILENAME


def _read_legacy_owner_model() -> str | None:
    try:
        payload = json.loads(_legacy_owner_path().read_text())
    except (OSError, ValueError):
        return None
    owner = payload.get("embedding_model")
    return owner if isinstance(owner, str) and owner else None


def _stamp_legacy_owner_model(model_name: str) -> None:
    try:
        path = _legacy_owner_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"embedding_model": model_name}))
    except OSError as exc:  # pragma: no cover - best effort marker
        logger.warning("Could not record legacy collection owner: %s", exc)


def get_collection(name: str | None = None) -> chromadb.Collection:
    """Get or create the ChromaDB collection for the active embedding model.

    Collections are namespaced per embedding model: vectors from different
    models have different dimensions, so a shared collection makes every
    insert and query fail after EMBEDDING_MODEL changes. The pre-namespacing
    legacy collection is adopted by whichever model is active at first use
    after upgrade (recorded in a marker file) and only served to that model;
    any other model gets its own namespaced collection and its content
    returns to semantic results after reprocessing.
    """
    client = get_chroma_client()
    if name is None:
        active_model = get_settings().embedding_model
        namespaced = _collection_name_for_model(active_model)
        existing = {collection.name for collection in client.list_collections()}
        name = namespaced
        if namespaced not in existing and _LEGACY_COLLECTION_NAME in existing:
            owner = _read_legacy_owner_model()
            if owner is None:
                _stamp_legacy_owner_model(active_model)
                name = _LEGACY_COLLECTION_NAME
            elif owner == active_model:
                name = _LEGACY_COLLECTION_NAME
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def delete_video_content(video_id: str) -> None:
    """Remove all indexed content for a video from every content collection."""
    try:
        client = get_chroma_client()
        for collection_info in client.list_collections():
            if not collection_info.name.startswith(_LEGACY_COLLECTION_NAME):
                continue
            client.get_collection(collection_info.name).delete(
                where={"video_id": video_id}
            )
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
    async with _embedding_inference_lock:
        model = await manager.get_model(ModelType.EMBEDDING)
        try:
            loop = asyncio.get_event_loop()

            def do_embed():
                embeddings = model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()

            return await loop.run_in_executor(None, do_embed)

        finally:
            await manager.release_model(ModelType.EMBEDDING)


def _coerce_documents(result: dict) -> list[str]:
    documents = result.get("documents")
    if not documents:
        return []
    if isinstance(documents, Sequence) and documents and isinstance(documents[0], str):
        return [str(doc).strip() for doc in documents if str(doc).strip()]
    return []


def _build_similarity_query_text(documents: list[str], max_documents: int = 4, max_chars: int = 5000) -> str:
    selected: list[str] = []
    total_chars = 0
    for document in documents[:max_documents]:
        normalized = " ".join(document.split())
        if not normalized:
            continue
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        selected.append(normalized[:remaining])
        total_chars += min(len(normalized), remaining)
    return "\n\n".join(selected)


def _normalize_chunk_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _chunk_token_count(text: str) -> int:
    return len(CHUNK_TOKEN_REGEX.findall(text.lower()))


def _is_meaningful_chunk(text: str, content_type: str) -> bool:
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return False
    if not any(char.isalpha() for char in normalized):
        return False
    if normalized.strip(":;-|/ ") == "":
        return False

    token_count = _chunk_token_count(normalized)
    minimum_tokens = 3 if content_type in {"transcript", "summary", "topic"} else 2
    minimum_chars = 18 if content_type in {"transcript", "summary"} else 12
    return token_count >= minimum_tokens or len(normalized) >= minimum_chars


def _append_chunk(
    *,
    chunks: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
    seen_texts: set[str],
    text: str,
    metadata: dict[str, Any],
    chunk_id: str,
) -> None:
    normalized = _normalize_chunk_text(text)
    content_type = str(metadata.get("type") or "")
    if not _is_meaningful_chunk(normalized, content_type):
        return

    dedupe_key = normalized.lower()
    if dedupe_key in seen_texts:
        return

    seen_texts.add(dedupe_key)
    chunks.append(normalized)
    metadatas.append(_sanitize_metadata(metadata))
    ids.append(chunk_id)


def _sanitize_metadata_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, str):
        normalized = " ".join(value.split()).strip()
        return normalized or None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        compact = [item for item in value if item is not None]
        if not compact:
            return None
        return json.dumps(compact, ensure_ascii=True, sort_keys=True, default=str)
    if isinstance(value, dict):
        compact = {str(key): item for key, item in value.items() if item is not None}
        if not compact:
            return None
        return json.dumps(compact, ensure_ascii=True, sort_keys=True, default=str)
    return str(value)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        sanitized_value = _sanitize_metadata_value(value)
        if sanitized_value is None:
            continue
        sanitized[str(key)] = sanitized_value
    return sanitized


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
    seen_texts: set[str] = set()

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
            _append_chunk(
                chunks=chunks,
                metadatas=metadatas,
                ids=ids,
                seen_texts=seen_texts,
                text=chunk_text,
                metadata={
                    "video_id": video_id,
                    "type": "transcript",
                    "timestamp": chunk_start_time,
                    "speaker": segment.get("speaker"),
                },
                chunk_id=chunk_id,
            )

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
        _append_chunk(
            chunks=chunks,
            metadatas=metadatas,
            ids=ids,
            seen_texts=seen_texts,
            text=chunk_text,
            metadata={
                "video_id": video_id,
                "type": "transcript",
                "timestamp": chunk_start_time,
            },
            chunk_id=chunk_id,
        )

    # Add summary content
    if summary:
        # Executive summary
        exec_summary = summary.get("executive_summary", "")
        if exec_summary:
            _append_chunk(
                chunks=chunks,
                metadatas=metadatas,
                ids=ids,
                seen_texts=seen_texts,
                text=exec_summary,
                metadata={
                    "video_id": video_id,
                    "type": "summary",
                    "section": "executive_summary",
                },
                chunk_id=f"{video_id}_summary_exec",
            )

        # Key points
        key_points = summary.get("key_points", [])
        for idx, point in enumerate(key_points):
            point_text = _normalize_chunk_text(str(point or ""))
            if not point_text:
                continue
            _append_chunk(
                chunks=chunks,
                metadatas=metadatas,
                ids=ids,
                seen_texts=seen_texts,
                text=point_text,
                metadata={
                    "video_id": video_id,
                    "type": "summary",
                    "section": "key_points",
                },
                chunk_id=f"{video_id}_summary_keypoint_{idx}",
            )

        # Topics
        for idx, topic in enumerate(summary.get("topics", [])):
            topic_name = _normalize_chunk_text(str(topic.get("topic", "") or ""))
            topic_summary = _normalize_chunk_text(str(topic.get("summary", "") or ""))
            topic_text = ": ".join(part for part in (topic_name, topic_summary) if part)
            _append_chunk(
                chunks=chunks,
                metadatas=metadatas,
                ids=ids,
                seen_texts=seen_texts,
                text=topic_text,
                metadata={
                    "video_id": video_id,
                    "type": "topic",
                    "timestamp": topic.get("timestamp"),
                    "topic_name": topic.get("topic"),
                },
                chunk_id=f"{video_id}_topic_{idx}",
            )

    # Add slide content
    if slides:
        for idx, slide in enumerate(slides):
            slide_title = _normalize_chunk_text(str(slide.get("title", "") or ""))
            slide_content = _normalize_chunk_text(str(slide.get("content", "") or ""))
            slide_text = ": ".join(part for part in (slide_title, slide_content) if part)
            _append_chunk(
                chunks=chunks,
                metadatas=metadatas,
                ids=ids,
                seen_texts=seen_texts,
                text=slide_text,
                metadata={
                    "video_id": video_id,
                    "type": "slide",
                    "timestamp": slide.get("timestamp"),
                    "slide_title": slide.get("title"),
                },
                chunk_id=f"{video_id}_slide_{idx}",
            )

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

    # Use source documents rather than stored embedding rows. Fetching embeddings
    # directly from the persistent Chroma collection has proven brittle on the
    # live Strix Halo instance.
    results = collection.get(
        where={"video_id": video_id},
        include=["documents", "metadatas"],
    )

    source_documents = _select_similarity_source_documents(results)

    query_text = _build_similarity_query_text(source_documents)
    if not query_text:
        return []

    query_embedding = await generate_embeddings([query_text])

    # Search for similar content in other videos
    similar = collection.query(
        query_embeddings=query_embedding,
        n_results=max(n_results * 6, 24),
        include=["documents", "metadatas", "distances"],
    )

    # Aggregate by video_id with per-chunk type weighting and multi-hit support.
    score_buckets: dict[str, list[float]] = {}
    metadatas = similar.get("metadatas", [[]])
    documents = similar.get("documents", [[]])
    distances = similar.get("distances", [[]])
    for idx in range(len(similar["ids"][0])):
        metadata = metadatas[0][idx]
        vid = metadata.get("video_id")
        if not vid or vid == video_id:
            continue
        document = documents[0][idx] if documents and documents[0] else ""
        if not _is_meaningful_chunk(str(document or ""), str(metadata.get("type") or "")):
            continue
        similarity = 1 - distances[0][idx]
        weighted_similarity = similarity * _similarity_match_weight(metadata)
        score_buckets.setdefault(vid, []).append(weighted_similarity)

    video_scores: dict[str, float] = {}
    for vid, scores in score_buckets.items():
        top_scores = sorted(scores, reverse=True)[:2]
        if not top_scores:
            continue
        video_scores[vid] = (top_scores[0] * 0.7) + ((sum(top_scores) / len(top_scores)) * 0.3)

    sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"video_id": vid, "score": score} for vid, score in sorted_videos[:n_results]]


def _select_similarity_source_documents(result: dict) -> list[str]:
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    if not documents:
        return []

    weighted_documents: list[tuple[float, str]] = []
    for index, document in enumerate(documents):
        normalized = _normalize_chunk_text(str(document or ""))
        metadata = metadatas[index] if index < len(metadatas) else {}
        content_type = str(metadata.get("type") or "")
        if not _is_meaningful_chunk(normalized, content_type):
            continue
        weighted_documents.append((_similarity_match_weight(metadata), normalized))

    weighted_documents.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in weighted_documents]


def _similarity_match_weight(metadata: dict[str, Any]) -> float:
    content_type = str(metadata.get("type") or "")
    section = str(metadata.get("section") or "")
    if content_type == "summary" and section == "key_points":
        return 1.0
    if content_type == "transcript":
        return 0.98
    if content_type == "summary":
        return 0.78
    if content_type == "topic":
        return 0.68
    if content_type == "slide":
        return 0.45
    return 0.65
