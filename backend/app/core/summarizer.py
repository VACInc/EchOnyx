"""Summarization module using local or remote LLMs."""

import asyncio
import json
import logging
import re
import time
from typing import Callable

from app.config import get_settings
from app.core.model_manager import ModelType, get_model_manager

import httpx

logger = logging.getLogger(__name__)


SUMMARY_SYSTEM_PROMPT = """You are an expert at summarizing video presentations and meetings.
Given a transcript and optional slide content and visual context from frames (including non-slide frames), create a comprehensive summary.
If audio-source hints are provided, incorporate them into the executive summary or key points as appropriate.

Your summary should include:
1. Executive Summary: A 2-3 sentence overview of the entire content
2. Key Points: The main takeaways (5-10 bullet points)
3. Action Items: Any tasks, to-dos, or next steps mentioned
4. Decisions: Any decisions that were made
5. Topic Breakdown: Major topics discussed with timestamps and brief summaries

Output your response as valid JSON with this structure:
{
    "executive_summary": "...",
    "key_points": ["point 1", "point 2", ...],
    "action_items": ["action 1", "action 2", ...],
    "decisions": ["decision 1", "decision 2", ...],
    "topics": [
        {"timestamp": "00:05:23", "topic": "Topic Name", "summary": "Brief summary", "speakers": ["Speaker 1"]}
    ]
}

Be thorough but concise. Focus on the most important information and use visual context to enrich the summary when available."""

MERGE_SYSTEM_PROMPT = """You are an expert at consolidating multi-part summaries.
You will receive a list of JSON summaries for sequential chunks of a video.

Your job:
1. Merge overlapping or repeated items.
2. Preserve all important details.
3. Produce a clean, unified timeline of topics.

Output a single JSON summary with the same schema as the chunk summaries."""


def _summarization_endpoint_url(settings) -> str:
    url = settings.summarization_endpoint_url.strip()
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    return f"{url.rstrip('/')}/chat/completions"


def _call_summarization_endpoint(
    settings,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> dict:
    endpoint = _summarization_endpoint_url(settings)
    if not endpoint:
        raise RuntimeError("Summarization endpoint URL not configured.")
    headers = {}
    if settings.summarization_endpoint_api_key:
        headers["Authorization"] = f"Bearer {settings.summarization_endpoint_api_key}"
    payload = {
        "model": settings.summarization_endpoint_model or settings.summarization_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request_timeout = min(settings.summarization_endpoint_timeout_s, 60.0)
    deadline = time.monotonic() + max(settings.summarization_endpoint_timeout_s, 5.0)
    attempt = 0

    while True:
        attempt += 1
        try:
            response = httpx.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=request_timeout,
            )
        except httpx.RequestError:
            if time.monotonic() >= deadline:
                raise
            retry_delay = min(5.0, 0.5 * attempt)
            logger.info(
                "Summarization endpoint request failed during startup; retrying in %.1fs (attempt %d).",
                retry_delay,
                attempt,
            )
            time.sleep(retry_delay)
            continue

        if response.status_code == 503 and "loading model" in response.text.lower():
            if time.monotonic() >= deadline:
                response.raise_for_status()
            retry_delay = min(5.0, 0.5 * attempt)
            logger.info(
                "Summarization endpoint is still loading; retrying in %.1fs (attempt %d).",
                retry_delay,
                attempt,
            )
            time.sleep(retry_delay)
            continue

        response.raise_for_status()
        return response.json()


def _extract_chat_content(response: dict) -> str:
    """Extract the assistant message content from a chat completion response."""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    return str(content).strip()


async def complete_with_summarization_model(
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """Generate a generic chat completion using the configured summarization model."""
    settings = get_settings()
    endpoint_url = _summarization_endpoint_url(settings)
    use_endpoint = bool(endpoint_url)
    manager = None
    model = None

    if not use_endpoint:
        manager = get_model_manager()
        model = await manager.get_model(ModelType.SUMMARIZATION)

    try:
        loop = asyncio.get_event_loop()

        def do_complete() -> str:
            response = (
                _call_summarization_endpoint(
                    settings,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if use_endpoint
                else model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            return _extract_chat_content(response)

        return await loop.run_in_executor(None, do_complete)
    finally:
        if manager:
            await manager.release_model(ModelType.SUMMARIZATION)


def parse_summary_json(content: str) -> dict:
    """Parse a JSON summary from a model response."""
    try:
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        raise json.JSONDecodeError("No JSON found", content, 0)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON summary, using fallback")
        return {
            "executive_summary": content[:500],
            "key_points": [],
            "action_items": [],
            "decisions": [],
            "topics": [],
        }


TV_REGEX = re.compile(
    r"\b("
    r"tv|television|smart tv|roku|fire tv|apple tv|cable tv|"
    r"broadcast|news channel|netflix|hulu|prime video"
    r")\b"
)


def _frame_text_for_audio_hint(frame: dict) -> str:
    parts: list[str] = []
    for key in ("description", "ocr_text", "slide_title", "title"):
        value = frame.get(key)
        if value:
            parts.append(str(value))
    key_points = frame.get("key_points") or []
    if isinstance(key_points, list):
        parts.extend(str(p) for p in key_points if p)
    scene_elements = frame.get("scene_elements") or []
    if isinstance(scene_elements, list):
        parts.extend(str(e) for e in scene_elements if e)
    return " ".join(parts).lower()


def infer_audio_source_hints(
    frames: list[dict] | None,
    min_ratio: float = 0.35,
    min_hits: int = 3,
) -> list[str]:
    """Infer lightweight audio-source hints from visual frames."""
    if not frames:
        return []

    total = 0
    hits = 0
    sample_times: list[str] = []

    for frame in frames:
        if frame.get("is_slide"):
            continue
        text = _frame_text_for_audio_hint(frame)
        if not text:
            continue
        total += 1
        if TV_REGEX.search(text):
            hits += 1
            if len(sample_times) < 3:
                sample_times.append(format_timestamp(frame.get("timestamp", 0.0)))

    if total == 0 or hits < min_hits:
        return []

    ratio = hits / total
    if ratio < min_ratio:
        return []

    hint = (
        f"Visuals show a TV/television playing content in {hits}/{total} sampled "
        f"non-slide frames (~{ratio:.0%}). Audio is likely dominated by the TV during those segments."
    )
    if sample_times:
        hint += f" Example timestamps: {', '.join(sample_times)}."
    return [hint]


def split_transcript_by_time(
    transcript: dict,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[dict]:
    """Split transcript segments into overlapping time windows."""
    segments = transcript.get("segments", [])
    if not segments or chunk_seconds <= 0:
        return []

    last_end = max((seg.get("end", 0.0) for seg in segments), default=0.0)
    chunks = []
    start = 0.0
    step = chunk_seconds - overlap_seconds
    if step <= 0:
        step = chunk_seconds

    while start <= last_end:
        end = start + chunk_seconds
        chunk_segments = [
            seg for seg in segments
            if seg.get("end", 0.0) >= start and seg.get("start", 0.0) <= end
        ]
        if chunk_segments:
            chunks.append({
                "start": start,
                "end": end,
                "segments": chunk_segments,
            })
        start += step

    return chunks


def filter_slides_by_time(
    slides: list[dict] | None,
    start: float,
    end: float,
) -> list[dict]:
    """Filter slides to those within a time window."""
    if not slides:
        return []
    filtered = []
    for slide in slides:
        timestamp = slide.get("timestamp")
        if timestamp is None:
            continue
        if start <= timestamp <= end:
            filtered.append(slide)
    return filtered


def filter_frames_by_time(
    frames: list[dict] | None,
    start: float,
    end: float,
) -> list[dict]:
    """Filter frames to those within a time window."""
    if not frames:
        return []
    filtered = []
    for frame in frames:
        timestamp = frame.get("timestamp")
        if timestamp is None:
            continue
        if start <= timestamp <= end:
            filtered.append(frame)
    return filtered


async def generate_summary(
    transcript: dict,
    slides: list[dict] | None = None,
    frames: list[dict] | None = None,
    audio_hints: list[str] | None = None,
    title: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict:
    """
    Generate a structured summary from transcript and slides.

    Args:
        transcript: Merged transcript with speaker labels
        slides: Optional list of extracted slide content
        frames: Optional list of analyzed frames (includes non-slide frames)
        title: Optional video title for context
        progress_callback: Progress callback (0-100)

    Returns:
        Structured summary:
        {
            "executive_summary": str,
            "key_points": list[str],
            "action_items": list[str],
            "decisions": list[str],
            "topics": list[dict],
        }
    """
    settings = get_settings()
    endpoint_url = _summarization_endpoint_url(settings)
    use_endpoint = bool(endpoint_url)
    manager = None
    model = None

    if not use_endpoint:
        manager = get_model_manager()
        model = await manager.get_model(ModelType.SUMMARIZATION)

    try:
        loop = asyncio.get_event_loop()

        def do_summarize():
            chunk_seconds = settings.summary_chunk_minutes * 60.0
            overlap_seconds = settings.summary_chunk_overlap_minutes * 60.0

            # Build transcript text with speaker labels
            transcript_text = format_transcript_for_summary(transcript)

            # Build slides text
            slides_text = ""
            if slides:
                slides_text = format_slides_for_summary(slides)

            # Build visual context from frames (includes non-slide frames)
            visual_text = ""
            if frames:
                visual_text = format_visual_context(
                    frames,
                    max_chars=settings.visual_context_max_chars,
                    max_frames=settings.visual_context_max_frames,
                )
            derived_audio_hints = infer_audio_source_hints(frames)
            merged_audio_hints = []
            if audio_hints:
                merged_audio_hints.extend(audio_hints)
            if derived_audio_hints:
                merged_audio_hints.extend(
                    hint for hint in derived_audio_hints if hint not in merged_audio_hints
                )

            duration = transcript.get("duration")
            if not duration:
                segments = transcript.get("segments", [])
                duration = max((s.get("end", 0.0) for s in segments), default=0.0)

            use_chunking = chunk_seconds > 0 and duration > (chunk_seconds * 1.2)

            if not use_chunking:
                # Build the prompt
                user_prompt = ""
                if title:
                    user_prompt += f"Video Title: {title}\n\n"

                user_prompt += f"## Transcript\n\n{transcript_text}\n"

                if slides_text:
                    user_prompt += f"\n## Slide Content\n\n{slides_text}\n"

                if visual_text:
                    user_prompt += (
                        "\n## Visual Context (key frames, includes non-slide frames)\n\n"
                        f"{visual_text}\n"
                    )

                if merged_audio_hints:
                    user_prompt += "\n## Audio Source Hints\n\n"
                    user_prompt += "\n".join(f"- {hint}" for hint in merged_audio_hints)
                    user_prompt += "\n"

                user_prompt += "\nPlease provide a comprehensive summary in JSON format."

                if progress_callback:
                    progress_callback(10)

                response = (
                    _call_summarization_endpoint(
                        settings,
                        messages=[
                            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=4096,
                        temperature=0.3,
                    )
                    if use_endpoint
                    else model.create_chat_completion(
                        messages=[
                            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=4096,
                        temperature=0.3,
                    )
                )

                if progress_callback:
                    progress_callback(100)

                content = response["choices"][0]["message"]["content"]
                return parse_summary_json(content)

            if progress_callback:
                progress_callback(5)

            chunks = split_transcript_by_time(transcript, chunk_seconds, overlap_seconds)
            if not chunks:
                if progress_callback:
                    progress_callback(100)
                return {
                    "executive_summary": "",
                    "key_points": [],
                    "action_items": [],
                    "decisions": [],
                    "topics": [],
                }

            chunk_summaries = []
            total_chunks = len(chunks)

            for idx, chunk in enumerate(chunks, start=1):
                chunk_transcript = {
                    "segments": chunk["segments"],
                }
                chunk_text = format_transcript_for_summary(chunk_transcript, max_length=50000)
                chunk_slides = filter_slides_by_time(slides, chunk["start"], chunk["end"])
                chunk_slides_text = format_slides_for_summary(chunk_slides) if chunk_slides else ""
                chunk_frames = filter_frames_by_time(frames, chunk["start"], chunk["end"])
                chunk_visual_text = ""
                if chunk_frames:
                    per_chunk_frames = max(
                        5,
                        settings.visual_context_max_frames // max(total_chunks, 1),
                    )
                    per_chunk_chars = max(
                        1000,
                        settings.visual_context_max_chars // max(total_chunks, 1),
                    )
                    chunk_visual_text = format_visual_context(
                        chunk_frames,
                        max_chars=per_chunk_chars,
                        max_frames=per_chunk_frames,
                    )
                chunk_audio_hints = infer_audio_source_hints(chunk_frames)
                if audio_hints:
                    chunk_audio_hints = list(audio_hints) + [
                        hint for hint in chunk_audio_hints if hint not in audio_hints
                    ]

                user_prompt = ""
                if title:
                    user_prompt += f"Video Title: {title}\n\n"
                user_prompt += (
                    f"Chunk {idx} of {total_chunks}\n"
                    f"Time Range: {format_timestamp(chunk['start'])} - {format_timestamp(chunk['end'])}\n\n"
                    f"## Transcript\n\n{chunk_text}\n"
                )
                if chunk_slides_text:
                    user_prompt += f"\n## Slide Content\n\n{chunk_slides_text}\n"
                if chunk_visual_text:
                    user_prompt += (
                        "\n## Visual Context (key frames, includes non-slide frames)\n\n"
                        f"{chunk_visual_text}\n"
                    )
                if chunk_audio_hints:
                    user_prompt += "\n## Audio Source Hints\n\n"
                    user_prompt += "\n".join(f"- {hint}" for hint in chunk_audio_hints)
                    user_prompt += "\n"
                user_prompt += "\nPlease provide a comprehensive summary in JSON format."

                response = (
                    _call_summarization_endpoint(
                        settings,
                        messages=[
                            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=3072,
                        temperature=0.3,
                    )
                    if use_endpoint
                    else model.create_chat_completion(
                        messages=[
                            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=3072,
                        temperature=0.3,
                    )
                )
                content = response["choices"][0]["message"]["content"]
                chunk_summary = parse_summary_json(content)
                chunk_summary["_chunk_start"] = chunk["start"]
                chunk_summary["_chunk_end"] = chunk["end"]
                chunk_summaries.append(chunk_summary)

                if progress_callback:
                    progress = 5 + (70 * idx / total_chunks)
                    progress_callback(progress)

            merge_prompt = "Chunk summaries (JSON list):\n"
            merge_prompt += json.dumps(chunk_summaries, indent=2)
            merge_prompt += "\n\nPlease merge into a single summary JSON."

            response = (
                _call_summarization_endpoint(
                    settings,
                    messages=[
                        {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                        {"role": "user", "content": merge_prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.2,
                )
                if use_endpoint
                else model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                        {"role": "user", "content": merge_prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.2,
                )
            )
            content = response["choices"][0]["message"]["content"]

            if progress_callback:
                progress_callback(100)

            return parse_summary_json(content)

        return await loop.run_in_executor(None, do_summarize)

    finally:
        if manager:
            await manager.release_model(ModelType.SUMMARIZATION)


def format_transcript_for_summary(transcript: dict, max_length: int = 50000) -> str:
    """
    Format transcript for summarization, respecting token limits.

    Args:
        transcript: Merged transcript dict
        max_length: Maximum character length

    Returns:
        Formatted transcript string
    """
    segments = transcript.get("segments", [])
    lines = []
    total_length = 0

    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "").strip()
        timestamp = format_timestamp(segment.get("start", 0))

        line = f"[{timestamp}] {speaker}: {text}"
        line_length = len(line)

        if total_length + line_length > max_length:
            lines.append("... [transcript truncated for length]")
            break

        lines.append(line)
        total_length += line_length + 1

    return "\n".join(lines)


def format_slides_for_summary(slides: list[dict]) -> str:
    """Format slide content for summarization."""
    lines = []

    for idx, slide in enumerate(slides):
        timestamp = format_timestamp(slide.get("timestamp", 0))
        title = slide.get("title", f"Slide {idx + 1}")
        content = slide.get("content", "")
        key_points = slide.get("key_points", [])

        lines.append(f"### [{timestamp}] {title}")
        if content:
            lines.append(content[:500])  # Limit content length
        if key_points:
            for point in key_points[:5]:  # Limit points
                lines.append(f"- {point}")
        lines.append("")

    return "\n".join(lines)


def format_visual_context(
    frames: list[dict],
    max_chars: int = 8000,
    max_frames: int = 60,
) -> str:
    """Format non-slide and slide frames into a compact visual context section."""
    lines: list[str] = []
    total_length = 0
    count = 0

    ordered_frames = sorted(frames, key=lambda f: f.get("timestamp", 0.0))
    for frame in ordered_frames:
        if count >= max_frames:
            break
        description = (frame.get("description") or "").strip()
        ocr_text = (frame.get("ocr_text") or "").strip()
        key_points = frame.get("key_points") or []
        slide_title = frame.get("slide_title") or frame.get("title") or ""
        if not description and not ocr_text and not key_points:
            continue

        timestamp = format_timestamp(frame.get("timestamp", 0.0))
        is_slide = frame.get("is_slide")
        line_parts = [
            f"[{timestamp}] is_slide={is_slide}",
        ]
        if slide_title:
            line_parts.append(f"title: {slide_title}")
        if ocr_text:
            line_parts.append(f"ocr: {ocr_text}")
        if key_points:
            line_parts.append(f"key_points: {', '.join(str(p) for p in key_points[:5])}")
        if description:
            line_parts.append(f"description: {description}")

        line = "; ".join(line_parts)
        line = re.sub(r"\s+", " ", line).strip()
        if total_length + len(line) + 1 > max_chars:
            break

        lines.append(line)
        total_length += len(line) + 1
        count += 1

    return "\n".join(lines)


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


async def generate_topic_summary(
    transcript: dict,
    topic: str,
    start_time: float,
    end_time: float,
) -> str:
    """
    Generate a summary for a specific topic/section.

    Args:
        transcript: Full transcript
        topic: Topic name
        start_time: Section start time
        end_time: Section end time

    Returns:
        Summary string for the topic
    """
    # Extract relevant segments
    segments = transcript.get("segments", [])
    relevant_segments = [
        s for s in segments
        if s.get("start", 0) >= start_time and s.get("end", 0) <= end_time
    ]

    if not relevant_segments:
        return "No content found for this section."

    section_transcript = {"segments": relevant_segments}
    text = format_transcript_for_summary(section_transcript, max_length=5000)

    settings = get_settings()
    endpoint_url = _summarization_endpoint_url(settings)
    use_endpoint = bool(endpoint_url)
    manager = None
    model = None

    if not use_endpoint:
        manager = get_model_manager()
        model = await manager.get_model(ModelType.SUMMARIZATION)

    try:
        loop = asyncio.get_event_loop()

        def do_summarize():
            response = (
                _call_summarization_endpoint(
                    settings,
                    messages=[
                        {
                            "role": "system",
                            "content": "Summarize this section of a presentation/meeting in 2-3 sentences.",
                        },
                        {
                            "role": "user",
                            "content": f"Topic: {topic}\n\nTranscript:\n{text}",
                        },
                    ],
                    max_tokens=256,
                    temperature=0.3,
                )
                if use_endpoint
                else model.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "Summarize this section of a presentation/meeting in 2-3 sentences.",
                        },
                        {
                            "role": "user",
                            "content": f"Topic: {topic}\n\nTranscript:\n{text}",
                        },
                    ],
                    max_tokens=256,
                    temperature=0.3,
                )
            )
            return response["choices"][0]["message"]["content"]

        return await loop.run_in_executor(None, do_summarize)

    finally:
        if manager:
            await manager.release_model(ModelType.SUMMARIZATION)
