"""Vision analysis module using Qwen3-Omni."""

import asyncio
import base64
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable

import httpx

from app.config import get_settings
from app.core.model_manager import ModelType, get_model_manager

logger = logging.getLogger(__name__)
ENDPOINT_STARTUP_WAIT_CAP_S = 90.0


def encode_image_base64(image_path: Path) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _vision_debug_enabled(settings) -> bool:
    return bool(settings.vision_debug or settings.debug)


def _vision_debug_dir(image_path: Path) -> Path:
    return image_path.parent.parent / "vision_debug"


def _image_fingerprint(image_path: Path) -> dict:
    try:
        data = image_path.read_bytes()
    except OSError as exc:
        return {"error": f"read_failed: {exc}"}
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _image_properties(image_path: Path) -> dict:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - debug-only path
        return {"error": f"PIL_unavailable: {exc}"}
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
            }
    except Exception as exc:  # pragma: no cover - debug-only path
        return {"error": f"open_failed: {exc}"}


def _write_vision_debug(
    image_path: Path,
    payload: dict,
    response_text: str | None,
    parsed: dict | None,
    error: str | None = None,
    extra: dict | None = None,
) -> None:
    debug_dir = _vision_debug_dir(image_path)
    debug_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "frame": image_path.name,
        "frame_path": str(image_path),
        "payload": payload,
        "response_text": response_text,
        "parsed": parsed,
        "error": error,
        "extra": extra or {},
    }
    output_path = debug_dir / f"{image_path.stem}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def _vision_endpoint_url(settings) -> str:
    url = settings.vision_endpoint_url.strip()
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    return f"{url.rstrip('/')}/chat/completions"


def _call_vision_endpoint(
    settings,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> dict:
    endpoint = _vision_endpoint_url(settings)
    if not endpoint:
        raise RuntimeError("Vision endpoint URL not configured.")
    headers = {}
    if settings.vision_endpoint_api_key:
        headers["Authorization"] = f"Bearer {settings.vision_endpoint_api_key}"
    payload = {
        "model": settings.vision_endpoint_model or settings.vision_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request_timeout = min(settings.vision_endpoint_timeout_s, 60.0)
    deadline = time.monotonic() + max(min(settings.vision_endpoint_timeout_s, ENDPOINT_STARTUP_WAIT_CAP_S), 5.0)
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
                "Vision endpoint request failed during startup; retrying in %.1fs (attempt %d).",
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
                "Vision endpoint is still loading; retrying in %.1fs (attempt %d).",
                retry_delay,
                attempt,
            )
            time.sleep(retry_delay)
            continue

        response.raise_for_status()
        return response.json()


def _normalize_vision_result(result: object) -> dict:
    if not isinstance(result, dict):
        return {
            "description": str(result),
            "ocr_text": "",
            "is_slide": False,
            "slide_title": None,
            "key_points": [],
        }

    normalized = dict(result)

    if "is_slide" not in normalized and "is_presentation_slide" in normalized:
        normalized["is_slide"] = normalized.get("is_presentation_slide")

    if "ocr_text" not in normalized:
        if "text_visible" in normalized:
            text_visible = normalized.get("text_visible")
            if isinstance(text_visible, list):
                normalized["ocr_text"] = "\n".join(str(t) for t in text_visible if t)
            else:
                normalized["ocr_text"] = str(text_visible or "")
        elif "ocr" in normalized:
            normalized["ocr_text"] = str(normalized.get("ocr") or "")
        elif "text" in normalized:
            normalized["ocr_text"] = str(normalized.get("text") or "")

    if "slide_title" not in normalized and "title" in normalized:
        normalized["slide_title"] = normalized.get("title")

    if "key_points" not in normalized:
        if "bullet_points" in normalized:
            normalized["key_points"] = normalized.get("bullet_points")
        elif "bullets" in normalized:
            normalized["key_points"] = normalized.get("bullets")

    if normalized.get("key_points") is None:
        normalized["key_points"] = []
    if not isinstance(normalized.get("key_points"), list):
        normalized["key_points"] = [normalized.get("key_points")]

    if normalized.get("is_slide") not in (True, False):
        normalized["is_slide"] = False

    if normalized.get("slide_title") is None:
        normalized["slide_title"] = None

    if normalized.get("ocr_text") is None:
        normalized["ocr_text"] = ""

    if normalized.get("description") is None:
        normalized["description"] = ""

    return normalized


async def analyze_frame(
    image_path: Path,
    context: str | None = None,
) -> dict:
    """
    Analyze a single frame/slide using the vision model.

    Args:
        image_path: Path to the image file
        context: Optional context about the video/presentation

    Returns:
        Analysis result:
        {
            "description": str,  # What's shown in the image
            "ocr_text": str,     # Extracted text
            "is_slide": bool,    # Whether this appears to be a slide
            "slide_title": str,  # Title if detected
            "key_points": list,  # Key points from the slide
        }
    """
    settings = get_settings()
    endpoint_url = _vision_endpoint_url(settings)
    use_endpoint = bool(endpoint_url)
    manager = None
    model = None
    if not use_endpoint:
        manager = get_model_manager()
        model = await manager.get_model(ModelType.VISION)

    try:
        loop = asyncio.get_event_loop()

        def do_analyze():
            # Encode image
            image_base64 = encode_image_base64(image_path)

            # Build prompt
            system_prompt = """You are an expert at analyzing presentation slides and video frames.
For each image, provide:
1. A brief description of what's shown
2. Any text visible in the image (OCR)
3. Whether this appears to be a presentation slide
4. If it's a slide, extract the title and key bullet points
If a TV/television or monitor showing video content is visible, mention it explicitly in the description.

Respond ONLY with valid JSON using this exact schema:
{
  "description": "string",
  "ocr_text": "string",
  "is_slide": true/false,
  "slide_title": "string or null",
  "key_points": ["string", ...]
}
If not a slide, set is_slide=false, slide_title=null, key_points=[].
Keep responses concise to fit within limited context windows."""

            user_prompt = "Analyze this image from a video presentation."
            if context:
                user_prompt += f"\n\nContext: {context}"

            # Create message with image
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                },
            ]

            # Generate response
            if use_endpoint:
                response = _call_vision_endpoint(
                    settings,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.1,
                )
            else:
                response = model.create_chat_completion(
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.1,
                )

            content = response["choices"][0]["message"]["content"]

            # Parse JSON response
            parse_error = None
            raw_result = None
            try:
                raw_result = json.loads(content)
            except json.JSONDecodeError as exc:
                # If not valid JSON, structure the response
                parse_error = f"json_decode_error: {exc}"
                raw_result = {
                    "description": content,
                    "ocr_text": "",
                    "is_slide": False,
                    "slide_title": None,
                    "key_points": [],
                }

            result = _normalize_vision_result(raw_result)

            if _vision_debug_enabled(settings):
                payload_meta = {
                    "model": settings.vision_endpoint_model or settings.vision_model,
                    "max_tokens": 1024,
                    "temperature": 0.1,
                    "endpoint": endpoint_url if use_endpoint else "local",
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "context_chars": len(context or ""),
                    "image_base64_chars": len(image_base64),
                    "image_fingerprint": _image_fingerprint(image_path),
                    "image_properties": _image_properties(image_path),
                }
                _write_vision_debug(
                    image_path=image_path,
                    payload=payload_meta,
                    response_text=content,
                    parsed=result,
                    error=parse_error,
                    extra={"raw_parsed": raw_result},
                )
                logger.info(
                    "Vision debug frame=%s is_slide=%s ocr_chars=%d desc_chars=%d",
                    image_path.name,
                    result.get("is_slide"),
                    len(result.get("ocr_text") or ""),
                    len(result.get("description") or ""),
                )

            return result

        return await loop.run_in_executor(None, do_analyze)

    finally:
        if manager:
            await manager.release_model(ModelType.VISION)


async def analyze_frames(
    frames: list[dict],
    context: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> list[dict]:
    """
    Analyze multiple frames/slides.

    Args:
        frames: List of frame info from extract_keyframes()
        context: Optional context about the video
        progress_callback: Progress callback (0-100)

    Returns:
        List of analyzed frames with added analysis data
    """
    results = []
    total = len(frames)
    settings = get_settings()
    endpoint = _vision_endpoint_url(settings)
    logger.info(
        "Starting vision analysis for %d frame(s) using %s",
        total,
        "endpoint" if endpoint else "local model",
    )

    for idx, frame in enumerate(frames):
        image_path = Path(frame["path"])

        if not image_path.exists():
            logger.warning(f"Frame not found: {image_path}")
            continue

        try:
            logger.info(
                "Analyzing frame %d/%d: %s",
                idx + 1,
                total,
                image_path.name,
            )
            analysis = await analyze_frame(image_path, context)

            results.append({
                **frame,
                **analysis,
            })

        except Exception as e:
            logger.error(f"Error analyzing frame {image_path}: {e}")
            results.append({
                **frame,
                "description": "Analysis failed",
                "ocr_text": "",
                "is_slide": False,
                "error": str(e),
            })
            if _vision_debug_enabled(settings):
                _write_vision_debug(
                    image_path=image_path,
                    payload={"context_chars": len(context or "")},
                    response_text=None,
                    parsed=None,
                    error=str(e),
                    extra={"stage": "analyze_frames"},
                )

        # Report progress
        if progress_callback:
            progress = ((idx + 1) / total) * 100
            progress_callback(progress)

    logger.info(f"Analyzed {len(results)} frames")
    return results


async def extract_slide_content(frames: list[dict]) -> list[dict]:
    """
    Extract and consolidate slide content from analyzed frames.

    Filters to only slides and merges consecutive duplicate slides.

    Args:
        frames: List of analyzed frames

    Returns:
        List of unique slide content:
        [{
            "timestamp": float,
            "title": str,
            "content": str,
            "key_points": list,
            "image_path": str,
        }]
    """
    slides = []
    prev_title = None
    settings = get_settings()
    debug = _vision_debug_enabled(settings)

    for frame in frames:
        if frame.get("low_relevance"):
            if debug:
                logger.info(
                    "Slide skip low_relevance frame=%s score=%s ocr_chars=%d",
                    Path(frame.get("path", "")).name,
                    frame.get("relevance_score"),
                    len(frame.get("ocr_text") or ""),
                )
            continue
        if not frame.get("is_slide", False):
            if debug:
                logger.info(
                    "Slide skip is_slide=False frame=%s ocr_chars=%d desc_chars=%d",
                    Path(frame.get("path", "")).name,
                    len(frame.get("ocr_text") or ""),
                    len(frame.get("description") or ""),
                )
            continue

        title = frame.get("slide_title") or frame.get("ocr_text", "")[:50]

        # Skip if same as previous slide
        if title and title == prev_title:
            if debug:
                logger.info(
                    "Slide skip duplicate title frame=%s title=%s",
                    Path(frame.get("path", "")).name,
                    title,
                )
            continue

        slides.append({
            "timestamp": frame.get("timestamp", 0),
            "title": title,
            "content": frame.get("ocr_text", ""),
            "ocr_text": frame.get("ocr_text", ""),
            "key_points": frame.get("key_points", []),
            "description": frame.get("description", ""),
            "image_path": frame.get("path", ""),
            "relevance_score": frame.get("relevance_score"),
        })

        prev_title = title

    logger.info(f"Extracted {len(slides)} unique slides")
    if debug and not slides:
        logger.info(
            "No slides extracted. Frame summary: %s",
            [
                {
                    "frame": Path(f.get("path", "")).name,
                    "is_slide": f.get("is_slide"),
                    "low_relevance": f.get("low_relevance"),
                    "relevance_score": f.get("relevance_score"),
                    "ocr_chars": len(f.get("ocr_text") or ""),
                    "desc_chars": len(f.get("description") or ""),
                }
                for f in frames
            ],
        )
    return slides


def annotate_frame_relevance(
    frames: list[dict],
    transcript: dict | None,
    window_seconds: float = 45.0,
    min_relevance_score: float = 0.05,
) -> list[dict]:
    """
    Annotate frames with a lightweight relevance score using nearby transcript text.

    Frames flagged as low relevance are skipped when extracting slide content.
    """
    if not transcript:
        return frames

    segments = transcript.get("segments", [])
    if not segments:
        return frames

    def tokenize(text: str) -> set[str]:
        return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}

    noise_markers = {
        "slack",
        "teams",
        "discord",
        "dm",
        "lunch",
        "coffee",
        "calendar",
        "chat",
        "inbox",
    }

    settings = get_settings()
    debug = _vision_debug_enabled(settings)

    for frame in frames:
        ocr_text = frame.get("ocr_text") or ""
        if not ocr_text.strip():
            frame["relevance_score"] = 0.0
            if debug:
                logger.info(
                    "Relevance frame=%s score=0.0 reason=no_ocr",
                    Path(frame.get("path", "")).name,
                )
            continue

        timestamp = frame.get("timestamp", 0.0)
        window_start = max(0.0, timestamp - window_seconds)
        window_end = timestamp + window_seconds
        window_text_parts = []

        for segment in segments:
            seg_start = segment.get("start", 0.0)
            seg_end = segment.get("end", seg_start)
            if seg_end < window_start or seg_start > window_end:
                continue
            window_text_parts.append(segment.get("text", ""))

        window_text = " ".join(window_text_parts)
        ocr_tokens = tokenize(ocr_text)
        window_tokens = tokenize(window_text)

        if not ocr_tokens or not window_tokens:
            score = 0.0
        else:
            overlap = ocr_tokens & window_tokens
            score = len(overlap) / max(len(ocr_tokens), 1)

        frame["relevance_score"] = round(score, 4)

        if score < min_relevance_score and (ocr_tokens & noise_markers):
            frame["low_relevance"] = True
        if debug:
            logger.info(
                "Relevance frame=%s score=%.4f ocr_tokens=%d window_tokens=%d overlap=%d low_relevance=%s",
                Path(frame.get("path", "")).name,
                score,
                len(ocr_tokens),
                len(window_tokens),
                len(ocr_tokens & window_tokens),
                frame.get("low_relevance", False),
            )

    return frames
