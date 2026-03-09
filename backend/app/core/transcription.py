"""Transcription module using faster-whisper or Granite Speech."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from app.core.model_manager import ModelType, get_model_manager

logger = logging.getLogger(__name__)


def _words_to_text(words: list[dict]) -> str:
    return "".join(word.get("word", "") for word in words).strip()


def _dedupe_words_by_timestamp(words: list[dict], tolerance_s: float) -> list[dict]:
    deduped: list[dict] = []
    last_end: float | None = None
    for word in words:
        start = float(word.get("start", 0.0))
        end = float(word.get("end", 0.0))
        if last_end is None or start >= (last_end - tolerance_s):
            deduped.append(word)
            last_end = max(last_end or 0.0, end)
        elif end > (last_end + tolerance_s):
            deduped.append(word)
            last_end = end
    return deduped


async def transcribe_audio(
    audio_path: Path,
    language: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict:
    """
    Transcribe audio file using Whisper.

    Args:
        audio_path: Path to the audio file
        language: Optional language code (auto-detected if None)
        progress_callback: Optional callback for progress updates (0-100)

    Returns:
        Dictionary with transcription results:
        {
            "text": str,  # Full transcript text
            "segments": [
                {
                    "start": float,  # Start time in seconds
                    "end": float,    # End time in seconds
                    "text": str,     # Segment text
                    "confidence": float,  # Optional confidence score
                }
            ],
            "language": str,  # Detected or specified language
        }
    """
    manager = get_model_manager()
    model = await manager.get_model(ModelType.WHISPER)

    try:
        loop = asyncio.get_event_loop()

        if isinstance(model, dict):
            model_type = model.get("type")
            if model_type in {"granite", "whisper_transformers", "nemo_canary"}:
                result = await _transcribe_transformers_asr(
                    model,
                    audio_path,
                    language=language,
                    progress_callback=progress_callback,
                )
            else:
                result = await _transcribe_with_faster_whisper(
                    audio_path,
                    None,
                    language=language,
                    progress_callback=progress_callback,
                    model_override=model,
                )
        else:
            result = await _transcribe_with_faster_whisper(
                audio_path,
                None,
                language=language,
                progress_callback=progress_callback,
                model_override=model,
            )
        logger.info(f"Transcription complete: {len(result['segments'])} segments")
        return result

    finally:
        # Release model if in sequential mode
        await manager.release_model(ModelType.WHISPER)


async def get_audio_duration(audio_path: Path) -> float:
    """Get the duration of an audio file in seconds."""
    import ffmpeg

    loop = asyncio.get_event_loop()

    def probe():
        try:
            info = ffmpeg.probe(str(audio_path))
            return float(info["format"]["duration"])
        except Exception as e:
            logger.warning(f"Could not probe audio duration: {e}")
            return 0.0

    return await loop.run_in_executor(None, probe)


async def _transcribe_transformers_asr(
    model_bundle: dict,
    audio_path: Path,
    language: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict:
    """Transcribe audio using transformers ASR models (Granite or Whisper)."""
    from transformers import pipeline
    import torch

    if model_bundle.get("type") == "nemo_canary":
        return await _transcribe_nemo_canary(
            model_bundle,
            audio_path,
            language=language,
            progress_callback=progress_callback,
        )

    processor = model_bundle["processor"]
    model = model_bundle["model"]
    device_name = model_bundle.get("device", "cpu")
    settings = get_settings()

    device_id = 0 if device_name == "cuda" and torch.cuda.is_available() else -1
    if model_bundle.get("type") == "granite":
        import numpy as np
        import soundfile as sf

        loop = asyncio.get_event_loop()

        def run_granite():
            audio_array, sample_rate = sf.read(str(audio_path))
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)

            target_sr = getattr(
                getattr(processor, "audio_processor", None),
                "sampling_rate",
                16000,
            )
            if sample_rate and sample_rate != target_sr:
                from scipy.signal import resample_poly

                audio_array = resample_poly(audio_array, target_sr, sample_rate)
                sample_rate = target_sr

            chunk_length_s = settings.asr_chunk_length_s
            chunk_samples = int(chunk_length_s * sample_rate)
            texts: list[str] = []

            for start in range(0, len(audio_array), chunk_samples):
                chunk = audio_array[start:start + chunk_samples]
                prompt = getattr(processor, "audio_token", "<|audio|>")
                inputs = processor(
                    text=prompt,
                    audio=chunk,
                    return_tensors="pt",
                )
                if device_name == "cuda" and torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        use_cache=False,
                    )
                chunk_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0].strip()
                if chunk_text:
                    texts.append(chunk_text)
                if progress_callback:
                    progress = min(100, ((start + len(chunk)) / len(audio_array)) * 100)
                    progress_callback(progress)

            full_text = " ".join(texts).strip()
            duration = len(audio_array) / sample_rate if sample_rate else 0.0
            segments_list = [{
                "start": 0.0,
                "end": duration,
                "text": full_text,
                "words": [],
            }] if full_text else []
            detected_language = language or "en"

            return {
                "text": full_text,
                "segments": segments_list,
                "language": detected_language,
                "duration": duration,
            }

        return await loop.run_in_executor(None, run_granite)
    pipeline_kwargs: dict[str, Any] = {
        "model": model,
        "device": device_id,
    }
    if "tokenizer" in model_bundle and "feature_extractor" in model_bundle:
        pipeline_kwargs["tokenizer"] = model_bundle["tokenizer"]
        pipeline_kwargs["feature_extractor"] = model_bundle["feature_extractor"]
    elif hasattr(processor, "feature_extractor") and hasattr(processor, "tokenizer"):
        pipeline_kwargs["tokenizer"] = processor.tokenizer
        pipeline_kwargs["feature_extractor"] = processor.feature_extractor
    else:
        pipeline_kwargs["processor"] = processor
    asr = pipeline("automatic-speech-recognition", **pipeline_kwargs)

    loop = asyncio.get_event_loop()

    def run_pipeline():
        call_kwargs: dict[str, Any] = {}
        if model_bundle.get("type") == "whisper_transformers":
            call_kwargs["chunk_length_s"] = settings.asr_chunk_length_s
            call_kwargs["stride_length_s"] = settings.asr_chunk_overlap_s
            call_kwargs["return_timestamps"] = "word"
        import numpy as np
        import soundfile as sf

        audio_array, sample_rate = sf.read(str(audio_path))
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        inputs = {"array": audio_array, "sampling_rate": sample_rate}
        return asr(inputs, **call_kwargs)

    result = await loop.run_in_executor(None, run_pipeline)

    text = result.get("text", "").strip()
    chunks = result.get("chunks") or []
    segments_list: list[dict] = []
    words: list[dict] = []

    for chunk in chunks:
        timestamp = chunk.get("timestamp") or chunk.get("timestamps")
        if not timestamp or len(timestamp) != 2:
            continue
        start, end = timestamp
        word = chunk.get("text", "").strip()
        if not word:
            continue
        words.append({
            "word": word,
            "start": float(start),
            "end": float(end),
        })

    if words:
        words = _dedupe_words_by_timestamp(words, settings.asr_dedupe_tolerance_s)
        current = {
            "start": words[0]["start"],
            "end": words[0]["end"],
            "text": words[0]["word"],
            "words": [words[0]],
        }
        for word in words[1:]:
            gap = word["start"] - current["end"]
            duration = word["end"] - current["start"]
            if gap > 1.5 or duration > 15.0:
                segments_list.append(current)
                current = {
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["word"],
                    "words": [word],
                }
            else:
                current["end"] = word["end"]
                current["words"].append(word)
                current["text"] = _words_to_text(current["words"])
        segments_list.append(current)
    else:
        duration = await get_audio_duration(audio_path)
        segments_list = [{
            "start": 0.0,
            "end": duration,
            "text": text,
            "words": [],
        }]

    duration = max((segment["end"] for segment in segments_list), default=0.0)
    detected_language = result.get("language") or language or "en"

    if progress_callback and duration > 0:
        for segment in segments_list:
            progress = min(100, (segment["end"] / duration) * 100)
            progress_callback(progress)

    merged_text = text or " ".join([s["text"] for s in segments_list]).strip()
    return {
        "text": merged_text,
        "segments": segments_list,
        "language": detected_language,
        "duration": duration,
    }


async def _transcribe_with_faster_whisper(
    audio_path: Path,
    model_name: str | None,
    language: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
    model_override: Any | None = None,
) -> dict:
    """Transcribe audio using faster-whisper, optionally with a preloaded model."""
    settings = get_settings()

    model = model_override
    if model is None:
        from faster_whisper import WhisperModel

        chosen = model_name or settings.whisper_model
        if settings.gpu_backend.value == "cuda":
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "float32"
        loop = asyncio.get_event_loop()

        model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                chosen,
                device=device,
                compute_type=compute_type,
                download_root=str(settings.model_cache_dir),
            ),
        )

    loop = asyncio.get_event_loop()

    def do_transcribe():
        segments_list = []
        full_text_parts = []

        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
        )

        duration = info.duration if hasattr(info, "duration") else 0
        detected_language = info.language if hasattr(info, "language") else language or "en"
        tolerance = get_settings().asr_dedupe_tolerance_s
        last_word_end: float | None = None

        for segment in segments:
            raw_words = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                }
                for w in (segment.words or [])
            ]
            deduped_words: list[dict] = []
            for word in raw_words:
                start = float(word["start"])
                end = float(word["end"])
                if last_word_end is None or start >= (last_word_end - tolerance):
                    deduped_words.append(word)
                    last_word_end = max(last_word_end or 0.0, end)
                elif end > (last_word_end + tolerance):
                    deduped_words.append(word)
                    last_word_end = end

            if deduped_words:
                segment_text = _words_to_text(deduped_words)
                segments_list.append({
                    "start": deduped_words[0]["start"],
                    "end": deduped_words[-1]["end"],
                    "text": segment_text or segment.text.strip(),
                    "words": deduped_words,
                })
                full_text_parts.append(segment_text or segment.text.strip())
            else:
                if segment.text.strip():
                    segments_list.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "words": [],
                    })
                    full_text_parts.append(segment.text.strip())

            if progress_callback and duration > 0:
                progress = min(100, (segment.end / duration) * 100)
                progress_callback(progress)

        return {
            "text": " ".join(full_text_parts).strip(),
            "segments": segments_list,
            "language": detected_language,
            "duration": duration,
        }

    return await loop.run_in_executor(None, do_transcribe)


async def _transcribe_nemo_canary(
    model_bundle: dict,
    audio_path: Path,
    language: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict:
    """Transcribe audio using NVIDIA Canary (NeMo SALM)."""
    import tempfile
    import numpy as np
    import soundfile as sf

    model = model_bundle["model"]
    settings = get_settings()

    loop = asyncio.get_event_loop()

    def run_canary():
        audio_array, sample_rate = sf.read(str(audio_path))
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        target_sr = 16000
        if sample_rate and sample_rate != target_sr:
            from scipy.signal import resample_poly

            audio_array = resample_poly(audio_array, target_sr, sample_rate)
            sample_rate = target_sr

        chunk_length_s = settings.asr_chunk_length_s
        overlap_s = settings.asr_chunk_overlap_s
        chunk_samples = int(chunk_length_s * sample_rate)
        step_samples = max(1, int((chunk_length_s - overlap_s) * sample_rate))
        total_len = len(audio_array)
        texts: list[str] = []

        def dedupe_overlap(prev: str, new: str, max_words: int = 12) -> str:
            prev_words = prev.strip().split()
            new_words = new.strip().split()
            max_k = min(max_words, len(prev_words), len(new_words))
            for k in range(max_k, 0, -1):
                if prev_words[-k:] == new_words[:k]:
                    return " ".join(new_words[k:])
            return new

        for start in range(0, total_len, step_samples):
            end = min(start + chunk_samples, total_len)
            chunk = audio_array[start:end]
            if len(chunk) == 0:
                continue
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                dir=str(audio_path.parent),
            ) as tmp:
                sf.write(tmp.name, chunk, sample_rate)
                prompt = [{
                    "role": "user",
                    "content": f"Transcribe the following: {model.audio_locator_tag}",
                    "audio": [tmp.name],
                }]
                ids = model.generate(prompts=[prompt], max_new_tokens=256)

            text = None
            tok = getattr(model, "tokenizer", None)
            if tok is not None:
                try:
                    text = tok.ids_to_text(ids[0].tolist())
                except Exception:
                    text = None

            if text:
                if texts:
                    text = dedupe_overlap(texts[-1], text)
                if text:
                    texts.append(text.strip())

            if progress_callback and total_len > 0:
                progress = min(100, (end / total_len) * 100)
                progress_callback(progress)

        full_text = " ".join(texts).strip()
        duration = total_len / sample_rate if sample_rate else 0.0
        segments_list = [{
            "start": 0.0,
            "end": duration,
            "text": full_text,
            "words": [],
        }] if full_text else []

        return {
            "text": full_text,
            "segments": segments_list,
            "language": language or "en",
            "duration": duration,
        }

    return await loop.run_in_executor(None, run_canary)
