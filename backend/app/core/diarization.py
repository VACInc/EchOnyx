"""Speaker diarization module using pyannote-audio."""

import asyncio
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

from app.config import GPUBackend, HardwareProfile, get_settings
from app.core.model_manager import ModelType, get_model_manager

logger = logging.getLogger(__name__)


def _is_gpu_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ("miopen", "hip", "rocm"))


def _should_retry_on_cpu_after_gpu_error(exc: RuntimeError) -> bool:
    if not _is_gpu_runtime_error(exc):
        return False
    settings = get_settings()
    return not (
        settings.hardware_profile == HardwareProfile.STRIX_HALO
        and settings.gpu_backend == GPUBackend.ROCM
    )


def _skipped_diarization_result(reason: str) -> dict:
    return {
        "speakers": [],
        "segments": [],
        "num_speakers": 0,
        "skipped": True,
        "reason": reason,
    }


def _run_diarization_pipeline(pipeline, waveform, sample_rate: int, params: dict, torch_module):
    if hasattr(pipeline, "eval"):
        pipeline.eval()

    inference_mode = getattr(torch_module, "inference_mode", None)
    inference_context = inference_mode() if callable(inference_mode) else nullcontext()

    cudnn_context = nullcontext()
    settings = get_settings()
    cudnn_backend = getattr(getattr(torch_module, "backends", None), "cudnn", None)
    cudnn_flags = getattr(cudnn_backend, "flags", None)
    if settings.gpu_backend == GPUBackend.ROCM and callable(cudnn_flags):
        cudnn_context = cudnn_flags(enabled=False)

    with cudnn_context:
        with inference_context:
            return pipeline({"waveform": waveform, "sample_rate": sample_rate}, **params)


async def diarize_audio(
    audio_path: Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict:
    """
    Perform speaker diarization on audio file.

    Args:
        audio_path: Path to the audio file
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum expected speakers
        max_speakers: Maximum expected speakers
        progress_callback: Optional callback for progress updates (0-100)

    Returns:
        Dictionary with diarization results:
        {
            "speakers": [
                {
                    "id": str,        # Speaker identifier (e.g., "SPEAKER_00")
                    "name": str,      # Display name
                    "total_time": float,  # Total speaking time in seconds
                }
            ],
            "segments": [
                {
                    "start": float,   # Start time in seconds
                    "end": float,     # End time in seconds
                    "speaker": str,   # Speaker identifier
                }
            ],
            "num_speakers": int,  # Number of detected speakers
        }
    """
    manager = get_model_manager()
    try:
        pipeline = await manager.get_model(ModelType.DIARIZATION)
    except ValueError as exc:
        if "HF_TOKEN is required for pyannote models" in str(exc):
            logger.warning("Skipping diarization because HF_TOKEN is not configured.")
            if progress_callback:
                progress_callback(100)
            return _skipped_diarization_result("missing_hf_token")
        raise

    try:
        loop = asyncio.get_event_loop()

        def do_diarize():
            # Configure pipeline parameters
            params = {}
            if num_speakers is not None:
                params["num_speakers"] = num_speakers
            if min_speakers is not None:
                params["min_speakers"] = min_speakers
            if max_speakers is not None:
                params["max_speakers"] = max_speakers

            # Run diarization
            if progress_callback:
                progress_callback(10)  # Started
            import numpy as np
            import soundfile as sf
            import torch

            audio_array, sample_rate = sf.read(str(audio_path))
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)

            target_sr = 16000
            if sample_rate and sample_rate != target_sr:
                from scipy.signal import resample_poly

                audio_array = resample_poly(audio_array, target_sr, sample_rate)
                sample_rate = target_sr

            waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
            try:
                diarization = _run_diarization_pipeline(
                    pipeline,
                    waveform,
                    sample_rate,
                    params,
                    torch,
                )
            except RuntimeError as exc:
                if _should_retry_on_cpu_after_gpu_error(exc):
                    logger.warning(
                        "Diarization GPU failed (%s); retrying on CPU.",
                        exc,
                    )
                    pipeline.to(torch.device("cpu"))
                    diarization = _run_diarization_pipeline(
                        pipeline,
                        waveform,
                        sample_rate,
                        params,
                        torch,
                    )
                elif _is_gpu_runtime_error(exc):
                    logger.error(
                        "Diarization GPU failed and CPU fallback is disabled for this runtime: %s",
                        exc,
                    )
                    raise
                else:
                    raise

            if progress_callback:
                progress_callback(80)  # Processing done

            # Extract results
            segments = []
            speaker_times: dict[str, float] = {}

            # Handle different pyannote output formats
            # The community model returns DiarizeOutput dataclass with speaker_diarization attribute
            annotation = diarization
            if hasattr(diarization, 'speaker_diarization'):
                annotation = diarization.speaker_diarization
            elif hasattr(diarization, 'exclusive_speaker_diarization'):
                annotation = diarization.exclusive_speaker_diarization
            elif hasattr(diarization, 'annotation'):
                annotation = diarization.annotation
            elif hasattr(diarization, 'get_annotation'):
                annotation = diarization.get_annotation()

            # Debug logging
            logger.info(f"Diarization output type: {type(diarization)}")
            logger.info(f"Annotation type: {type(annotation)}")

            if hasattr(annotation, 'itertracks'):
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    segment = {
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker,
                    }
                    segments.append(segment)

                    # Track speaking time per speaker
                    duration = turn.end - turn.start
                    speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
            elif hasattr(diarization, '__iter__'):
                # Try iterating directly if it's iterable
                for item in diarization:
                    if hasattr(item, 'start') and hasattr(item, 'end') and hasattr(item, 'speaker'):
                        segment = {
                            "start": item.start,
                            "end": item.end,
                            "speaker": item.speaker,
                        }
                        segments.append(segment)
                        duration = item.end - item.start
                        speaker_times[item.speaker] = speaker_times.get(item.speaker, 0) + duration
            else:
                logger.warning(f"Unknown diarization output format: {type(diarization)}, attrs: {dir(diarization)}")

            # Build speaker list
            speakers = []
            for i, (speaker_id, total_time) in enumerate(sorted(speaker_times.items())):
                speakers.append({
                    "id": speaker_id,
                    "name": f"Speaker {i + 1}",  # Default name
                    "total_time": round(total_time, 2),
                })

            if progress_callback:
                progress_callback(100)

            return {
                "speakers": speakers,
                "segments": segments,
                "num_speakers": len(speakers),
            }

        result = await loop.run_in_executor(None, do_diarize)
        logger.info(f"Diarization complete: {result['num_speakers']} speakers detected")
        return result

    finally:
        await manager.release_model(ModelType.DIARIZATION)


def merge_transcript_with_diarization(
    transcript: dict,
    diarization: dict,
) -> dict:
    """
    Merge transcription segments with speaker labels from diarization.

    Args:
        transcript: Output from transcribe_audio()
        diarization: Output from diarize_audio()

    Returns:
        Merged transcript with speaker labels:
        {
            "text": str,
            "segments": [
                {
                    "start": float,
                    "end": float,
                    "text": str,
                    "speaker": str,  # Added speaker label
                }
            ],
            "speakers": [...],
            "language": str,
        }
    """
    diarization_segments = diarization.get("segments", [])

    def find_speaker(start: float, end: float) -> str | None:
        """Find the speaker for a given time range."""
        best_overlap = 0
        best_speaker = None

        for d_seg in diarization_segments:
            # Calculate overlap
            overlap_start = max(start, d_seg["start"])
            overlap_end = min(end, d_seg["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg["speaker"]

        return best_speaker

    # Add speaker labels to transcript segments
    merged_segments = []
    for segment in transcript.get("segments", []):
        speaker = find_speaker(segment["start"], segment["end"])
        merged_segments.append({
            **segment,
            "speaker": speaker,
        })

    return {
        "text": transcript.get("text", ""),
        "segments": merged_segments,
        "speakers": diarization.get("speakers", []),
        "language": transcript.get("language", "en"),
        "duration": transcript.get("duration", 0),
    }
