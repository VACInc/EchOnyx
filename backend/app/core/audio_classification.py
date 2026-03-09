"""Audio hint extraction for lightweight source-context cues."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

import torch
import torchaudio

from app.config import get_settings
from app.core.model_manager import ModelType, get_model_manager

logger = logging.getLogger(__name__)


CLAP_HINT_CANDIDATES = (
    {
        "key": "meeting_room_speech",
        "label": "meeting-room speech",
        "prompt": "near-field speech in a meeting room or office",
        "hint": "Audio sounds like direct meeting-room or office speech.",
    },
    {
        "key": "broadcast_playback",
        "label": "broadcast playback",
        "prompt": "broadcast television news or talk-show audio playing from speakers",
        "hint": "Audio sounds like broadcast or TV-style playback rather than participants speaking directly.",
    },
    {
        "key": "podcast_voiceover",
        "label": "podcast voice-over",
        "prompt": "podcast, voice-over, or studio narration",
        "hint": "Audio sounds like produced narration or podcast-style speech.",
    },
    {
        "key": "software_demo",
        "label": "software demo narration",
        "prompt": "screen recording or software demo with spoken narration",
        "hint": "Audio sounds like narrated software-demo or screencast audio.",
    },
    {
        "key": "music_heavy",
        "label": "music-heavy content",
        "prompt": "music-heavy entertainment audio",
        "hint": "Audio contains strong music or entertainment-style backing audio.",
    },
    {
        "key": "crowd_applause",
        "label": "crowd or applause",
        "prompt": "audience applause or crowd reaction",
        "hint": "Audio contains crowd or applause cues.",
    },
)

CLAP_BROADCAST_KEYS = {"broadcast_playback"}
CLAP_SPEECH_KEYS = {"meeting_room_speech", "podcast_voiceover", "software_demo"}


def _select_offsets(total_frames: int, sample_frames: int, num_samples: int) -> list[int]:
    if total_frames <= 0 or sample_frames <= 0:
        return [0]
    if total_frames <= sample_frames:
        return [0]
    if num_samples <= 1:
        return [max(0, (total_frames - sample_frames) // 2)]
    step = (total_frames - sample_frames) / max(num_samples - 1, 1)
    return [int(round(step * idx)) for idx in range(num_samples)]


def _load_audio_segment(
    audio_path: Path,
    offset_frames: int,
    num_frames: int,
) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(
        str(audio_path),
        frame_offset=offset_frames,
        num_frames=num_frames,
    )
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def _probe_audio_info(audio_path: Path) -> tuple[int, int]:
    info_fn = getattr(torchaudio, "info", None)
    if callable(info_fn):
        info = info_fn(str(audio_path))
        return int(info.num_frames), int(info.sample_rate)

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if hasattr(waveform, "shape"):
        total_frames = waveform.shape[-1]
    elif hasattr(waveform, "size"):
        total_frames = waveform.size(-1)
    else:
        total_frames = len(waveform.squeeze(0).numpy())
    return int(total_frames), int(sample_rate)


async def classify_audio_events(
    audio_path: Path,
    progress_callback: Callable[[float], None] | None = None,
) -> dict:
    """
    Run lightweight audio event classification and return hints + top labels.

    Returns:
        {
          "hints": [str, ...],
          "top_labels": [{"label": str, "score": float}, ...],
          "tv_score": float,
          "speech_score": float,
          "sample_offsets_s": [float, ...],
        }
    """
    settings = get_settings()
    manager = get_model_manager()
    model_bundle = await manager.get_model(ModelType.AUDIO_EVENT)

    try:
        loop = asyncio.get_event_loop()

        def do_classify() -> dict:
            model = model_bundle["model"]
            processor = model_bundle["processor"]
            device = model_bundle["device"]
            bundle_type = model_bundle.get("type", "audio_event_classifier")

            total_frames, sample_rate = _probe_audio_info(audio_path)
            target_sample_rate = getattr(
                getattr(processor, "feature_extractor", processor),
                "sampling_rate",
                sample_rate,
            )
            sample_seconds = settings.audio_event_sample_seconds
            sample_frames = int(sample_seconds * sample_rate)
            offsets = _select_offsets(total_frames, sample_frames, settings.audio_event_num_samples)
            offsets = [max(0, min(offset, max(total_frames - sample_frames, 0))) for offset in offsets]

            logits_accum = None
            sample_offsets_s: list[float] = []

            if progress_callback:
                progress_callback(10)

            for idx, offset in enumerate(offsets, start=1):
                waveform, sr = _load_audio_segment(audio_path, offset, sample_frames)
                if sr != target_sample_rate and sr > 0:
                    resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                    waveform = resampler(waveform)
                if bundle_type == "audio_event_clap":
                    inputs = processor(
                        text=[candidate["prompt"] for candidate in CLAP_HINT_CANDIDATES],
                        audios=waveform.squeeze(0).numpy(),
                        sampling_rate=target_sample_rate,
                        return_tensors="pt",
                        padding=True,
                    )
                else:
                    inputs = processor(
                        waveform.squeeze(0).numpy(),
                        sampling_rate=target_sample_rate,
                        return_tensors="pt",
                    )
                if device == "cuda":
                    inputs = {k: v.to(torch.device("cuda")) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                if bundle_type == "audio_event_clap":
                    logits = outputs.logits_per_audio.detach().float().cpu()
                else:
                    logits = outputs.logits.detach().float().cpu()
                logits_accum = logits if logits_accum is None else logits_accum + logits
                sample_offsets_s.append(offset / sample_rate)
                if progress_callback:
                    progress_callback(10 + (80 * idx / max(len(offsets), 1)))

            if logits_accum is None:
                return {
                    "hints": [],
                    "top_labels": [],
                    "tv_score": 0.0,
                    "speech_score": 0.0,
                    "sample_offsets_s": [],
                }

            avg_logits = logits_accum / max(len(offsets), 1)
            probs = torch.softmax(avg_logits, dim=-1).squeeze(0)

            topk = min(8, probs.numel())
            scores, indices = torch.topk(probs, k=topk)
            top_labels = []
            if bundle_type == "audio_event_clap":
                for score, idx in zip(scores.tolist(), indices.tolist(), strict=False):
                    candidate = CLAP_HINT_CANDIDATES[idx]
                    top_labels.append({"label": candidate["label"], "score": float(score)})

                tv_score = float(
                    sum(
                        probs[idx].item()
                        for idx, candidate in enumerate(CLAP_HINT_CANDIDATES)
                        if candidate["key"] in CLAP_BROADCAST_KEYS
                    )
                )
                speech_score = float(
                    sum(
                        probs[idx].item()
                        for idx, candidate in enumerate(CLAP_HINT_CANDIDATES)
                        if candidate["key"] in CLAP_SPEECH_KEYS
                    )
                )

                hints = []
                min_score = settings.audio_event_min_score
                for score, idx in zip(scores.tolist(), indices.tolist(), strict=False):
                    if float(score) < min_score:
                        continue
                    hint = CLAP_HINT_CANDIDATES[idx]["hint"]
                    if hint not in hints:
                        hints.append(hint)
                    if len(hints) >= 2:
                        break
            else:
                id2label = model.config.id2label
                for score, idx in zip(scores.tolist(), indices.tolist(), strict=False):
                    label = id2label.get(idx, str(idx))
                    top_labels.append({"label": label, "score": float(score)})

                tv_score = 0.0
                speech_score = 0.0
                for idx, score in enumerate(probs.tolist()):
                    label = id2label.get(idx, "").lower()
                    if any(keyword in label for keyword in ("television", "tv", "broadcast", "radio", "news", "talk show")):
                        tv_score += score
                    if any(keyword in label for keyword in ("speech", "conversation", "narration", "monologue", "lecture", "meeting", "presentation")):
                        speech_score += score

                hints = []
                min_score = settings.audio_event_min_score
                if tv_score >= min_score and tv_score > speech_score:
                    hints.append(
                        "Audio event classification suggests TV/broadcast content dominates the audio."
                    )
                elif speech_score >= min_score and speech_score >= tv_score:
                    hints.append(
                        "Audio event classification suggests near-field speech dominates the audio."
                    )

            if settings.audio_event_debug:
                logger.info(
                    "Audio event labels: top=%s tv_score=%.3f speech_score=%.3f offsets=%s",
                    top_labels,
                    tv_score,
                    speech_score,
                    sample_offsets_s,
                )

            if progress_callback:
                progress_callback(100)

            return {
                "hints": hints,
                "top_labels": top_labels,
                "tv_score": float(tv_score),
                "speech_score": float(speech_score),
                "sample_offsets_s": sample_offsets_s,
            }

        return await loop.run_in_executor(None, do_classify)

    finally:
        await manager.release_model(ModelType.AUDIO_EVENT)
