"""Audio hint extraction for lightweight source-context cues."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable

import torch
import torchaudio

from app.config import get_settings
from app.core.model_manager import ModelType, get_model_manager

logger = logging.getLogger(__name__)
PACKAGED_CLAP_RUNTIME_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "audio_event_calibration.json"
)


CLAP_PRIMARY_CANDIDATES = (
    {
        "key": "meeting_room_speech",
        "label": "meeting-room speech",
        "prompt": "in-person meeting room or office discussion recorded directly at the table",
        "prompt_variants": (
            "in-person meeting room or office discussion recorded directly at the table",
            "unproduced conference room conversation recorded in the room",
            "office meeting captured by a room microphone with coworkers talking back and forth",
            "conference room discussion with multiple people speaking naturally in the room",
        ),
        "hint": "Audio most likely sounds like direct meeting-room or office speech.",
        "summary_label": "direct meeting or office speech",
        "family": "direct_speech",
    },
    {
        "key": "broadcast_playback",
        "label": "broadcast playback",
        "prompt": "television news, sports broadcast, or talk-show audio playing from nearby speakers",
        "prompt_variants": (
            "television news, sports broadcast, or talk-show audio playing from nearby speakers",
            "tv news or sports audio coming from a television in the room",
            "news anchor or sports commentary playing through television speakers in a room",
            "talk show or news broadcast audio coming from nearby TV speakers",
        ),
        "hint": "Audio most likely sounds like television or broadcast playback rather than direct participant speech.",
        "summary_label": "broadcast or TV playback",
        "family": "playback",
    },
    {
        "key": "podcast_voiceover",
        "label": "podcast voice-over",
        "prompt": "produced podcast, voice-over, or studio narration",
        "prompt_variants": (
            "produced podcast, voice-over, or studio narration",
            "studio-quality voice-over or podcast host narration",
            "edited explainer or polished studio voice-over narration",
        ),
        "hint": "Audio most likely sounds like produced narration or podcast-style speech.",
        "summary_label": "produced narration or voice-over",
        "family": "produced_speech",
    },
    {
        "key": "software_demo",
        "label": "software demo narration",
        "prompt": "screen recording or software demo with a presenter narrating steps directly",
        "prompt_variants": (
            "screen recording or software demo with a presenter narrating steps directly",
            "screen-share software walkthrough with live spoken narration",
            "software tutorial with mouse clicks and narrated button-by-button steps",
            "screen recording tutorial with interface clicks and spoken software guidance",
        ),
        "hint": "Audio most likely sounds like a presenter directly narrating a software demo or walkthrough.",
        "summary_label": "direct software-demo narration",
        "family": "direct_speech",
    },
)

CLAP_SUPPORTING_CANDIDATES = (
    {
        "key": "music_heavy",
        "label": "music-heavy content",
        "prompt": "spoken narration with background music or soundtrack",
        "prompt_variants": (
            "spoken narration with background music or soundtrack",
            "corporate explainer narration with light underscore music",
            "video intro or outro with speech over music",
            "spoken presentation with faint background music",
        ),
        "hint": "Audio includes noticeable music or soundtrack backing.",
        "summary_label": "noticeable music bed or soundtrack",
        "family": "soundtrack",
    },
    {
        "key": "crowd_applause",
        "label": "crowd or applause",
        "prompt": "audience applause or crowd reaction",
        "prompt_variants": (
            "audience applause or crowd reaction",
            "live event cheering or clapping audience",
            "room applause after a talk or presentation",
        ),
        "hint": "Audio includes crowd, applause, or live-event reaction cues.",
        "summary_label": "crowd or applause cues",
        "family": "event",
    },
)

CLAP_HINT_CANDIDATES = CLAP_PRIMARY_CANDIDATES + CLAP_SUPPORTING_CANDIDATES
CLAP_CANDIDATE_BY_KEY = {candidate["key"]: candidate for candidate in CLAP_HINT_CANDIDATES}
CLAP_BROADCAST_KEYS = {"broadcast_playback"}
CLAP_SPEECH_KEYS = {"meeting_room_speech", "podcast_voiceover", "software_demo"}
CLAP_ORTHOGONAL_KEYS = {"music_heavy", "crowd_applause"}
DEFAULT_CLAP_SUPPORTING_RULES = {
    "music_heavy": {
        "aggregation": "top2_mean",
        "absolute_min_score": 0.07,
        "relative_ratio": 0.12,
    },
    "crowd_applause": {
        "aggregation": "top2_mean",
        "absolute_min_score": 0.09,
        "relative_ratio": 0.16,
    },
}


def build_default_clap_runtime_profile(min_score: float) -> dict:
    """Build the default runtime profile for CLAP prompt and threshold selection."""
    return {
        "version": 1,
        "primary_prompts": {
            candidate["key"]: candidate["prompt"]
            for candidate in CLAP_PRIMARY_CANDIDATES
        },
        "supporting_prompts": {
            candidate["key"]: list(candidate["prompt_variants"])
            for candidate in CLAP_SUPPORTING_CANDIDATES
        },
        "supporting_rules": {
            key: {
                **rule,
                "absolute_min_score": max(float(min_score) * 0.5, float(rule["absolute_min_score"])),
            }
            for key, rule in DEFAULT_CLAP_SUPPORTING_RULES.items()
        },
    }


def _deep_merge_profile(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_profile(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_profile_payload(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read CLAP calibration profile %s: %s", path, exc)
        return None

    if not isinstance(payload, dict):
        logger.warning("Ignoring CLAP calibration profile %s because it is not a JSON object.", path)
        return None

    return payload


def load_clap_runtime_profile(settings) -> dict:
    """Load a CLAP runtime profile from disk when present, otherwise use defaults."""
    profile = build_default_clap_runtime_profile(settings.audio_event_min_score)
    packaged_payload = _read_profile_payload(PACKAGED_CLAP_RUNTIME_PROFILE_PATH)
    if packaged_payload:
        profile = _deep_merge_profile(profile, packaged_payload)

    calibration_path = getattr(settings, "audio_event_calibration_path", None)
    if not calibration_path:
        return profile

    path = Path(calibration_path)
    if not path.exists() or path.resolve() == PACKAGED_CLAP_RUNTIME_PROFILE_PATH.resolve():
        return profile

    payload = _read_profile_payload(path)
    if payload:
        profile = _deep_merge_profile(profile, payload)

    return profile


def _selected_primary_candidates(profile: dict) -> tuple[dict, ...]:
    selected = []
    primary_prompts = profile.get("primary_prompts", {})
    for candidate in CLAP_PRIMARY_CANDIDATES:
        selected.append({
            **candidate,
            "prompt": str(primary_prompts.get(candidate["key"], candidate["prompt"])).strip()
            or candidate["prompt"],
        })
    return tuple(selected)


def _selected_supporting_prompt_specs(profile: dict) -> list[dict]:
    prompt_map = profile.get("supporting_prompts", {})
    prompt_specs: list[dict] = []

    for candidate in CLAP_SUPPORTING_CANDIDATES:
        raw_prompts = prompt_map.get(candidate["key"], candidate["prompt_variants"])
        prompts = raw_prompts if isinstance(raw_prompts, list) else list(candidate["prompt_variants"])
        normalized_prompts: list[str] = []
        for prompt in prompts:
            normalized = str(prompt).strip()
            if normalized and normalized not in normalized_prompts:
                normalized_prompts.append(normalized)
        if not normalized_prompts:
            normalized_prompts.append(candidate["prompt"])
        for prompt in normalized_prompts:
            prompt_specs.append({
                "key": candidate["key"],
                "label": candidate["label"],
                "prompt": prompt,
            })

    return prompt_specs


def _confidence_band(score: float, margin: float) -> str:
    if score >= 0.55 or margin >= 0.25:
        return "high"
    if score >= 0.32 or margin >= 0.12:
        return "medium"
    return "low"


def _aggregate_series(values: list[float], strategy: str) -> float:
    if not values:
        return 0.0
    normalized = strategy.strip().lower()
    if normalized == "max":
        return max(values)
    if normalized == "top2_mean":
        ranked = sorted(values, reverse=True)
        top_values = ranked[: min(2, len(ranked))]
        return float(sum(top_values) / len(top_values))
    return float(sum(values) / len(values))


def _move_inputs_to_device(inputs: dict, device: str) -> dict:
    if not (device == "cuda" or device.startswith("cuda:")):
        return inputs
    cuda_device = torch.device(device)
    return {
        key: value.to(cuda_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _run_clap_prompt_set(
    model,
    processor,
    device: str,
    waveform,
    target_sample_rate: int,
    prompt_texts: list[str],
) -> torch.Tensor:
    inputs = processor(
        text=prompt_texts,
        audios=waveform.squeeze(0).numpy(),
        sampling_rate=target_sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = _move_inputs_to_device(inputs, device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits_per_audio.detach().float().cpu(), dim=-1).squeeze(0)


def _score_clap_candidates(score_by_key: dict[str, float], candidates: tuple[dict, ...]) -> list[dict]:
    scored = []
    for candidate in candidates:
        scored.append(
            {
                "key": candidate["key"],
                "label": candidate["label"],
                "summary_label": candidate["summary_label"],
                "hint": candidate["hint"],
                "family": candidate["family"],
                "score": float(score_by_key.get(candidate["key"], 0.0)),
            }
        )
    return sorted(scored, key=lambda item: item["score"], reverse=True)


def _resolve_primary_clap_context(
    scored_candidates: list[dict],
    min_score: float,
) -> tuple[dict | None, dict[str, float]]:
    if not scored_candidates:
        return None, {}

    score_by_key = {item["key"]: item["score"] for item in scored_candidates}
    software_score = score_by_key.get("software_demo", 0.0)
    meeting_score = score_by_key.get("meeting_room_speech", 0.0)
    podcast_score = score_by_key.get("podcast_voiceover", 0.0)
    broadcast_score = score_by_key.get("broadcast_playback", 0.0)

    primary = scored_candidates[0]
    primary_key = primary["key"]
    primary_label = primary["summary_label"]
    primary_hint = primary["hint"]
    primary_score = primary["score"]

    if (
        software_score >= min_score
        and meeting_score >= min_score
        and abs(software_score - meeting_score) <= 0.12
        and max(software_score, meeting_score) >= max(podcast_score, broadcast_score)
    ):
        primary_key = "software_demo_direct_speech"
        primary_label = "direct software-demo narration"
        primary_hint = (
            "Audio most likely sounds like a presenter or participant directly narrating "
            "a software demo or walkthrough."
        )
        primary_score = max(software_score, meeting_score)
        primary_family = "direct_speech"
    else:
        primary_family = primary["family"]

    competing_scores = [
        item["score"]
        for item in scored_candidates
        if item["family"] in {"direct_speech", "produced_speech", "playback"}
        and item["key"] != primary_key
    ]
    competitor = max(competing_scores, default=0.0)
    confidence = _confidence_band(primary_score, primary_score - competitor)

    if primary_score < min_score:
        return None, score_by_key

    return (
        {
            "key": primary_key,
            "label": primary_label,
            "hint": primary_hint,
            "score": float(primary_score),
            "family": primary_family,
            "confidence": confidence,
        },
        score_by_key,
    )


def _select_supporting_clap_contexts(
    supporting_score_by_key: dict[str, float],
    primary_context: dict | None,
    profile: dict,
    min_score: float,
) -> list[dict]:
    if not primary_context:
        return []

    primary_score = float(primary_context["score"])
    supporting_rules = profile.get("supporting_rules", {})
    supporting = []

    ranked_supporting = sorted(
        supporting_score_by_key.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for key, score in ranked_supporting:
        candidate = CLAP_CANDIDATE_BY_KEY.get(key)
        if not candidate or key not in CLAP_ORTHOGONAL_KEYS:
            continue

        rule = supporting_rules.get(key, {})
        absolute_min = float(rule.get("absolute_min_score", min_score))
        relative_ratio = float(rule.get("relative_ratio", 0.35))
        cutoff = max(absolute_min, primary_score * relative_ratio)
        if score < cutoff:
            continue

        supporting.append(
            {
                "key": key,
                "label": candidate["summary_label"],
                "hint": candidate["hint"],
                "score": float(score),
            }
        )
        if len(supporting) >= 2:
            break

    return supporting


def _build_clap_audio_context(
    primary_score_by_key: dict[str, float],
    supporting_score_by_key: dict[str, float],
    profile: dict,
    min_score: float,
) -> dict:
    scored_primary = _score_clap_candidates(primary_score_by_key, CLAP_PRIMARY_CANDIDATES)
    primary_context, primary_scores = _resolve_primary_clap_context(scored_primary, min_score)
    supporting_contexts = _select_supporting_clap_contexts(
        supporting_score_by_key,
        primary_context,
        profile,
        min_score,
    )

    combined_labels = list(scored_primary)
    for candidate in CLAP_SUPPORTING_CANDIDATES:
        combined_labels.append(
            {
                "key": candidate["key"],
                "label": candidate["label"],
                "summary_label": candidate["summary_label"],
                "hint": candidate["hint"],
                "family": candidate["family"],
                "score": float(supporting_score_by_key.get(candidate["key"], 0.0)),
            }
        )
    combined_labels.sort(key=lambda item: item["score"], reverse=True)

    top_labels = [
        {"label": item["label"], "score": float(item["score"])}
        for item in combined_labels[: min(8, len(combined_labels))]
    ]

    hints: list[str] = []
    summary_parts: list[str] = []

    if primary_context:
        hints.append(str(primary_context["hint"]))
        summary_parts.append(
            f"Primary audio context: {primary_context['label']} "
            f"({primary_context['confidence']} confidence)."
        )

    if supporting_contexts:
        hints.extend(item["hint"] for item in supporting_contexts if item["hint"] not in hints)
        summary_parts.append(
            "Supporting audio cues: "
            + "; ".join(item["label"] for item in supporting_contexts)
            + "."
        )

    return {
        "hints": hints[:2],
        "top_labels": top_labels,
        "primary_context": primary_context,
        "supporting_contexts": supporting_contexts,
        "summary_context": " ".join(summary_parts).strip(),
        "tv_score": float(sum(primary_scores.get(key, 0.0) for key in CLAP_BROADCAST_KEYS)),
        "speech_score": float(sum(primary_scores.get(key, 0.0) for key in CLAP_SPEECH_KEYS)),
        "supporting_scores": supporting_score_by_key,
    }


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
            offsets = [
                max(0, min(offset, max(total_frames - sample_frames, 0)))
                for offset in offsets
            ]

            logits_accum = None
            sample_offsets_s: list[float] = []
            primary_window_scores = {
                candidate["key"]: []
                for candidate in CLAP_PRIMARY_CANDIDATES
            }
            supporting_window_scores = {
                candidate["key"]: []
                for candidate in CLAP_SUPPORTING_CANDIDATES
            }
            clap_profile = load_clap_runtime_profile(settings)
            primary_candidates = _selected_primary_candidates(clap_profile)
            supporting_prompt_specs = _selected_supporting_prompt_specs(clap_profile)

            if progress_callback:
                progress_callback(10)

            for idx, offset in enumerate(offsets, start=1):
                waveform, sr = _load_audio_segment(audio_path, offset, sample_frames)
                if sr != target_sample_rate and sr > 0:
                    resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                    waveform = resampler(waveform)

                if bundle_type == "audio_event_clap":
                    primary_probs = _run_clap_prompt_set(
                        model,
                        processor,
                        device,
                        waveform,
                        target_sample_rate,
                        [candidate["prompt"] for candidate in primary_candidates],
                    )
                    for prompt_idx, candidate in enumerate(primary_candidates):
                        primary_window_scores[candidate["key"]].append(
                            float(primary_probs[prompt_idx].item())
                        )

                    if supporting_prompt_specs:
                        supporting_probs = _run_clap_prompt_set(
                            model,
                            processor,
                            device,
                            waveform,
                            target_sample_rate,
                            [spec["prompt"] for spec in supporting_prompt_specs],
                        )
                        window_scores_by_key = {
                            candidate["key"]: []
                            for candidate in CLAP_SUPPORTING_CANDIDATES
                        }
                        for prompt_idx, spec in enumerate(supporting_prompt_specs):
                            window_scores_by_key[spec["key"]].append(
                                float(supporting_probs[prompt_idx].item())
                            )
                        for key, values in window_scores_by_key.items():
                            supporting_window_scores[key].append(max(values) if values else 0.0)
                else:
                    inputs = processor(
                        waveform.squeeze(0).numpy(),
                        sampling_rate=target_sample_rate,
                        return_tensors="pt",
                    )
                    inputs = _move_inputs_to_device(inputs, device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    logits = outputs.logits.detach().float().cpu()
                    logits_accum = logits if logits_accum is None else logits_accum + logits

                sample_offsets_s.append(offset / sample_rate)
                if progress_callback:
                    progress_callback(10 + (80 * idx / max(len(offsets), 1)))

            if bundle_type == "audio_event_clap":
                primary_score_by_key = {
                    key: _aggregate_series(scores, "mean")
                    for key, scores in primary_window_scores.items()
                }
                supporting_rules = clap_profile.get("supporting_rules", {})
                supporting_score_by_key = {}
                for key, scores in supporting_window_scores.items():
                    rule = supporting_rules.get(key, {})
                    supporting_score_by_key[key] = _aggregate_series(
                        scores,
                        str(rule.get("aggregation", "top2_mean")),
                    )

                clap_context = _build_clap_audio_context(
                    primary_score_by_key,
                    supporting_score_by_key,
                    clap_profile,
                    min_score=settings.audio_event_min_score,
                )
                hints = clap_context["hints"]
                top_labels = clap_context["top_labels"]
                tv_score = clap_context["tv_score"]
                speech_score = clap_context["speech_score"]
                primary_context = clap_context["primary_context"]
                supporting_contexts = clap_context["supporting_contexts"]
                summary_context = clap_context["summary_context"]
            else:
                if logits_accum is None:
                    return {
                        "hints": [],
                        "top_labels": [],
                        "tv_score": 0.0,
                        "speech_score": 0.0,
                        "primary_context": None,
                        "supporting_contexts": [],
                        "summary_context": "",
                        "sample_offsets_s": [],
                    }

                avg_logits = logits_accum / max(len(offsets), 1)
                probs = torch.softmax(avg_logits, dim=-1).squeeze(0)
                topk = min(8, probs.numel())
                scores, indices = torch.topk(probs, k=topk)
                top_labels = []
                id2label = model.config.id2label
                for score, item_idx in zip(scores.tolist(), indices.tolist(), strict=False):
                    label = id2label.get(item_idx, str(item_idx))
                    top_labels.append({"label": label, "score": float(score)})

                tv_score = 0.0
                speech_score = 0.0
                for item_idx, score in enumerate(probs.tolist()):
                    label = id2label.get(item_idx, "").lower()
                    if any(
                        keyword in label
                        for keyword in ("television", "tv", "broadcast", "radio", "news", "talk show")
                    ):
                        tv_score += score
                    if any(
                        keyword in label
                        for keyword in (
                            "speech",
                            "conversation",
                            "narration",
                            "monologue",
                            "lecture",
                            "meeting",
                            "presentation",
                        )
                    ):
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
                primary_context = None
                supporting_contexts = []
                summary_context = ""

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
                "primary_context": primary_context,
                "supporting_contexts": supporting_contexts,
                "summary_context": summary_context,
                "sample_offsets_s": sample_offsets_s,
            }

        return await loop.run_in_executor(None, do_classify)

    finally:
        await manager.release_model(ModelType.AUDIO_EVENT)
