"""Fixture-driven calibration for CLAP audio-event scoring."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from app.config import get_settings
from app.core.audio_classification import (
    CLAP_PRIMARY_CANDIDATES,
    CLAP_SUPPORTING_CANDIDATES,
    _aggregate_series,
    _load_audio_segment,
    _probe_audio_info,
    _run_clap_prompt_set,
    _selected_primary_candidates,
    _selected_supporting_prompt_specs,
    _select_offsets,
    build_default_clap_runtime_profile,
)
from app.core.model_manager import ModelType, get_model_manager
from app.utils.ffmpeg import extract_audio

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
PRIMARY_AGGREGATION = "mean"
SUPPORTING_AGGREGATIONS = ("mean", "max", "top2_mean")
ABSOLUTE_THRESHOLD_GRID = (0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2)
RELATIVE_RATIO_GRID = (0.05, 0.08, 0.1, 0.12, 0.16, 0.2)


@dataclass(frozen=True)
class AudioCalibrationFixture:
    media_path: Path
    expected_primary_key: str | None
    expected_supporting_keys: tuple[str, ...]
    label: str


def load_audio_calibration_manifest(manifest_path: Path) -> list[AudioCalibrationFixture]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures_raw = payload.get("fixtures")
    if not isinstance(fixtures_raw, list) or not fixtures_raw:
        raise ValueError("Manifest must contain a non-empty 'fixtures' list.")

    fixtures: list[AudioCalibrationFixture] = []
    for idx, item in enumerate(fixtures_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Fixture #{idx} must be an object.")
        raw_media_path = item.get("media_path") or item.get("audio_path") or item.get("video_path")
        if not raw_media_path:
            raise ValueError(f"Fixture #{idx} is missing 'media_path'.")
        media_path = Path(str(raw_media_path))
        if not media_path.is_absolute():
            media_path = (manifest_path.parent / media_path).resolve()

        expected_supporting_raw = item.get("expected_supporting_keys") or []
        if not isinstance(expected_supporting_raw, list):
            raise ValueError(f"Fixture #{idx} expected_supporting_keys must be a list.")

        fixtures.append(
            AudioCalibrationFixture(
                media_path=media_path,
                expected_primary_key=str(item["expected_primary_key"]).strip()
                if item.get("expected_primary_key")
                else None,
                expected_supporting_keys=tuple(
                    sorted(str(value).strip() for value in expected_supporting_raw if str(value).strip())
                ),
                label=str(item.get("label") or media_path.stem),
            )
        )

    return fixtures


async def _materialize_audio_path(
    media_path: Path,
    scratch_dir: Path,
) -> Path:
    if media_path.suffix.lower() in AUDIO_EXTENSIONS:
        return media_path
    audio_path = scratch_dir / f"{media_path.stem}.wav"
    await extract_audio(media_path, audio_path)
    return audio_path


async def collect_clap_fixture_observations(
    fixtures: list[AudioCalibrationFixture],
    *,
    scratch_dir: Path | None = None,
) -> list[dict]:
    settings = get_settings()
    manager = get_model_manager()
    model_bundle = await manager.get_model(ModelType.AUDIO_EVENT)
    if model_bundle.get("type") != "audio_event_clap":
        raise RuntimeError("Audio calibration requires a CLAP audio-event model.")

    runtime_profile = build_default_clap_runtime_profile(settings.audio_event_min_score)
    primary_candidates = _selected_primary_candidates(runtime_profile)
    supporting_prompt_specs = _selected_supporting_prompt_specs(runtime_profile)
    loop = asyncio.get_event_loop()
    owned_scratch_dir = None
    if scratch_dir is None:
        owned_scratch_dir = Path(tempfile.mkdtemp(prefix="echonyx-audio-calibration-"))
        scratch_dir = owned_scratch_dir
    scratch_dir.mkdir(parents=True, exist_ok=True)

    try:
        observations: list[dict] = []

        for fixture in fixtures:
            audio_path = await _materialize_audio_path(fixture.media_path, scratch_dir)

            def do_score_fixture() -> dict:
                model = model_bundle["model"]
                processor = model_bundle["processor"]
                device = model_bundle["device"]

                total_frames, sample_rate = _probe_audio_info(audio_path)
                target_sample_rate = getattr(
                    getattr(processor, "feature_extractor", processor),
                    "sampling_rate",
                    sample_rate,
                )
                sample_frames = int(settings.audio_event_sample_seconds * sample_rate)
                offsets = _select_offsets(total_frames, sample_frames, settings.audio_event_num_samples)
                offsets = [
                    max(0, min(offset, max(total_frames - sample_frames, 0)))
                    for offset in offsets
                ]

                primary_window_scores = {candidate["key"]: [] for candidate in CLAP_PRIMARY_CANDIDATES}
                supporting_prompt_scores = {
                    candidate["key"]: {
                        prompt: []
                        for prompt in candidate["prompt_variants"]
                    }
                    for candidate in CLAP_SUPPORTING_CANDIDATES
                }

                for offset in offsets:
                    waveform, sr = _load_audio_segment(audio_path, offset, sample_frames)
                    if sr != target_sample_rate and sr > 0:
                        resampler = __import__("torchaudio").transforms.Resample(sr, target_sample_rate)
                        waveform = resampler(waveform)

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
                        for prompt_idx, spec in enumerate(supporting_prompt_specs):
                            supporting_prompt_scores[spec["key"]].setdefault(spec["prompt"], []).append(
                                float(supporting_probs[prompt_idx].item())
                            )

                return {
                    "label": fixture.label,
                    "media_path": str(fixture.media_path),
                    "expected_primary_key": fixture.expected_primary_key,
                    "expected_supporting_keys": list(fixture.expected_supporting_keys),
                    "primary_window_scores": primary_window_scores,
                    "supporting_prompt_scores": supporting_prompt_scores,
                }

            observations.append(await loop.run_in_executor(None, do_score_fixture))

        return observations
    finally:
        await manager.release_model(ModelType.AUDIO_EVENT)
        if owned_scratch_dir:
            for path in owned_scratch_dir.glob("*"):
                path.unlink(missing_ok=True)
            owned_scratch_dir.rmdir()


def _safe_mean(values: list[float]) -> float:
    return float(fmean(values)) if values else 0.0


def _separation_score(positives: list[float], negatives: list[float]) -> float:
    if not positives:
        return float("-inf")
    return _safe_mean(positives) - (_safe_mean(negatives) * 0.6)


def _choose_primary_prompts(observations: list[dict]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for candidate in CLAP_PRIMARY_CANDIDATES:
        best_prompt = candidate["prompt"]
        best_score = float("-inf")
        for prompt in candidate["prompt_variants"]:
            positives = []
            negatives = []
            for observation in observations:
                values = observation["primary_window_scores"].get(candidate["key"], [])
                score = _aggregate_series(values, PRIMARY_AGGREGATION)
                if observation.get("expected_primary_key") == candidate["key"]:
                    positives.append(score)
                else:
                    negatives.append(score)
            metric = _separation_score(positives, negatives)
            if metric > best_score:
                best_score = metric
                best_prompt = prompt
        selected[candidate["key"]] = best_prompt
    return selected


def _choose_supporting_prompts(observations: list[dict]) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for candidate in CLAP_SUPPORTING_CANDIDATES:
        ranked_variants: list[tuple[float, str]] = []
        prompt_variants = candidate["prompt_variants"]
        for prompt in prompt_variants:
            positives = []
            negatives = []
            for observation in observations:
                prompt_scores = observation["supporting_prompt_scores"].get(candidate["key"], {})
                score = _aggregate_series(prompt_scores.get(prompt, []), "top2_mean")
                if candidate["key"] in observation.get("expected_supporting_keys", []):
                    positives.append(score)
                else:
                    negatives.append(score)
            ranked_variants.append((_separation_score(positives, negatives), prompt))
        ranked_variants.sort(reverse=True)
        chosen = [prompt for metric, prompt in ranked_variants if metric > 0][:2]
        selected[candidate["key"]] = chosen or [candidate["prompt"]]
    return selected


def _fixture_support_score(observation: dict, key: str, prompts: list[str], aggregation: str) -> float:
    per_prompt = observation["supporting_prompt_scores"].get(key, {})
    if not prompts:
        return 0.0
    window_count = max((len(per_prompt.get(prompt, [])) for prompt in prompts), default=0)
    per_window_max: list[float] = []
    for idx in range(window_count):
        per_window_max.append(
            max(
                (
                    per_prompt.get(prompt, [0.0] * window_count)[idx]
                    if idx < len(per_prompt.get(prompt, []))
                    else 0.0
                )
                for prompt in prompts
            )
        )
    return _aggregate_series(per_window_max, aggregation)


def _fixture_primary_score(observation: dict) -> float:
    return max(
        (_aggregate_series(values, PRIMARY_AGGREGATION) for values in observation["primary_window_scores"].values()),
        default=0.0,
    )


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _choose_supporting_rules(
    observations: list[dict],
    supporting_prompts: dict[str, list[str]],
) -> dict[str, dict]:
    rules: dict[str, dict] = {}
    for candidate in CLAP_SUPPORTING_CANDIDATES:
        key = candidate["key"]
        prompts = supporting_prompts.get(key, [candidate["prompt"]])
        best_choice = {
            "aggregation": "top2_mean",
            "absolute_min_score": 0.07,
            "relative_ratio": 0.12,
            "_metric": float("-inf"),
        }
        for aggregation in SUPPORTING_AGGREGATIONS:
            fixture_scores = [
                _fixture_support_score(observation, key, prompts, aggregation)
                for observation in observations
            ]
            primary_scores = [_fixture_primary_score(observation) for observation in observations]
            for absolute_min in ABSOLUTE_THRESHOLD_GRID:
                for relative_ratio in RELATIVE_RATIO_GRID:
                    tp = fp = fn = 0
                    for observation, score, primary_score in zip(
                        observations, fixture_scores, primary_scores, strict=False
                    ):
                        predicted = score >= max(absolute_min, primary_score * relative_ratio)
                        expected = key in observation.get("expected_supporting_keys", [])
                        if predicted and expected:
                            tp += 1
                        elif predicted and not expected:
                            fp += 1
                        elif not predicted and expected:
                            fn += 1

                    precision = tp / (tp + fp) if (tp + fp) else 0.0
                    recall = tp / (tp + fn) if (tp + fn) else 0.0
                    metric = _f1(precision, recall)
                    tie_break = precision + (recall * 0.1)
                    current_tie_break = (
                        best_choice.get("_precision", 0.0)
                        + (best_choice.get("_recall", 0.0) * 0.1)
                    )
                    if metric > best_choice["_metric"] or (
                        metric == best_choice["_metric"] and tie_break > current_tie_break
                    ):
                        best_choice = {
                            "aggregation": aggregation,
                            "absolute_min_score": absolute_min,
                            "relative_ratio": relative_ratio,
                            "_metric": metric,
                            "_precision": precision,
                            "_recall": recall,
                        }

        rules[key] = {
            "aggregation": best_choice["aggregation"],
            "absolute_min_score": best_choice["absolute_min_score"],
            "relative_ratio": best_choice["relative_ratio"],
        }

    return rules


def calibrate_clap_profile_from_observations(
    observations: list[dict],
    *,
    base_profile: dict | None = None,
) -> dict:
    if not observations:
        raise ValueError("At least one fixture observation is required for calibration.")

    profile = dict(base_profile or build_default_clap_runtime_profile(get_settings().audio_event_min_score))
    primary_prompts = _choose_primary_prompts(observations)
    supporting_prompts = _choose_supporting_prompts(observations)
    supporting_rules = _choose_supporting_rules(observations, supporting_prompts)

    profile["primary_prompts"] = primary_prompts
    profile["supporting_prompts"] = supporting_prompts
    profile["supporting_rules"] = supporting_rules
    profile["metrics"] = {
        "fixtures_evaluated": len(observations),
        "labels": [observation["label"] for observation in observations],
    }
    return profile


async def calibrate_audio_events_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    scratch_dir: Path | None = None,
) -> dict:
    fixtures = load_audio_calibration_manifest(manifest_path)
    observations = await collect_clap_fixture_observations(fixtures, scratch_dir=scratch_dir)
    settings = get_settings()
    profile = calibrate_clap_profile_from_observations(
        observations,
        base_profile=build_default_clap_runtime_profile(settings.audio_event_min_score),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate CLAP audio-event prompts and thresholds.")
    parser.add_argument("--manifest", required=True, help="Path to the calibration manifest JSON.")
    parser.add_argument("--output", required=True, help="Where to write the calibration profile JSON.")
    parser.add_argument(
        "--scratch-dir",
        help="Optional directory for temporary extracted audio artifacts during calibration.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    profile = asyncio.run(
        calibrate_audio_events_manifest(
            Path(args.manifest).resolve(),
            Path(args.output).resolve(),
            scratch_dir=Path(args.scratch_dir).resolve() if args.scratch_dir else None,
        )
    )
    print(json.dumps(profile, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
