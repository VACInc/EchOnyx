"""Settings and configuration endpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import (
    DuplicateHandlingPolicy,
    GPUBackend,
    HardwareProfile,
    ModelLoadingStrategy,
    ROCmLLMRuntime,
    Settings,
    get_asr_family,
    get_hardware_info,
    get_settings,
)
from app.core.model_manager import reset_model_manager

router = APIRouter()


def _enum_value(value) -> str:
    return getattr(value, "value", value)


def _stringify_env_value(value: Any) -> str:
    enum_value = getattr(value, "value", value)
    if enum_value is None:
        return ""
    if isinstance(enum_value, bool):
        return "true" if enum_value else "false"
    return str(enum_value)


def _resolve_env_file_path() -> Path:
    env_file = Settings.model_config.get("env_file", ".env")
    if isinstance(env_file, (list, tuple)):
        env_file = env_file[0]
    return Path(env_file or ".env")


def _write_env_updates(path: Path, updates: dict[str, Any]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    key_to_index: dict[str, int] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        key_to_index[key] = index

    for key, value in updates.items():
        if value is None:
            if key in key_to_index:
                lines.pop(key_to_index[key])
                key_to_index = {}
                for index, line in enumerate(lines):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in line:
                        continue
                    current_key = line.split("=", 1)[0].strip()
                    key_to_index[current_key] = index
            continue

        updated_line = f"{key}={_stringify_env_value(value)}"
        if key in key_to_index:
            lines[key_to_index[key]] = updated_line
        else:
            lines.append(updated_line)

    payload = "\n".join(lines).rstrip()
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


async def _reload_runtime_state() -> None:
    get_settings.cache_clear()
    await reset_model_manager()


class GPUInfo(BaseModel):
    """GPU information."""

    name: str
    vram_gb: float
    index: int | None = None
    used_vram_gb: float | None = None
    free_vram_gb: float | None = None
    utilization_gpu: float | None = None
    bus_id: str | None = None


class HardwareInfo(BaseModel):
    """Detected hardware information."""

    nvidia_gpus: list[GPUInfo]
    amd_gpus: list[GPUInfo]
    unified_memory_gb: float | None
    total_vram_gb: float
    available_vram_gb: float
    active_profile: str
    active_backend: str
    whisper_backend: str
    asr_family: str
    model_loading_strategy: str
    rocm_llm_runtime: str
    rocm_llm_idle_timeout_s: int
    runtime_planner_enabled: bool
    runtime_memory_ceiling_gb: float | None
    gpu_memory_fraction: float
    runtime_plan: dict[str, Any]


class ModelConfig(BaseModel):
    """Model configuration."""

    asr_family: str
    asr_model: str
    granite_force_cpu: bool
    diarization_model: str
    vision_model: str
    vision_mmproj: str
    vision_chat_format: str
    vision_endpoint_url: str
    vision_endpoint_model: str
    summarization_model: str
    summarization_endpoint_url: str
    summarization_endpoint_model: str
    embedding_model: str
    audio_event_model: str
    rocm_llm_runtime: str
    rocm_llm_idle_timeout_s: int


class RuntimePlannerConfig(BaseModel):
    """Runtime planner state and user-configurable limits."""

    enabled: bool
    gpu_memory_fraction: float
    memory_ceiling_gb: float | None
    accelerator_count: int
    total_accelerator_memory_gb: float
    available_accelerator_memory_gb: float
    effective_memory_budget_gb: float
    placement_mode: str
    worker_model_loading: str
    keep_resident_models: list[str]
    preferred_worker_devices: list[str]
    preferred_endpoint_devices: list[str]
    can_keep_all_worker_models_loaded: bool
    can_keep_endpoint_models_loaded: bool
    requires_endpoint_idle_teardown: bool
    endpoint_idle_timeout_recommendation_s: int
    estimated_memory_by_model_gb: dict[str, float]
    notes: list[str]


class ProcessingConfig(BaseModel):
    """Processing configuration."""

    max_video_length_hours: int
    keyframe_extraction_interval: int
    frame_persistence_seconds: float
    frame_change_threshold: float
    frame_stability_threshold: float
    frame_dedupe_threshold: float
    frame_resize_width: int
    max_keyframes: int
    min_speech_duration: float
    batch_concurrent_jobs: int
    summary_chunk_minutes: float
    summary_chunk_overlap_minutes: float


class DuplicateConfig(BaseModel):
    """Duplicate detection configuration."""

    policy: str
    exact_threshold: float
    probable_threshold: float


class SettingsResponse(BaseModel):
    """Full settings response."""

    hardware_profile: str
    gpu_backend: str
    model_loading: str
    models: ModelConfig
    runtime_planner: RuntimePlannerConfig
    duplicates: DuplicateConfig
    processing: ProcessingConfig


class SettingsUpdate(BaseModel):
    """Settings update request."""

    hardware_profile: HardwareProfile | None = None
    gpu_backend: GPUBackend | None = None
    model_loading: ModelLoadingStrategy | None = None
    runtime_planner_enabled: bool | None = None
    gpu_memory_fraction: float | None = None
    runtime_memory_ceiling_gb: float | None = None
    rocm_llm_runtime: ROCmLLMRuntime | None = None
    rocm_llm_idle_timeout_s: int | None = None
    duplicate_detection_policy: DuplicateHandlingPolicy | None = None
    duplicate_exact_threshold: float | None = None
    duplicate_probable_threshold: float | None = None
    asr_model: str | None = None
    granite_force_cpu: bool | None = None
    diarization_model: str | None = None
    vision_model: str | None = None
    vision_mmproj: str | None = None
    vision_chat_format: str | None = None
    vision_endpoint_url: str | None = None
    vision_endpoint_model: str | None = None
    summarization_model: str | None = None
    summarization_endpoint_url: str | None = None
    summarization_endpoint_model: str | None = None
    embedding_model: str | None = None
    audio_event_model: str | None = None
    max_video_length_hours: int | None = None
    keyframe_extraction_interval: int | None = None
    frame_persistence_seconds: float | None = None
    frame_change_threshold: float | None = None
    frame_stability_threshold: float | None = None
    frame_dedupe_threshold: float | None = None
    frame_resize_width: int | None = None
    max_keyframes: int | None = None
    min_speech_duration: float | None = None
    batch_concurrent_jobs: int | None = None
    summary_chunk_minutes: float | None = None
    summary_chunk_overlap_minutes: float | None = None


ENV_FIELD_MAP: dict[str, str] = {
    "hardware_profile": "HARDWARE_PROFILE",
    "gpu_backend": "GPU_BACKEND",
    "model_loading": "MODEL_LOADING",
    "runtime_planner_enabled": "RUNTIME_PLANNER_ENABLED",
    "gpu_memory_fraction": "GPU_MEMORY_FRACTION",
    "runtime_memory_ceiling_gb": "RUNTIME_MEMORY_CEILING_GB",
    "rocm_llm_runtime": "ROCM_LLM_RUNTIME",
    "rocm_llm_idle_timeout_s": "ROCM_LLM_IDLE_TIMEOUT_S",
    "duplicate_detection_policy": "DUPLICATE_DETECTION_POLICY",
    "duplicate_exact_threshold": "DUPLICATE_EXACT_THRESHOLD",
    "duplicate_probable_threshold": "DUPLICATE_PROBABLE_THRESHOLD",
    "asr_model": "WHISPER_MODEL",
    "granite_force_cpu": "GRANITE_FORCE_CPU",
    "diarization_model": "DIARIZATION_MODEL",
    "vision_model": "VISION_MODEL",
    "vision_mmproj": "VISION_MMPROJ",
    "vision_chat_format": "VISION_CHAT_FORMAT",
    "vision_endpoint_url": "VISION_ENDPOINT_URL",
    "vision_endpoint_model": "VISION_ENDPOINT_MODEL",
    "summarization_model": "SUMMARIZATION_MODEL",
    "summarization_endpoint_url": "SUMMARIZATION_ENDPOINT_URL",
    "summarization_endpoint_model": "SUMMARIZATION_ENDPOINT_MODEL",
    "embedding_model": "EMBEDDING_MODEL",
    "audio_event_model": "AUDIO_EVENT_MODEL",
    "max_video_length_hours": "MAX_VIDEO_LENGTH_HOURS",
    "keyframe_extraction_interval": "KEYFRAME_EXTRACTION_INTERVAL",
    "frame_persistence_seconds": "FRAME_PERSISTENCE_SECONDS",
    "frame_change_threshold": "FRAME_CHANGE_THRESHOLD",
    "frame_stability_threshold": "FRAME_STABILITY_THRESHOLD",
    "frame_dedupe_threshold": "FRAME_DEDUPE_THRESHOLD",
    "frame_resize_width": "FRAME_RESIZE_WIDTH",
    "max_keyframes": "MAX_KEYFRAMES",
    "min_speech_duration": "MIN_SPEECH_DURATION",
    "batch_concurrent_jobs": "BATCH_CONCURRENT_JOBS",
    "summary_chunk_minutes": "SUMMARY_CHUNK_MINUTES",
    "summary_chunk_overlap_minutes": "SUMMARY_CHUNK_OVERLAP_MINUTES",
}


def _collect_env_updates(update: SettingsUpdate) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    payload = update.model_dump(exclude_unset=True)
    for field_name, value in payload.items():
        env_name = ENV_FIELD_MAP.get(field_name)
        if env_name:
            updates[env_name] = value
    return updates


def _runtime_planner_response(settings, runtime_plan: dict) -> RuntimePlannerConfig:
    return RuntimePlannerConfig(
        enabled=settings.runtime_planner_enabled,
        gpu_memory_fraction=settings.gpu_memory_fraction,
        memory_ceiling_gb=settings.runtime_memory_ceiling_gb,
        accelerator_count=runtime_plan["accelerator_count"],
        total_accelerator_memory_gb=runtime_plan["total_accelerator_memory_gb"],
        available_accelerator_memory_gb=runtime_plan["available_accelerator_memory_gb"],
        effective_memory_budget_gb=runtime_plan["effective_memory_budget_gb"],
        placement_mode=runtime_plan["placement_mode"],
        worker_model_loading=runtime_plan["worker_model_loading"],
        keep_resident_models=runtime_plan["keep_resident_models"],
        preferred_worker_devices=runtime_plan["preferred_worker_devices"],
        preferred_endpoint_devices=runtime_plan["preferred_endpoint_devices"],
        can_keep_all_worker_models_loaded=runtime_plan["can_keep_all_worker_models_loaded"],
        can_keep_endpoint_models_loaded=runtime_plan["can_keep_endpoint_models_loaded"],
        requires_endpoint_idle_teardown=runtime_plan["requires_endpoint_idle_teardown"],
        endpoint_idle_timeout_recommendation_s=runtime_plan["endpoint_idle_timeout_recommendation_s"],
        estimated_memory_by_model_gb=runtime_plan["estimated_memory_by_model_gb"],
        notes=runtime_plan["notes"],
    )


@router.get("", response_model=SettingsResponse)
async def get_current_settings() -> SettingsResponse:
    """Get current application settings."""
    settings = get_settings()
    hardware_info = get_hardware_info()
    runtime_plan = hardware_info["runtime_plan"]

    return SettingsResponse(
        hardware_profile=_enum_value(settings.hardware_profile) if settings.hardware_profile else "unknown",
        gpu_backend=_enum_value(settings.gpu_backend) if settings.gpu_backend else "unknown",
        model_loading=_enum_value(settings.model_loading),
        models=ModelConfig(
            asr_family=get_asr_family(settings.whisper_model),
            asr_model=settings.whisper_model,
            granite_force_cpu=settings.granite_force_cpu,
            diarization_model=settings.diarization_model,
            vision_model=settings.vision_model,
            vision_mmproj=settings.vision_mmproj,
            vision_chat_format=settings.vision_chat_format,
            vision_endpoint_url=settings.vision_endpoint_url,
            vision_endpoint_model=settings.vision_endpoint_model,
            summarization_model=settings.summarization_model,
            summarization_endpoint_url=settings.summarization_endpoint_url,
            summarization_endpoint_model=settings.summarization_endpoint_model,
            embedding_model=settings.embedding_model,
            audio_event_model=settings.audio_event_model,
            rocm_llm_runtime=_enum_value(settings.rocm_llm_runtime),
            rocm_llm_idle_timeout_s=settings.rocm_llm_idle_timeout_s,
        ),
        runtime_planner=_runtime_planner_response(settings, runtime_plan),
        duplicates=DuplicateConfig(
            policy=_enum_value(settings.duplicate_detection_policy),
            exact_threshold=settings.duplicate_exact_threshold,
            probable_threshold=settings.duplicate_probable_threshold,
        ),
        processing=ProcessingConfig(
            max_video_length_hours=settings.max_video_length_hours,
            keyframe_extraction_interval=settings.keyframe_extraction_interval,
            frame_persistence_seconds=settings.frame_persistence_seconds,
            frame_change_threshold=settings.frame_change_threshold,
            frame_stability_threshold=settings.frame_stability_threshold,
            frame_dedupe_threshold=settings.frame_dedupe_threshold,
            frame_resize_width=settings.frame_resize_width,
            max_keyframes=settings.max_keyframes,
            min_speech_duration=settings.min_speech_duration,
            batch_concurrent_jobs=settings.batch_concurrent_jobs,
            summary_chunk_minutes=settings.summary_chunk_minutes,
            summary_chunk_overlap_minutes=settings.summary_chunk_overlap_minutes,
        ),
    )


@router.put("", response_model=SettingsResponse)
async def update_settings(update: SettingsUpdate) -> SettingsResponse:
    """
    Update application settings.

    Note: Some settings may require a restart to take effect.
    Settings are persisted to the .env file.
    """
    env_updates = _collect_env_updates(update)
    if env_updates:
        _write_env_updates(_resolve_env_file_path(), env_updates)
        for key, value in env_updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = _stringify_env_value(value)
        await _reload_runtime_state()
    return await get_current_settings()


@router.get("/hardware", response_model=HardwareInfo)
async def get_hardware() -> HardwareInfo:
    """Get detailed hardware information."""
    info = get_hardware_info()

    nvidia_gpus = [
        GPUInfo(
            name=gpu["name"],
            vram_gb=gpu["vram_gb"],
            index=gpu.get("index"),
            used_vram_gb=gpu.get("used_vram_gb"),
            free_vram_gb=gpu.get("free_vram_gb"),
            utilization_gpu=gpu.get("utilization_gpu"),
            bus_id=gpu.get("bus_id"),
        )
        for gpu in info["detected_gpus"]["nvidia"]
    ]

    amd_gpus = [
        GPUInfo(
            name=gpu["name"],
            vram_gb=gpu["vram_gb"],
            index=gpu.get("index"),
            used_vram_gb=gpu.get("used_vram_gb"),
            free_vram_gb=gpu.get("free_vram_gb"),
            utilization_gpu=gpu.get("utilization_gpu"),
            bus_id=gpu.get("bus_id"),
        )
        for gpu in info["detected_gpus"]["amd"]
    ]

    return HardwareInfo(
        nvidia_gpus=nvidia_gpus,
        amd_gpus=amd_gpus,
        unified_memory_gb=info.get("unified_memory_gb"),
        total_vram_gb=info["total_vram_gb"],
        available_vram_gb=info.get("available_vram_gb", info["total_vram_gb"]),
        active_profile=info["active_profile"],
        active_backend=info["active_backend"],
        whisper_backend=info["whisper_backend"],
        asr_family=info["asr_family"],
        model_loading_strategy=info["model_loading_strategy"],
        rocm_llm_runtime=info["rocm_llm_runtime"],
        rocm_llm_idle_timeout_s=info["rocm_llm_idle_timeout_s"],
        runtime_planner_enabled=info["runtime_planner_enabled"],
        runtime_memory_ceiling_gb=info["runtime_memory_ceiling_gb"],
        gpu_memory_fraction=info["gpu_memory_fraction"],
        runtime_plan=info["runtime_plan"],
    )


@router.get("/models/available")
async def list_available_models() -> dict:
    """List available models for each component."""
    return {
        "asr": [
            {"name": "nvidia/canary-qwen-2.5b", "size_gb": 6.0, "recommended": True},
            {"name": "ibm-granite/granite-speech-3.3-8b", "size_gb": 16.0, "recommended": True},
            {"name": "large-v3", "size_gb": 6.0, "recommended": False},
            {"name": "large-v3-turbo", "size_gb": 3.0, "recommended": False},
            {"name": "medium", "size_gb": 1.5, "recommended": False},
        ],
        "diarization": [
            {"name": "pyannote/speaker-diarization-community-1", "size_gb": 2.0, "recommended": True},
            {"name": "pyannote/speaker-diarization-3.1", "size_gb": 2.0, "recommended": False},
        ],
        "vision": [
            {"name": "llama.cpp endpoint (Qwen3-VL)", "size_gb": 0.0, "recommended": False},
            {"name": "qwen3-omni-30b-a3b-q4_k_m.gguf", "size_gb": 15.0, "recommended": True},
            {"name": "qwen3-omni-30b-a3b-q5_k_m.gguf", "size_gb": 20.0, "recommended": False},
        ],
        "summarization": [
            {"name": "llama.cpp endpoint (gptoss-120b)", "size_gb": 0.0, "recommended": True},
            {"name": "qwen3-30b-a3b-q4_k_m.gguf", "size_gb": 15.0, "recommended": False},
            {"name": "qwen3-30b-a3b-q5_k_m.gguf", "size_gb": 20.0, "recommended": False},
        ],
        "embedding": [
            {"name": "Qwen/Qwen3-Embedding-8B", "size_gb": 16.0, "recommended": True},
            {"name": "nomic-ai/nomic-embed-text-v1.5", "size_gb": 0.6, "recommended": False},
        ],
        "audio_event": [
            {"name": "laion/clap-htsat-fused", "size_gb": 2.5, "recommended": True},
            {"name": "MIT/ast-finetuned-audioset-10-10-0.4593", "size_gb": 0.4, "recommended": False},
        ],
    }


@router.get("/models/status")
async def get_model_download_status() -> dict:
    """Get status of all model downloads (in progress or completed)."""
    from app.core.model_downloader import MODEL_REGISTRY, get_all_download_progress

    settings = get_settings()
    cache_dir = settings.model_cache_dir
    loaded_models = _loaded_models_from_state(cache_dir)

    download_progress = get_all_download_progress()

    models_status = {}
    required_models = {
        "whisper": settings.whisper_model,
        "diarization": settings.diarization_model,
        "vision": settings.vision_model,
        "summarization": settings.summarization_model,
        "embedding": settings.embedding_model,
    }
    for model_type, model_name in required_models.items():
        if model_type == "vision" and settings.vision_endpoint_url.strip():
            status = await _endpoint_status(settings.vision_endpoint_url, settings.vision_endpoint_api_key)
            models_status[model_type] = {
                "model_name": settings.vision_endpoint_model or model_name or "vision-endpoint",
                "status": status,
            }
            continue
        if model_type == "summarization" and settings.summarization_endpoint_url.strip():
            status = await _endpoint_status(
                settings.summarization_endpoint_url,
                settings.summarization_endpoint_api_key,
            )
            models_status[model_type] = {
                "model_name": settings.summarization_endpoint_model or model_name or "summary-endpoint",
                "status": status,
            }
            continue

        if model_name.endswith(".gguf"):
            model_path = cache_dir / model_name
            if model_path.exists():
                file_size = model_path.stat().st_size
                models_status[model_type] = {
                    "model_name": model_name,
                    "status": "cached",
                    "file_size_gb": round(file_size / 1e9, 2),
                    "path": str(model_path),
                }
            elif model_name in download_progress:
                models_status[model_type] = _normalize_progress_status(download_progress[model_name])
            else:
                size_gb = MODEL_REGISTRY.get(model_name, {}).get("size_gb", 0)
                models_status[model_type] = {
                    "model_name": model_name,
                    "status": "uncached",
                    "expected_size_gb": size_gb,
                }
        else:
            if model_name in download_progress:
                models_status[model_type] = _normalize_progress_status(download_progress[model_name])
            elif _hf_model_cached(model_name):
                models_status[model_type] = {
                    "model_name": model_name,
                    "status": "cached",
                }
            else:
                models_status[model_type] = {
                    "model_name": model_name,
                    "status": "uncached",
                }
        if model_type in models_status and model_type in loaded_models:
            current_status = models_status[model_type].get("status")
            if current_status not in {"downloading", "failed", "online", "offline"}:
                models_status[model_type]["status"] = "loaded"

    return {
        "models": models_status,
        "active_downloads": [
            value for value in download_progress.values()
            if value.get("status") == "downloading"
        ],
    }


async def _endpoint_status(url: str, api_key: str | None) -> str:
    url = url.strip()
    if not url:
        return "offline"
    try:
        import httpx

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(url, headers=headers)
        if response.status_code < 500:
            return "online"
    except Exception:
        return "offline"
    return "offline"


def _hf_cache_dir() -> Path:
    cache = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
    )
    if cache:
        return Path(cache)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_model_cached(model_name: str) -> bool:
    if "/" not in model_name:
        return False
    cache_dir = _hf_cache_dir()
    repo_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if snapshots.exists():
        try:
            return any(path.is_dir() for path in snapshots.iterdir())
        except OSError:
            return False
    return False


def _loaded_models_from_state(cache_dir: Path) -> set[str]:
    path = cache_dir / ".loaded_models.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        models = data.get("models") or []
        return {str(model) for model in models}
    except Exception:
        return set()


def _normalize_progress_status(progress: dict) -> dict:
    normalized = dict(progress)
    if normalized.get("status") == "completed":
        normalized["status"] = "cached"
    if normalized.get("status") == "pending":
        normalized["status"] = "uncached"
    return normalized
