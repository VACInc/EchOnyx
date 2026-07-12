"""Settings and configuration endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import (
    DuplicateHandlingPolicy,
    GPUBackend,
    HardwareProfile,
    ModelLoadingStrategy,
    ROCmLLMRuntime,
    Settings,
    detect_gpu_info,
    get_asr_family,
    get_hardware_info,
    get_settings,
)
from app.core.model_downloader import (
    companion_model_names,
    download_model_async,
    get_all_download_progress,
    get_download_progress,
    get_model_expected_size_gb,
    is_model_cached,
    pyannote_token_guidance,
    reserve_download_progress,
    resolve_local_model_path,
)
from app.core.model_manager import reset_model_manager
from app.env_utils import resolve_env_file_path, stringify_env_value, write_env_updates
from app.runtime.planner import build_runtime_plan
from app.security import validate_endpoint_url, validate_model_name

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024 ** 3
DOWNLOAD_HEADROOM_FRACTION = 0.10
DOWNLOAD_MIN_HEADROOM_GB = 1.0

MODEL_OPTIONS: dict[str, list[dict[str, Any]]] = {
    "asr": [
        {"name": "nvidia/canary-qwen-2.5b", "size_gb": 6.0, "recommended": True},
        {"name": "ibm-granite/granite-speech-3.3-8b", "size_gb": 16.0, "recommended": True},
        {"name": "large-v3", "size_gb": 6.0, "recommended": False},
        {"name": "large-v3-turbo", "size_gb": 3.0, "recommended": False},
        {"name": "medium", "size_gb": 1.5, "recommended": False},
        {"name": "small", "size_gb": 0.5, "recommended": False},
    ],
    "diarization": [
        {"name": "pyannote/speaker-diarization-community-1", "size_gb": 2.0, "recommended": True},
        {"name": "pyannote/speaker-diarization-3.1", "size_gb": 2.0, "recommended": False},
    ],
    "vision": [
        {"name": "Qwen3VL-32B-Instruct-Q4_K_M.gguf", "size_gb": 24.0, "recommended": True},
        {"name": "Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf", "size_gb": 4.0, "recommended": False},
    ],
    "summarization": [
        {"name": "Qwen3-30B-A3B-Q4_K_M.gguf", "size_gb": 24.0, "recommended": True},
        {"name": "Qwen2.5-3B-Instruct.Q4_K_M.gguf", "size_gb": 3.0, "recommended": False},
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

MODEL_COMPONENTS = tuple(MODEL_OPTIONS.keys())

router = APIRouter()
_resolve_env_file_path = resolve_env_file_path
_write_env_updates = write_env_updates
_stringify_env_value = stringify_env_value


def _enum_value(value) -> str:
    return getattr(value, "value", value)
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
    worker_execution_mode: str
    worker_model_loading: str
    endpoint_model_loading: str
    keep_resident_models: list[str]
    preferred_worker_devices: list[str]
    preferred_endpoint_devices: list[str]
    preferred_model_devices: dict[str, list[str]]
    can_keep_all_worker_models_loaded: bool
    can_keep_endpoint_models_loaded: bool
    requires_endpoint_idle_teardown: bool
    endpoint_idle_timeout_recommendation_s: int
    shutdown_endpoint_after_request: bool
    estimated_memory_by_model_gb: dict[str, float]
    notes: list[str]


class ProcessingConfig(BaseModel):
    """Processing configuration."""

    max_video_length_hours: int
    max_upload_size_gb: int
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


class ActionItemsConfig(BaseModel):
    """Action item feature configuration."""

    enabled: bool


class SettingsResponse(BaseModel):
    """Full settings response."""

    hardware_profile: str
    gpu_backend: str
    model_loading: str
    models: ModelConfig
    runtime_planner: RuntimePlannerConfig
    duplicates: DuplicateConfig
    action_items: ActionItemsConfig
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
    action_items_enabled: bool | None = None
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


class ModelVerifyRequest(BaseModel):
    component: str
    model_name: str


class ModelVerifyResponse(BaseModel):
    component: str
    model_name: str
    exists: bool
    source: str
    detail: str


class ModelDownloadRequest(BaseModel):
    component: str
    model_name: str


class ModelDownloadResponse(BaseModel):
    model_name: str
    status: str
    note: str | None = None


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
    "action_items_enabled": "ACTION_ITEMS_ENABLED",
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


def _validate_settings_update(update: SettingsUpdate) -> None:
    model_fields = {
        "asr_model": update.asr_model,
        "diarization_model": update.diarization_model,
        "vision_model": update.vision_model,
        "vision_mmproj": update.vision_mmproj,
        "summarization_model": update.summarization_model,
        "embedding_model": update.embedding_model,
        "audio_event_model": update.audio_event_model,
        "vision_endpoint_model": update.vision_endpoint_model,
        "summarization_endpoint_model": update.summarization_endpoint_model,
    }
    for field_name, value in model_fields.items():
        if value is None:
            continue
        try:
            validate_model_name(value, allow_gguf=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{field_name}: {exc}") from exc

    endpoint_fields = {
        "vision_endpoint_url": update.vision_endpoint_url,
        "summarization_endpoint_url": update.summarization_endpoint_url,
    }
    for field_name, value in endpoint_fields.items():
        if value is None:
            continue
        try:
            validate_endpoint_url(value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{field_name}: {exc}") from exc


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
        worker_execution_mode=runtime_plan["worker_execution_mode"],
        worker_model_loading=runtime_plan["worker_model_loading"],
        endpoint_model_loading=runtime_plan["endpoint_model_loading"],
        keep_resident_models=runtime_plan["keep_resident_models"],
        preferred_worker_devices=runtime_plan["preferred_worker_devices"],
        preferred_endpoint_devices=runtime_plan["preferred_endpoint_devices"],
        preferred_model_devices=runtime_plan["preferred_model_devices"],
        can_keep_all_worker_models_loaded=runtime_plan["can_keep_all_worker_models_loaded"],
        can_keep_endpoint_models_loaded=runtime_plan["can_keep_endpoint_models_loaded"],
        requires_endpoint_idle_teardown=runtime_plan["requires_endpoint_idle_teardown"],
        endpoint_idle_timeout_recommendation_s=runtime_plan["endpoint_idle_timeout_recommendation_s"],
        shutdown_endpoint_after_request=runtime_plan["shutdown_endpoint_after_request"],
        estimated_memory_by_model_gb=runtime_plan["estimated_memory_by_model_gb"],
        notes=runtime_plan["notes"],
    )


def _catalog_expected_size_gb(component: str, model_name: str) -> float | None:
    catalog_match = _find_catalog_model(component, model_name)
    if catalog_match and catalog_match.get("size_gb") is not None:
        return float(catalog_match["size_gb"])
    return None


def _expected_download_size_gb(component: str, model_name: str) -> float | None:
    catalog_size = _catalog_expected_size_gb(component, model_name)
    if catalog_size is not None:
        return catalog_size
    return get_model_expected_size_gb(model_name)


def _free_disk_gb(cache_dir: Path) -> float:
    cache_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(cache_dir)
    return usage.free / BYTES_PER_GB


def _required_disk_gb(expected_size_gb: float) -> float:
    headroom_gb = max(expected_size_gb * DOWNLOAD_HEADROOM_FRACTION, DOWNLOAD_MIN_HEADROOM_GB)
    return expected_size_gb + headroom_gb


def _ensure_download_disk_space(cache_dir: Path, model_name: str, expected_size_gb: float) -> float:
    free_gb = _free_disk_gb(cache_dir)
    required_gb = _required_disk_gb(expected_size_gb)
    if free_gb < required_gb:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Insufficient disk space for {model_name}: required {required_gb:.1f} GB "
                f"including download headroom, free {free_gb:.1f} GB in {cache_dir}."
            ),
        )
    return free_gb


def _is_hf_token_gated_model(component: str, model_name: str) -> bool:
    return component == "diarization" and model_name.startswith("pyannote/")


def _settings_hf_token(settings: Settings) -> str:
    return (
        getattr(settings, "hf_token", "")
        or os.environ.get("HF_TOKEN", "")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    ).strip()


def _task_log_download_result(task) -> None:
    try:
        task.result()
    except Exception as exc:
        logger.warning("Background model download failed: %s", exc)


def _plan_value(runtime_plan, key: str, default: Any = None) -> Any:
    if isinstance(runtime_plan, dict):
        return runtime_plan.get(key, default)
    return getattr(runtime_plan, key, default)


def _recommendation_set(settings: Settings, runtime_plan) -> tuple[str, dict[str, str]]:
    profile = _enum_value(settings.hardware_profile)
    budget_gb = float(_plan_value(runtime_plan, "effective_memory_budget_gb", 0.0) or 0.0)

    base = {
        "asr": "large-v3",
        "diarization": "pyannote/speaker-diarization-community-1",
        "vision": "Qwen3VL-32B-Instruct-Q4_K_M.gguf",
        "summarization": "Qwen3-30B-A3B-Q4_K_M.gguf",
        "embedding": "Qwen/Qwen3-Embedding-8B",
        "audio_event": "laion/clap-htsat-fused",
    }
    small_overrides = {
        "asr": "small",
        "vision": "Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf",
        "summarization": "Qwen2.5-3B-Instruct.Q4_K_M.gguf",
        "embedding": "nomic-ai/nomic-embed-text-v1.5",
    }
    apple_overrides = {
        **small_overrides,
    }

    if profile == HardwareProfile.APPLE_SILICON.value:
        return "apple_silicon", {**base, **apple_overrides}
    if profile == HardwareProfile.CPU_ONLY.value or budget_gb < 16:
        return "small", {**base, **small_overrides}
    return "large", base


def _recommendation_reason(component: str, tier: str, runtime_plan) -> str:
    budget_gb = float(_plan_value(runtime_plan, "effective_memory_budget_gb", 0.0) or 0.0)
    if tier == "apple_silicon":
        if component in {"asr", "vision", "summarization", "embedding"}:
            return "Apple Silicon default keeps first-run memory and download size modest."
        return "Keeps the supported default for this component on Apple Silicon."
    if tier == "small":
        if component in {"asr", "vision", "summarization", "embedding"}:
            return f"Selected for CPU or sub-16 GB accelerator budget ({budget_gb:.1f} GB detected)."
        return "Keeps the supported default while the core model set stays small."
    return f"Selected for the active hardware budget ({budget_gb:.1f} GB detected)."


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
        action_items=ActionItemsConfig(enabled=settings.action_items_enabled),
        processing=ProcessingConfig(
            max_video_length_hours=settings.max_video_length_hours,
            max_upload_size_gb=settings.max_upload_size_gb,
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
    _validate_settings_update(update)
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
    return MODEL_OPTIONS


@router.post("/models/verify", response_model=ModelVerifyResponse)
async def verify_model_candidate(request: ModelVerifyRequest) -> ModelVerifyResponse:
    component = request.component.strip()
    try:
        model_name = validate_model_name(request.model_name, allow_gguf=True)
    except ValueError as exc:
        return ModelVerifyResponse(
            component=component,
            model_name=request.model_name.strip(),
            exists=False,
            source="invalid",
            detail=str(exc),
        )
    if component not in MODEL_COMPONENTS:
        return ModelVerifyResponse(
            component=component,
            model_name=model_name,
            exists=False,
            source="invalid",
            detail="Unknown model component.",
        )
    if not model_name:
        return ModelVerifyResponse(
            component=component,
            model_name=model_name,
            exists=False,
            source="invalid",
            detail="Model name is required.",
        )

    catalog_match = _find_catalog_model(component, model_name)
    if catalog_match:
        return ModelVerifyResponse(
            component=component,
            model_name=catalog_match["name"],
            exists=True,
            source="catalog",
            detail="Model is already available in the built-in catalog.",
        )

    if model_name in _gguf_registry_names():
        return ModelVerifyResponse(
            component=component,
            model_name=model_name,
            exists=True,
            source="registry",
            detail="Model is available in the built-in GGUF registry.",
        )

    if "/" in model_name:
        exists = await _huggingface_model_exists(model_name)
        return ModelVerifyResponse(
            component=component,
            model_name=model_name,
            exists=exists,
            source="huggingface",
            detail="Hugging Face model repo found." if exists else "Hugging Face model repo was not found.",
        )

    return ModelVerifyResponse(
        component=component,
        model_name=model_name,
        exists=False,
        source="unsupported",
        detail="Unknown local alias or GGUF filename. Use a built-in registry name or a Hugging Face repo id.",
    )


@router.post("/models/download", response_model=ModelDownloadResponse)
async def download_model_candidate(request: ModelDownloadRequest) -> JSONResponse:
    """Start an explicit, user-requested model download."""
    verification = await verify_model_candidate(
        ModelVerifyRequest(component=request.component, model_name=request.model_name)
    )
    if not verification.exists:
        raise HTTPException(status_code=400, detail=verification.detail)

    settings = get_settings()
    component = verification.component
    model_name = verification.model_name
    backend = _enum_value(settings.gpu_backend)

    if _is_hf_token_gated_model(component, model_name) and not _settings_hf_token(settings):
        raise HTTPException(status_code=400, detail=pyannote_token_guidance())

    progress = get_download_progress(model_name)
    if progress and progress.get("status") == "downloading":
        raise HTTPException(
            status_code=409,
            detail=f"Model {model_name} is already downloading.",
        )

    if is_model_cached(
        model_name,
        settings.model_cache_dir,
        component=component,
        backend=backend,
    ):
        content = {"model_name": model_name, "status": "cached"}
        # A cached primary can still be missing its companion (e.g. a vision
        # GGUF downloaded before companion handling existed); heal it here.
        companions = _launch_companion_downloads(
            model_name, component, backend, settings
        )
        if companions:
            content["companions"] = companions
        return JSONResponse(status_code=200, content=content)

    expected_size_gb = _expected_download_size_gb(component, model_name)
    note = None
    total_bytes = 0
    if expected_size_gb is None:
        _free_disk_gb(settings.model_cache_dir)
        note = "Expected download size is unknown; disk space could not be preflighted."
    else:
        _ensure_download_disk_space(settings.model_cache_dir, model_name, expected_size_gb)
        total_bytes = int(expected_size_gb * BYTES_PER_GB)

    if not reserve_download_progress(model_name, total_bytes=total_bytes):
        raise HTTPException(
            status_code=409,
            detail=f"Model {model_name} is already downloading.",
        )

    task = asyncio.create_task(
        download_model_async(
            model_name,
            settings.model_cache_dir,
            component=component,
            backend=backend,
        )
    )
    task.add_done_callback(_task_log_download_result)

    content = {"model_name": model_name, "status": "downloading"}
    if note:
        content["note"] = note
    companions = _launch_companion_downloads(model_name, component, backend, settings)
    if companions:
        content["companions"] = companions
    return JSONResponse(status_code=202, content=content)


def _launch_companion_downloads(
    model_name: str, component: str, backend: str, settings
) -> list[dict]:
    """Start registry-declared companion downloads (e.g. a vision mmproj).

    Best-effort: a companion problem is reported in the response instead of
    failing the primary download, but it is never silently skipped, because a
    vision endpoint cannot start with the projector missing.
    """
    results: list[dict] = []
    for companion in companion_model_names(model_name):
        progress = get_download_progress(companion)
        if progress and progress.get("status") == "downloading":
            results.append({"model_name": companion, "status": "downloading"})
            continue
        if is_model_cached(
            companion,
            settings.model_cache_dir,
            component=component,
            backend=backend,
        ):
            results.append({"model_name": companion, "status": "cached"})
            continue
        try:
            expected_size_gb = _expected_download_size_gb(component, companion)
            total_bytes = 0
            if expected_size_gb is not None:
                _ensure_download_disk_space(
                    settings.model_cache_dir, companion, expected_size_gb
                )
                total_bytes = int(expected_size_gb * BYTES_PER_GB)
            if not reserve_download_progress(companion, total_bytes=total_bytes):
                results.append({"model_name": companion, "status": "downloading"})
                continue
            task = asyncio.create_task(
                download_model_async(
                    companion,
                    settings.model_cache_dir,
                    component=component,
                    backend=backend,
                )
            )
            task.add_done_callback(_task_log_download_result)
            results.append({"model_name": companion, "status": "downloading"})
        except HTTPException as exc:
            results.append(
                {"model_name": companion, "status": "error", "detail": str(exc.detail)}
            )
        except Exception as exc:  # pragma: no cover - defensive
            results.append(
                {"model_name": companion, "status": "error", "detail": str(exc)}
            )
    return results


@router.get("/models/recommendations")
async def get_model_recommendations() -> dict:
    """Return hardware-aware model recommendations for the active profile."""
    settings = get_settings()
    gpu_info = detect_gpu_info()
    runtime_plan = build_runtime_plan(settings, gpu_info)
    tier, choices = _recommendation_set(settings, runtime_plan)
    cache_dir = settings.model_cache_dir
    backend = _enum_value(settings.gpu_backend)
    free_disk_gb = _free_disk_gb(cache_dir)

    recommendations: dict[str, dict[str, Any]] = {}
    total_additional_gb = 0.0
    for component in MODEL_COMPONENTS:
        model_name = choices[component]
        expected_size_gb = _expected_download_size_gb(component, model_name)
        cached = is_model_cached(
            model_name,
            cache_dir,
            component=component,
            backend=backend,
        )
        additional_gb = 0.0 if cached or expected_size_gb is None else expected_size_gb
        total_additional_gb += additional_gb
        recommendations[component] = {
            "model_name": model_name,
            "expected_size_gb": expected_size_gb,
            "cached": cached,
            "additional_download_gb": round(additional_gb, 2),
            "reason": _recommendation_reason(component, tier, runtime_plan),
        }

    return {
        "hardware_profile": _enum_value(settings.hardware_profile),
        "effective_memory_budget_gb": _plan_value(runtime_plan, "effective_memory_budget_gb", 0.0),
        "free_disk_gb": round(free_disk_gb, 2),
        "total_additional_download_gb": round(total_additional_gb, 2),
        "recommendations": recommendations,
    }


@router.get("/models/status")
async def get_model_download_status() -> dict:
    """Get status of all model downloads (in progress or completed)."""
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
        "audio_event": settings.audio_event_model,
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

        if model_name.lower().endswith(".gguf"):
            model_path = resolve_local_model_path(model_name, cache_dir)
            if model_path is not None:
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
                size_gb = _expected_download_size_gb(
                    "asr" if model_type == "whisper" else model_type,
                    model_name,
                ) or 0
                models_status[model_type] = {
                    "model_name": model_name,
                    "status": "uncached",
                    "expected_size_gb": size_gb,
                }
        else:
            if model_name in download_progress:
                models_status[model_type] = _normalize_progress_status(download_progress[model_name])
            elif is_model_cached(
                model_name,
                cache_dir,
                component="asr" if model_type == "whisper" else model_type,
                backend=_enum_value(settings.gpu_backend),
            ):
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


def _find_catalog_model(component: str, model_name: str) -> dict[str, Any] | None:
    normalized = model_name.strip().lower()
    for item in MODEL_OPTIONS.get(component, []):
        if str(item["name"]).strip().lower() == normalized:
            return item
    return None


def _gguf_registry_names() -> set[str]:
    from app.core.model_downloader import MODEL_REGISTRY

    return set(MODEL_REGISTRY.keys())


async def _huggingface_model_exists(model_name: str) -> bool:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def check() -> bool:
        api = HfApi()
        try:
            api.model_info(model_name, token=token)
            return True
        except Exception:
            return False

    import asyncio

    return await asyncio.to_thread(check)


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
