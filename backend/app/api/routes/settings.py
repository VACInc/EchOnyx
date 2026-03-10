"""Settings and configuration endpoints."""

import json
import os
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import (
    GPUBackend,
    HardwareProfile,
    ModelLoadingStrategy,
    get_asr_family,
    get_hardware_info,
    get_settings,
)

router = APIRouter()


class GPUInfo(BaseModel):
    """GPU information."""

    name: str
    vram_gb: float


class HardwareInfo(BaseModel):
    """Detected hardware information."""

    nvidia_gpus: list[GPUInfo]
    amd_gpus: list[GPUInfo]
    unified_memory_gb: float | None
    total_vram_gb: float
    active_profile: str
    active_backend: str
    whisper_backend: str
    asr_family: str
    model_loading_strategy: str


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


class SettingsResponse(BaseModel):
    """Full settings response."""

    hardware_profile: str
    gpu_backend: str
    model_loading: str
    models: ModelConfig
    processing: ProcessingConfig


class SettingsUpdate(BaseModel):
    """Settings update request."""

    hardware_profile: HardwareProfile | None = None
    gpu_backend: GPUBackend | None = None
    model_loading: ModelLoadingStrategy | None = None
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
    max_video_length_hours: int | None = None
    keyframe_extraction_interval: int | None = None
    frame_persistence_seconds: float | None = None
    frame_change_threshold: float | None = None
    frame_stability_threshold: float | None = None
    frame_dedupe_threshold: float | None = None
    frame_resize_width: int | None = None
    max_keyframes: int | None = None
    batch_concurrent_jobs: int | None = None
    summary_chunk_minutes: float | None = None
    summary_chunk_overlap_minutes: float | None = None


@router.get("", response_model=SettingsResponse)
async def get_current_settings() -> SettingsResponse:
    """Get current application settings."""
    settings = get_settings()

    return SettingsResponse(
        hardware_profile=settings.hardware_profile.value if settings.hardware_profile else "unknown",
        gpu_backend=settings.gpu_backend.value if settings.gpu_backend else "unknown",
        model_loading=settings.model_loading.value,
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
    # TODO: Implement settings persistence
    # For now, return current settings (updates would need restart)

    # In a production implementation, we would:
    # 1. Validate the new settings
    # 2. Write them to .env file
    # 3. Optionally reload configuration
    # 4. Return the updated settings

    return await get_current_settings()


@router.get("/hardware", response_model=HardwareInfo)
async def get_hardware() -> HardwareInfo:
    """Get detailed hardware information."""
    info = get_hardware_info()

    nvidia_gpus = [
        GPUInfo(name=gpu["name"], vram_gb=gpu["vram_gb"])
        for gpu in info["detected_gpus"]["nvidia"]
    ]

    amd_gpus = [
        GPUInfo(name=gpu["name"], vram_gb=gpu["vram_gb"])
        for gpu in info["detected_gpus"]["amd"]
    ]

    return HardwareInfo(
        nvidia_gpus=nvidia_gpus,
        amd_gpus=amd_gpus,
        unified_memory_gb=info.get("unified_memory_gb"),
        total_vram_gb=info["total_vram_gb"],
        active_profile=info["active_profile"],
        active_backend=info["active_backend"],
        whisper_backend=info["whisper_backend"],
        asr_family=info["asr_family"],
        model_loading_strategy=info["model_loading_strategy"],
    )


@router.get("/models/available")
async def list_available_models() -> dict:
    """List available models for each component."""
    # TODO: Scan model cache directory for available GGUF files
    # For now, return recommended models

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
    from app.core.model_downloader import get_all_download_progress, MODEL_REGISTRY

    settings = get_settings()
    cache_dir = settings.model_cache_dir
    loaded_models = _loaded_models_from_state(cache_dir)

    # Get download progress for active downloads
    download_progress = get_all_download_progress()

    # Check which models are already downloaded
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

        # Check if it's a GGUF model that needs to be downloaded
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
                # Model needs to be downloaded but hasn't started
                size_gb = MODEL_REGISTRY.get(model_name, {}).get("size_gb", 0)
                models_status[model_type] = {
                    "model_name": model_name,
                    "status": "uncached",
                    "expected_size_gb": size_gb,
                }
        else:
            # HuggingFace model (whisper, diarization, embedding)
            # These download automatically on first use; detect cache when possible.
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
            v for v in download_progress.values()
            if v.get("status") == "downloading"
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
            return any(p.is_dir() for p in snapshots.iterdir())
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
