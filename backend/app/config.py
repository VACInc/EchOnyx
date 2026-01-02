"""Application configuration with hardware detection."""

import os
import subprocess
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HardwareProfile(str, Enum):
    STRIX_HALO = "strix_halo"
    RTX_5090 = "rtx_5090"
    MULTI_GPU = "multi_gpu"
    CPU_ONLY = "cpu_only"


class GPUBackend(str, Enum):
    CUDA = "cuda"
    VULKAN = "vulkan"
    ROCM = "rocm"
    CPU = "cpu"


class ModelLoadingStrategy(str, Enum):
    SEQUENTIAL = "sequential"  # Load/unload models as needed (memory constrained)
    PARALLEL = "parallel"  # Keep all models loaded (high VRAM)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "Video Summarizer"
    debug: bool = False
    api_prefix: str = "/api"

    # Hardware (auto-detected if not set)
    hardware_profile: HardwareProfile | None = None
    gpu_backend: GPUBackend | None = None
    cuda_visible_devices: str = ""
    vulkan_device: int = 0
    gpu_memory_fraction: float = 0.75

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/video_summarizer"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # ChromaDB
    chroma_persist_dir: Path = Field(default=Path("/data/chroma"))

    # Models
    whisper_model: str = "large-v3"
    transcription_fallback_model: str = "large-v3"
    transcription_fallback_enabled: bool = True
    granite_force_cpu: bool = False
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    vision_model: str = "Qwen3-Omni-30B-A3B-Q4_K_M.gguf"
    vision_mmproj: str = ""
    vision_chat_format: str = ""
    vision_endpoint_url: str = ""
    vision_endpoint_model: str = ""
    vision_endpoint_api_key: str = ""
    vision_endpoint_timeout_s: float = 600.0
    vision_debug: bool = False
    summarization_model: str = "Qwen3-30B-A3B-Q4_K_M.gguf"
    summarization_endpoint_url: str = ""
    summarization_endpoint_model: str = ""
    summarization_endpoint_api_key: str = ""
    summarization_endpoint_timeout_s: float = 600.0
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    audio_event_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    audio_event_sample_seconds: float = 8.0
    audio_event_num_samples: int = 6
    audio_event_min_score: float = 0.15
    audio_event_debug: bool = False
    job_stale_minutes: int = 30

    # Model loading
    model_loading: ModelLoadingStrategy = ModelLoadingStrategy.SEQUENTIAL
    model_cache_dir: Path = Field(default=Path("/data/models"))

    # Processing
    max_video_length_hours: int = 4
    keyframe_extraction_interval: int = 5  # seconds
    frame_persistence_seconds: float = 3.0
    frame_change_threshold: float = 12.0
    frame_stability_threshold: float = 6.0
    frame_dedupe_threshold: float = 4.0
    frame_resize_width: int = 320
    max_keyframes: int = 0
    min_speech_duration: float = 0.5
    batch_concurrent_jobs: int = 1

    summary_chunk_minutes: float = 6.0
    summary_chunk_overlap_minutes: float = 0.6
    asr_chunk_length_s: float = 30.0
    asr_chunk_overlap_s: float = 2.0
    asr_dedupe_tolerance_s: float = 0.05
    visual_context_max_frames: int = 60
    visual_context_max_chars: int = 8000

    # Storage
    upload_dir: Path = Field(default=Path("/data/uploads"))
    max_upload_size_gb: int = 10

    # Hugging Face (for pyannote)
    hf_token: str = ""


def detect_gpu_info() -> dict:
    """Detect GPU information from the system."""
    gpu_info = {
        "nvidia_gpus": [],
        "amd_gpus": [],
        "total_vram_gb": 0,
    }

    # Check for NVIDIA GPUs
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(", ")
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        vram_mb = int(parts[1].strip())
                        gpu_info["nvidia_gpus"].append({
                            "name": name,
                            "vram_gb": vram_mb / 1024,
                        })
                        gpu_info["total_vram_gb"] += vram_mb / 1024
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for AMD GPUs (ROCm)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:  # Skip header
                if line:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        # ROCm reports in bytes
                        vram_bytes = int(parts[1].strip())
                        gpu_info["amd_gpus"].append({
                            "name": "AMD GPU",
                            "vram_gb": vram_bytes / (1024**3),
                        })
                        gpu_info["total_vram_gb"] += vram_bytes / (1024**3)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for AMD APU with unified memory (Strix Halo)
    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    total_ram_kb = int(line.split()[1])
                    total_ram_gb = total_ram_kb / (1024**2)
                    # If we have lots of RAM and AMD GPU, might be Strix Halo
                    if total_ram_gb >= 96 and not gpu_info["nvidia_gpus"]:
                        gpu_info["unified_memory_gb"] = total_ram_gb
    except FileNotFoundError:
        pass

    return gpu_info


def auto_detect_hardware_profile(gpu_info: dict) -> tuple[HardwareProfile, GPUBackend]:
    """Auto-detect the best hardware profile based on available GPUs."""
    nvidia_gpus = gpu_info.get("nvidia_gpus", [])
    amd_gpus = gpu_info.get("amd_gpus", [])
    unified_memory = gpu_info.get("unified_memory_gb", 0)
    total_vram = gpu_info.get("total_vram_gb", 0)

    # Check for multi-GPU NVIDIA setup
    if len(nvidia_gpus) > 1 or (len(nvidia_gpus) == 1 and total_vram >= 40):
        if len(nvidia_gpus) > 1:
            return HardwareProfile.MULTI_GPU, GPUBackend.CUDA
        # Single high-VRAM GPU (like RTX 5090)
        gpu_name = nvidia_gpus[0]["name"].lower()
        if "5090" in gpu_name or total_vram >= 30:
            return HardwareProfile.RTX_5090, GPUBackend.CUDA
        return HardwareProfile.MULTI_GPU, GPUBackend.CUDA

    # Check for AMD with unified memory (Strix Halo)
    if unified_memory >= 96:
        return HardwareProfile.STRIX_HALO, GPUBackend.VULKAN

    # Check for single AMD GPU
    if amd_gpus:
        return HardwareProfile.MULTI_GPU, GPUBackend.ROCM

    # Single NVIDIA GPU
    if nvidia_gpus:
        return HardwareProfile.RTX_5090, GPUBackend.CUDA

    # Fallback to CPU
    return HardwareProfile.CPU_ONLY, GPUBackend.CPU


def get_model_loading_strategy(profile: HardwareProfile) -> ModelLoadingStrategy:
    """Determine model loading strategy based on hardware profile."""
    if profile == HardwareProfile.STRIX_HALO:
        return ModelLoadingStrategy.SEQUENTIAL
    elif profile == HardwareProfile.MULTI_GPU:
        return ModelLoadingStrategy.PARALLEL
    elif profile == HardwareProfile.RTX_5090:
        # RTX 5090 has enough VRAM to keep most models loaded
        return ModelLoadingStrategy.PARALLEL
    return ModelLoadingStrategy.SEQUENTIAL


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance with auto-detected hardware."""
    settings = Settings()

    # Auto-detect hardware if not explicitly set
    if settings.hardware_profile is None or settings.gpu_backend is None:
        gpu_info = detect_gpu_info()
        detected_profile, detected_backend = auto_detect_hardware_profile(gpu_info)

        if settings.hardware_profile is None:
            settings.hardware_profile = detected_profile
        if settings.gpu_backend is None:
            settings.gpu_backend = detected_backend

    # Set model loading strategy based on hardware
    if settings.hardware_profile == HardwareProfile.STRIX_HALO:
        settings.model_loading = ModelLoadingStrategy.SEQUENTIAL
    elif settings.model_loading == ModelLoadingStrategy.SEQUENTIAL:
        settings.model_loading = get_model_loading_strategy(settings.hardware_profile)

    # Auto-attach mmproj/chat format for Qwen3VL GGUFs when not explicitly set
    if not settings.vision_mmproj:
        vision_name = settings.vision_model.lower()
        if "qwen3vl-32b-instruct" in vision_name:
            settings.vision_mmproj = "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"
    if not settings.vision_chat_format:
        vision_name = settings.vision_model.lower()
        if "qwen3vl-32b-instruct" in vision_name:
            settings.vision_chat_format = "qwen3-vl"

    # Adjust batch concurrent jobs based on hardware
    if settings.hardware_profile == HardwareProfile.MULTI_GPU:
        settings.batch_concurrent_jobs = max(2, settings.batch_concurrent_jobs)

    return settings


def get_hardware_info() -> dict:
    """Get detailed hardware information for the settings API."""
    gpu_info = detect_gpu_info()
    settings = get_settings()

    whisper_name = settings.whisper_model.lower()
    if "granite-speech" in whisper_name:
        if settings.gpu_backend == GPUBackend.CUDA and not settings.granite_force_cpu:
            whisper_backend = "cuda"
        else:
            whisper_backend = "cpu"
    else:
        if settings.gpu_backend == GPUBackend.CUDA:
            whisper_backend = "cuda"
        elif settings.gpu_backend == GPUBackend.ROCM:
            whisper_backend = "rocm"
        elif settings.gpu_backend == GPUBackend.VULKAN:
            whisper_backend = "vulkan"
        else:
            whisper_backend = "cpu"

    return {
        "detected_gpus": {
            "nvidia": gpu_info.get("nvidia_gpus", []),
            "amd": gpu_info.get("amd_gpus", []),
        },
        "unified_memory_gb": gpu_info.get("unified_memory_gb"),
        "total_vram_gb": gpu_info.get("total_vram_gb", 0),
        "active_profile": settings.hardware_profile.value,
        "active_backend": settings.gpu_backend.value,
        "whisper_backend": whisper_backend,
        "model_loading_strategy": settings.model_loading.value,
    }
