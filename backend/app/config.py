"""Application configuration with hardware detection."""

import os
import platform
import re
import subprocess
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.runtime.planner import build_runtime_plan

NVIDIA_SMI_TIMEOUT_S = 30


class HardwareProfile(str, Enum):
    STRIX_HALO = "strix_halo"
    APPLE_SILICON = "apple_silicon"
    RTX_5090 = "rtx_5090"
    MULTI_GPU = "multi_gpu"
    CPU_ONLY = "cpu_only"


class GPUBackend(str, Enum):
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    ROCM = "rocm"
    CPU = "cpu"


class ROCmLLMRuntime(str, Enum):
    LLAMA_SERVER = "llama_server"
    VLLM = "vllm"


class DuplicateHandlingPolicy(str, Enum):
    OFF = "off"
    WARN = "warn"
    COLLAPSE_EXACT = "collapse_exact"
    COLLAPSE_PROBABLE = "collapse_probable"


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
    app_name: str = "EchOnyx"
    debug: bool = False
    api_prefix: str = "/api"
    cors_allowed_origins: str = ""
    cors_allow_origin_regex: str = ""
    auth_required: bool = True
    auth_password_hash: str = ""
    auth_session_cookie_name: str = "echonyx_session"
    auth_csrf_cookie_name: str = "echonyx_csrf"
    auth_session_ttl_hours: int = 168
    login_rate_limit_attempts: int = 10
    login_rate_limit_window_seconds: int = 600
    write_rate_limit_requests: int = 120
    write_rate_limit_window_seconds: int = 60
    upload_rate_limit_requests: int = 12
    upload_rate_limit_window_seconds: int = 3600
    max_json_request_bytes: int = 1_048_576
    audit_log_retention_days: int = 90

    # Hardware (auto-detected if not set)
    hardware_profile: HardwareProfile | None = None
    gpu_backend: GPUBackend | None = None
    cuda_visible_devices: str = ""
    vulkan_device: int = 0
    gpu_memory_fraction: float = 0.75
    runtime_planner_enabled: bool = True
    runtime_memory_ceiling_gb: float | None = None
    rocm_llm_runtime: ROCmLLMRuntime = ROCmLLMRuntime.LLAMA_SERVER
    rocm_llm_idle_timeout_s: int = 120
    duplicate_detection_policy: DuplicateHandlingPolicy = DuplicateHandlingPolicy.COLLAPSE_EXACT
    duplicate_exact_threshold: float = 0.95
    duplicate_probable_threshold: float = 0.85
    action_items_enabled: bool = True

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/video_summarizer"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # ChromaDB
    chroma_persist_dir: Path = Field(default=Path("/data/chroma"))

    # Models
    whisper_model: str = "large-v3"
    transcription_fallback_model: str = "large-v3"
    transcription_fallback_enabled: bool = False
    granite_force_cpu: bool = False
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    vision_model: str = "Qwen3VL-32B-Instruct-Q4_K_M.gguf"
    vision_mmproj: str = ""
    vision_chat_format: str = ""
    vision_gpu_layers: int | None = None
    vision_endpoint_url: str = ""
    vision_endpoint_model: str = ""
    vision_endpoint_api_key: str = ""
    vision_endpoint_timeout_s: float = 600.0
    vision_debug: bool = False
    summarization_model: str = "Qwen3-30B-A3B-Q4_K_M.gguf"
    summarization_gpu_layers: int | None = None
    summarization_endpoint_url: str = ""
    summarization_endpoint_model: str = ""
    summarization_endpoint_api_key: str = ""
    summarization_endpoint_timeout_s: float = 600.0
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    audio_event_model: str = "laion/clap-htsat-fused"
    audio_event_calibration_path: Path = Field(default=Path("/data/models/audio_event_calibration.json"))
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

    @field_validator("hardware_profile", "gpu_backend", mode="before")
    @classmethod
    def _blank_enum_env_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value


def get_asr_family(model_name: str) -> str:
    """Classify the selected ASR model family for reporting and UI."""
    lowered = model_name.lower()
    if "canary" in lowered:
        return "canary"
    if "granite-speech" in lowered:
        return "granite"
    return "whisper"


def validate_hardware_requirements(settings: Settings) -> None:
    """Fail closed when a runtime violates a required hardware contract."""
    if settings.hardware_profile != HardwareProfile.STRIX_HALO:
        return

    if settings.gpu_backend != GPUBackend.ROCM:
        raise RuntimeError(
            "Strix Halo requires GPU_BACKEND=rocm. CPU and Vulkan fallbacks are disabled."
        )

    if settings.granite_force_cpu:
        raise RuntimeError(
            "Strix Halo requires ROCm acceleration for transcription; GRANITE_FORCE_CPU must stay false."
        )

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("Strix Halo requires a ROCm-enabled PyTorch runtime.") from exc

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    if hip_version is None:
        raise RuntimeError(
            "Strix Halo requires a ROCm-enabled PyTorch build; torch.version.hip is unavailable."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Strix Halo requires a visible ROCm device; torch.cuda.is_available() is false."
        )


def _detect_macos_total_memory_gb() -> float:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0.0
    if result.returncode != 0:
        return 0.0
    try:
        return int(result.stdout.strip()) / (1024**3)
    except ValueError:
        return 0.0


def _detect_macos_available_memory_gb() -> float:
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0.0
    if result.returncode != 0:
        return 0.0

    match = re.search(r"page size of (\d+) bytes", result.stdout)
    page_size = int(match.group(1)) if match else 4096
    page_counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = re.sub(r"[^0-9]", "", value)
        if digits:
            page_counts[key.strip().lower()] = int(digits)

    free_pages = (
        page_counts.get("pages free", 0)
        + page_counts.get("pages inactive", 0)
        + page_counts.get("pages speculative", 0)
    )
    return (free_pages * page_size) / (1024**3)


def _detect_apple_chip_name() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "Apple Silicon"
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return "Apple Silicon"


def _env_present(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and bool(str(value).strip())


def _project_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _apply_apple_silicon_defaults(settings: Settings, gpu_info: dict) -> None:
    if settings.hardware_profile != HardwareProfile.APPLE_SILICON:
        return

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    project_root = _project_root_dir()

    if not _env_present("WHISPER_MODEL"):
        settings.whisper_model = "small"
    if not _env_present("EMBEDDING_MODEL"):
        settings.embedding_model = "nomic-ai/nomic-embed-text-v1.5"
    if not _env_present("VISION_MODEL"):
        settings.vision_model = "Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf"
    if not _env_present("VISION_MMPROJ"):
        settings.vision_mmproj = "Qwen2.5-VL-3B-Instruct.mmproj-fp16.gguf"
    if not _env_present("VISION_CHAT_FORMAT"):
        settings.vision_chat_format = "qwen2.5-vl"
    if not _env_present("SUMMARIZATION_MODEL"):
        settings.summarization_model = "Qwen2.5-3B-Instruct.Q4_K_M.gguf"
    if not _env_present("UPLOAD_DIR"):
        settings.upload_dir = project_root / "data" / "uploads"
    if not _env_present("MODEL_CACHE_DIR"):
        settings.model_cache_dir = project_root / "data" / "models"
    if not _env_present("CHROMA_PERSIST_DIR"):
        settings.chroma_persist_dir = project_root / "data" / "chroma"
    if not _env_present("AUDIO_EVENT_CALIBRATION_PATH"):
        settings.audio_event_calibration_path = settings.model_cache_dir / "audio_event_calibration.json"

    unified_memory = float(gpu_info.get("unified_memory_gb") or 0.0)
    if unified_memory and unified_memory <= 24 and not _env_present("GPU_MEMORY_FRACTION"):
        settings.gpu_memory_fraction = 0.65


def detect_gpu_info() -> dict:
    """Detect GPU information from the system."""
    gpu_info = {
        "nvidia_gpus": [],
        "amd_gpus": [],
        "apple_gpus": [],
        "total_vram_gb": 0,
        "available_vram_gb": 0,
        "system_memory_gb": 0,
        "nvidia_topology": {
            "connections": {},
            "nvlink_groups": [],
        },
    }

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        total_memory_gb = _detect_macos_total_memory_gb()
        available_memory_gb = _detect_macos_available_memory_gb() or total_memory_gb
        if total_memory_gb > 0:
            chip_name = _detect_apple_chip_name()
            gpu_info["system_memory_gb"] = total_memory_gb
            gpu_info["unified_memory_gb"] = total_memory_gb
            gpu_info["total_vram_gb"] = total_memory_gb
            gpu_info["available_vram_gb"] = available_memory_gb
            gpu_info["apple_gpus"].append({
                "index": 0,
                "name": chip_name,
                "vram_gb": total_memory_gb,
                "free_vram_gb": available_memory_gb,
                "unified_memory": True,
            })
            return gpu_info

    # Check for NVIDIA GPUs
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
        )
        if result.returncode == 0:
            for gpu in _parse_nvidia_gpu_lines(result.stdout):
                gpu_info["nvidia_gpus"].append(gpu)
                gpu_info["total_vram_gb"] += gpu["vram_gb"]
                gpu_info["available_vram_gb"] += gpu.get("free_vram_gb", gpu["vram_gb"])

            topo_result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True,
                text=True,
                timeout=NVIDIA_SMI_TIMEOUT_S,
            )
            if topo_result.returncode == 0:
                gpu_info["nvidia_topology"] = _parse_nvidia_topology(topo_result.stdout)
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
                    gpu_info["system_memory_gb"] = total_ram_gb
                    # If we have lots of RAM and AMD GPU, might be Strix Halo
                    if total_ram_gb >= 96 and gpu_info["amd_gpus"] and not gpu_info["nvidia_gpus"]:
                        gpu_info["unified_memory_gb"] = total_ram_gb
    except FileNotFoundError:
        pass

    return gpu_info


def _parse_nvidia_gpu_lines(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        index = int(parts[0])
        total_mb = int(parts[2])
        used_mb = int(parts[3])
        free_mb = int(parts[4])
        try:
            utilization = float(parts[5])
        except ValueError:
            utilization = 0.0
        gpus.append({
            "index": index,
            "name": parts[1],
            "vram_gb": total_mb / 1024,
            "used_vram_gb": used_mb / 1024,
            "free_vram_gb": free_mb / 1024,
            "utilization_gpu": utilization,
            "bus_id": parts[6],
        })
    return gpus


def _parse_nvidia_topology(output: str) -> dict[str, Any]:
    lines = [line.rstrip("\n") for line in output.splitlines() if line.strip()]
    if not lines:
        return {"connections": {}, "nvlink_groups": []}

    header_line = next((line for line in lines if line.lstrip().startswith("GPU0")), "")
    if not header_line:
        return {"connections": {}, "nvlink_groups": []}
    header = [part.strip() for part in header_line.split() if part.strip().startswith("GPU")]
    connections: dict[int, dict[int, str]] = {}
    nvlink_pairs: set[tuple[int, int]] = set()

    for line in lines:
        stripped = line.lstrip()
        if not stripped.startswith("GPU"):
            continue
        parts = [part.strip() for part in stripped.split()]
        row_label = parts[0]
        if row_label not in header:
            continue
        row_index = int(row_label.removeprefix("GPU"))
        row_connections: dict[int, str] = {}
        for column_label, value in zip(header, parts[1:1 + len(header)], strict=False):
            if column_label == row_label:
                continue
            column_index = int(column_label.removeprefix("GPU"))
            row_connections[column_index] = value
            if value.startswith("NV"):
                nvlink_pairs.add(tuple(sorted((row_index, column_index))))
        connections[row_index] = row_connections

    nvlink_groups = [list(pair) for pair in sorted(nvlink_pairs)]
    return {
        "connections": connections,
        "nvlink_groups": nvlink_groups,
    }


def auto_detect_hardware_profile(gpu_info: dict) -> tuple[HardwareProfile, GPUBackend]:
    """Auto-detect the best hardware profile based on available GPUs."""
    nvidia_gpus = gpu_info.get("nvidia_gpus", [])
    amd_gpus = gpu_info.get("amd_gpus", [])
    apple_gpus = gpu_info.get("apple_gpus", [])
    unified_memory = gpu_info.get("unified_memory_gb", 0)
    total_vram = gpu_info.get("total_vram_gb", 0)

    if apple_gpus:
        return HardwareProfile.APPLE_SILICON, GPUBackend.METAL

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
        return HardwareProfile.STRIX_HALO, GPUBackend.ROCM

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
    if profile in {HardwareProfile.STRIX_HALO, HardwareProfile.APPLE_SILICON}:
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
    gpu_info = detect_gpu_info()

    # Auto-detect hardware if not explicitly set
    if settings.hardware_profile is None or settings.gpu_backend is None:
        detected_profile, detected_backend = auto_detect_hardware_profile(gpu_info)

        if settings.hardware_profile is None:
            settings.hardware_profile = detected_profile
        if settings.gpu_backend is None:
            settings.gpu_backend = detected_backend

    # Set model loading strategy based on hardware defaults before runtime planning.
    if settings.model_loading == ModelLoadingStrategy.SEQUENTIAL:
        settings.model_loading = get_model_loading_strategy(settings.hardware_profile)

    _apply_apple_silicon_defaults(settings, gpu_info)

    # Auto-attach mmproj/chat format for Qwen3VL GGUFs when not explicitly set
    if not settings.vision_mmproj:
        vision_name = settings.vision_model.lower()
        if "qwen3vl-32b-instruct" in vision_name:
            settings.vision_mmproj = "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"
        elif "qwen2.5-vl-3b-instruct" in vision_name:
            settings.vision_mmproj = "Qwen2.5-VL-3B-Instruct.mmproj-fp16.gguf"
    if not settings.vision_chat_format:
        vision_name = settings.vision_model.lower()
        if "qwen3vl-32b-instruct" in vision_name:
            settings.vision_chat_format = "qwen3-vl"
        elif "qwen2.5-vl-3b-instruct" in vision_name:
            settings.vision_chat_format = "qwen2.5-vl"

    # Adjust batch concurrent jobs based on hardware
    if settings.hardware_profile == HardwareProfile.MULTI_GPU:
        settings.batch_concurrent_jobs = max(2, settings.batch_concurrent_jobs)

    if settings.runtime_planner_enabled:
        runtime_plan = build_runtime_plan(settings, gpu_info)
        settings.model_loading = runtime_plan.worker_model_loading
        if runtime_plan.requires_endpoint_idle_teardown and settings.rocm_llm_idle_timeout_s <= 0:
            settings.rocm_llm_idle_timeout_s = runtime_plan.endpoint_idle_timeout_recommendation_s

    validate_hardware_requirements(settings)

    return settings


def get_hardware_info() -> dict:
    """Get detailed hardware information for the settings API."""
    gpu_info = detect_gpu_info()
    settings = get_settings()

    whisper_name = settings.whisper_model.lower()
    if get_asr_family(settings.whisper_model) == "granite":
        if settings.gpu_backend == GPUBackend.CUDA and not settings.granite_force_cpu:
            whisper_backend = "cuda"
        else:
            whisper_backend = "cpu"
    else:
        if settings.gpu_backend == GPUBackend.CUDA:
            whisper_backend = "cuda"
        elif settings.gpu_backend == GPUBackend.METAL:
            whisper_backend = "metal"
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
            "apple": gpu_info.get("apple_gpus", []),
        },
        "unified_memory_gb": gpu_info.get("unified_memory_gb"),
        "total_vram_gb": gpu_info.get("total_vram_gb", 0),
        "available_vram_gb": gpu_info.get("available_vram_gb", gpu_info.get("total_vram_gb", 0)),
        "active_profile": settings.hardware_profile.value,
        "active_backend": settings.gpu_backend.value,
        "whisper_backend": whisper_backend,
        "asr_family": get_asr_family(settings.whisper_model),
        "model_loading_strategy": settings.model_loading.value,
        "rocm_llm_runtime": settings.rocm_llm_runtime.value,
        "rocm_llm_idle_timeout_s": settings.rocm_llm_idle_timeout_s,
        "runtime_planner_enabled": settings.runtime_planner_enabled,
        "runtime_memory_ceiling_gb": settings.runtime_memory_ceiling_gb,
        "gpu_memory_fraction": settings.gpu_memory_fraction,
        "runtime_plan": build_runtime_plan(settings, gpu_info).to_dict(),
    }
