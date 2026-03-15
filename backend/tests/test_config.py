import os
import sys
import types
from io import StringIO
from pathlib import Path

import pytest

from app.config import (
    NVIDIA_SMI_TIMEOUT_S,
    GPUBackend,
    HardwareProfile,
    ROCmLLMRuntime,
    Settings,
    auto_detect_hardware_profile,
    detect_gpu_info,
    get_settings,
    validate_hardware_requirements,
)


def test_auto_detect_hardware_profile_prefers_rocm_for_strix_halo():
    profile, backend = auto_detect_hardware_profile({
        "amd_gpus": [{"name": "AMD GPU", "vram_gb": 16}],
        "unified_memory_gb": 128,
        "total_vram_gb": 16,
    })

    assert profile == HardwareProfile.STRIX_HALO
    assert backend == GPUBackend.ROCM


def test_auto_detect_hardware_profile_prefers_metal_for_apple_silicon():
    profile, backend = auto_detect_hardware_profile({
        "apple_gpus": [{"name": "Apple M4", "vram_gb": 16}],
        "unified_memory_gb": 16,
        "total_vram_gb": 16,
    })

    assert profile == HardwareProfile.APPLE_SILICON
    assert backend == GPUBackend.METAL


def test_detect_gpu_info_does_not_flag_strix_halo_without_amd_gpu(monkeypatch):
    class Completed:
        def __init__(self, returncode: int, stdout: str = ""):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(*_args, **_kwargs):
        return Completed(returncode=1)

    def fake_open(path, *_args, **_kwargs):
        assert path == "/proc/meminfo"
        return StringIO("MemTotal:       134217728 kB\n")

    monkeypatch.setattr("app.config.subprocess.run", fake_run)
    monkeypatch.setattr("builtins.open", fake_open)

    gpu_info = detect_gpu_info()

    assert "unified_memory_gb" not in gpu_info


def test_detect_gpu_info_parses_apple_silicon_memory(monkeypatch):
    class Completed:
        def __init__(self, returncode: int, stdout: str = ""):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(cmd, *_args, **_kwargs):
        if cmd == ["sysctl", "-n", "hw.memsize"]:
            return Completed(returncode=0, stdout=str(16 * 1024**3))
        if cmd == ["vm_stat"]:
            return Completed(
                returncode=0,
                stdout=(
                    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                    "Pages free: 65536.\n"
                    "Pages inactive: 131072.\n"
                    "Pages speculative: 32768.\n"
                ),
            )
        if cmd == ["sysctl", "-n", "machdep.cpu.brand_string"]:
            return Completed(returncode=0, stdout="Apple M4")
        return Completed(returncode=1)

    monkeypatch.setattr("app.config.platform.system", lambda: "Darwin")
    monkeypatch.setattr("app.config.platform.machine", lambda: "arm64")
    monkeypatch.setattr("app.config.subprocess.run", fake_run)

    gpu_info = detect_gpu_info()

    assert gpu_info["unified_memory_gb"] == pytest.approx(16.0, rel=1e-4)
    assert gpu_info["available_vram_gb"] == pytest.approx(3.5, rel=1e-4)
    assert gpu_info["apple_gpus"][0]["name"] == "Apple M4"


def test_detect_gpu_info_parses_nvidia_free_memory_and_topology(monkeypatch):
    class Completed:
        def __init__(self, returncode: int, stdout: str = ""):
            self.returncode = returncode
            self.stdout = stdout

    gpu_stdout = "\n".join([
        "0, NVIDIA GeForce RTX 3090, 24576, 0, 24127, 0, 00000000:01:00.0",
        "5, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 0, 97250, 0, 00000000:E1:00.0",
        "6, NVIDIA GeForce RTX 3090, 24576, 0, 24127, 0, 00000000:E2:00.0",
    ])
    topo_stdout = "\n".join([
        "GPU0\tGPU5\tGPU6\tCPU Affinity",
        "GPU0\tX\tNODE\tNV4\t0-31",
        "GPU5\tNODE\tX\tPHB\t0-31",
        "GPU6\tNV4\tPHB\tX\t0-31",
    ])

    def fake_run(cmd, *_args, **_kwargs):
        if cmd[:2] == ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,pci.bus_id"]:
            return Completed(returncode=0, stdout=gpu_stdout)
        if cmd[:3] == ["nvidia-smi", "topo", "-m"]:
            return Completed(returncode=0, stdout=topo_stdout)
        return Completed(returncode=1)

    def fake_open(path, *_args, **_kwargs):
        assert path == "/proc/meminfo"
        return StringIO("MemTotal:       134217728 kB\n")

    monkeypatch.setattr("app.config.subprocess.run", fake_run)
    monkeypatch.setattr("builtins.open", fake_open)

    gpu_info = detect_gpu_info()

    assert gpu_info["total_vram_gb"] == pytest.approx((24576 + 97887 + 24576) / 1024, rel=1e-4)
    assert gpu_info["available_vram_gb"] == pytest.approx((24127 + 97250 + 24127) / 1024, rel=1e-4)
    assert gpu_info["nvidia_gpus"][1]["name"] == "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    assert gpu_info["nvidia_gpus"][1]["free_vram_gb"] == pytest.approx(97250 / 1024, rel=1e-4)
    assert gpu_info["nvidia_topology"]["nvlink_groups"] == [[0, 6]]


def test_detect_gpu_info_uses_longer_nvidia_timeout(monkeypatch):
    class Completed:
        def __init__(self, returncode: int, stdout: str = ""):
            self.returncode = returncode
            self.stdout = stdout

    timeouts = []

    def fake_run(cmd, *_args, **kwargs):
        if cmd[0] == "nvidia-smi":
            timeouts.append(kwargs.get("timeout"))
        return Completed(returncode=1)

    def fake_open(path, *_args, **_kwargs):
        assert path == "/proc/meminfo"
        return StringIO("MemTotal:       134217728 kB\n")

    monkeypatch.setattr("app.config.subprocess.run", fake_run)
    monkeypatch.setattr("builtins.open", fake_open)

    detect_gpu_info()

    assert timeouts == [NVIDIA_SMI_TIMEOUT_S]


def test_validate_hardware_requirements_rejects_non_rocm_strix_halo():
    settings = Settings(
        hardware_profile=HardwareProfile.STRIX_HALO,
        gpu_backend=GPUBackend.VULKAN,
    )

    with pytest.raises(RuntimeError, match="requires GPU_BACKEND=rocm"):
        validate_hardware_requirements(settings)


def test_validate_hardware_requirements_rejects_cpu_forced_granite():
    settings = Settings(
        hardware_profile=HardwareProfile.STRIX_HALO,
        gpu_backend=GPUBackend.ROCM,
        granite_force_cpu=True,
    )

    with pytest.raises(RuntimeError, match="GRANITE_FORCE_CPU"):
        validate_hardware_requirements(settings)


def test_validate_hardware_requirements_rejects_non_rocm_torch(monkeypatch):
    settings = Settings(
        hardware_profile=HardwareProfile.STRIX_HALO,
        gpu_backend=GPUBackend.ROCM,
    )
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        version=types.SimpleNamespace(hip=None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="ROCm-enabled PyTorch build"):
        validate_hardware_requirements(settings)


def test_validate_hardware_requirements_requires_visible_rocm_device(monkeypatch):
    settings = Settings(
        hardware_profile=HardwareProfile.STRIX_HALO,
        gpu_backend=GPUBackend.ROCM,
    )
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        version=types.SimpleNamespace(hip="7.2.0"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="torch.cuda.is_available\\(\\) is false"):
        validate_hardware_requirements(settings)


def test_validate_hardware_requirements_accepts_rocm_ready_strix_halo(monkeypatch):
    settings = Settings(
        hardware_profile=HardwareProfile.STRIX_HALO,
        gpu_backend=GPUBackend.ROCM,
    )
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        version=types.SimpleNamespace(hip="7.2.0"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    validate_hardware_requirements(settings)


def test_settings_default_to_managed_llama_server_runtime(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNTIME_MEMORY_CEILING_GB", raising=False)
    monkeypatch.delenv("GPU_MEMORY_FRACTION", raising=False)
    monkeypatch.delenv("RUNTIME_PLANNER_ENABLED", raising=False)
    settings = Settings()

    assert settings.rocm_llm_runtime == ROCmLLMRuntime.LLAMA_SERVER
    assert settings.rocm_llm_idle_timeout_s == 120
    assert settings.runtime_planner_enabled is True
    assert settings.runtime_memory_ceiling_gb is None
    assert settings.vision_model == "Qwen3VL-32B-Instruct-Q4_K_M.gguf"


def test_get_settings_attaches_qwen3vl_defaults(monkeypatch):
    get_settings.cache_clear()


def test_get_settings_applies_apple_silicon_small_model_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    monkeypatch.delenv("HARDWARE_PROFILE", raising=False)
    monkeypatch.delenv("GPU_BACKEND", raising=False)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)
    monkeypatch.delenv("VISION_MMPROJ", raising=False)
    monkeypatch.delenv("VISION_CHAT_FORMAT", raising=False)
    monkeypatch.delenv("SUMMARIZATION_MODEL", raising=False)
    monkeypatch.delenv("GPU_MEMORY_FRACTION", raising=False)
    monkeypatch.setattr(
        "app.config.detect_gpu_info",
        lambda: {
            "apple_gpus": [{"index": 0, "name": "Apple M4", "vram_gb": 16.0, "free_vram_gb": 10.0}],
            "nvidia_gpus": [],
            "amd_gpus": [],
            "unified_memory_gb": 16.0,
            "total_vram_gb": 16.0,
            "available_vram_gb": 10.0,
        },
    )

    settings = get_settings()

    assert settings.hardware_profile == HardwareProfile.APPLE_SILICON
    assert settings.gpu_backend == GPUBackend.METAL
    assert settings.whisper_model == "small"
    assert settings.embedding_model == "nomic-ai/nomic-embed-text-v1.5"
    assert settings.vision_model == "Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf"
    assert settings.vision_mmproj == "Qwen2.5-VL-3B-Instruct.mmproj-fp16.gguf"
    assert settings.vision_chat_format == "qwen2.5-vl"
    assert settings.summarization_model == "Qwen2.5-3B-Instruct.Q4_K_M.gguf"
    assert settings.upload_dir == Path("/Users/vac/EchOnyx/data/uploads")
    assert settings.model_cache_dir == Path("/Users/vac/EchOnyx/data/models")
    assert settings.chroma_persist_dir == Path("/Users/vac/EchOnyx/data/chroma")
    assert settings.audio_event_calibration_path == Path("/Users/vac/EchOnyx/data/models/audio_event_calibration.json")
    assert settings.gpu_memory_fraction == 0.65
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"

    get_settings.cache_clear()

    monkeypatch.setenv("HARDWARE_PROFILE", "cpu_only")
    monkeypatch.setenv("GPU_BACKEND", "cpu")
    monkeypatch.setenv("RUNTIME_PLANNER_ENABLED", "false")
    monkeypatch.setenv("VISION_MODEL", "Qwen3VL-32B-Instruct-Q4_K_M.gguf")
    monkeypatch.delenv("VISION_MMPROJ", raising=False)
    monkeypatch.delenv("VISION_CHAT_FORMAT", raising=False)

    settings = get_settings()

    assert settings.vision_mmproj == "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"
    assert settings.vision_chat_format == "qwen3-vl"

    get_settings.cache_clear()


def test_get_settings_uses_runtime_planner_with_explicit_hardware(monkeypatch):
    get_settings.cache_clear()

    monkeypatch.setenv("HARDWARE_PROFILE", "strix_halo")
    monkeypatch.setenv("GPU_BACKEND", "rocm")
    monkeypatch.setenv("RUNTIME_PLANNER_ENABLED", "true")

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        version=types.SimpleNamespace(hip="7.2.0"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    monkeypatch.setattr(
        "app.config.detect_gpu_info",
        lambda: {
            "amd_gpus": [{"name": "AMD GPU", "vram_gb": 16}],
            "nvidia_gpus": [],
            "total_vram_gb": 16,
            "unified_memory_gb": 128,
        },
    )

    settings = get_settings()

    assert settings.hardware_profile == HardwareProfile.STRIX_HALO
    assert settings.gpu_backend == GPUBackend.ROCM

    get_settings.cache_clear()


def test_get_settings_treats_blank_autodetect_envs_as_unset(monkeypatch):
    get_settings.cache_clear()

    monkeypatch.setenv("HARDWARE_PROFILE", "")
    monkeypatch.setenv("GPU_BACKEND", "")
    monkeypatch.setenv("RUNTIME_PLANNER_ENABLED", "false")

    monkeypatch.setattr(
        "app.config.detect_gpu_info",
        lambda: {
            "amd_gpus": [],
            "nvidia_gpus": [
                {"index": 0, "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "vram_gb": 95.6},
                {"index": 1, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0},
            ],
            "total_vram_gb": 119.6,
            "available_vram_gb": 118.0,
        },
    )

    settings = get_settings()

    assert settings.hardware_profile == HardwareProfile.MULTI_GPU
    assert settings.gpu_backend == GPUBackend.CUDA

    get_settings.cache_clear()
