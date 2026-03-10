import sys
import types
from io import StringIO

import pytest

from app.config import (
    GPUBackend,
    HardwareProfile,
    ROCmLLMRuntime,
    Settings,
    auto_detect_hardware_profile,
    detect_gpu_info,
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


def test_settings_default_to_managed_llama_server_runtime():
    settings = Settings()

    assert settings.rocm_llm_runtime == ROCmLLMRuntime.LLAMA_SERVER
    assert settings.rocm_llm_idle_timeout_s == 120
