from types import SimpleNamespace

from app.config import GPUBackend, HardwareProfile
from app.core import diarization as diarization_module


def test_should_retry_on_cpu_after_gpu_error_for_non_strict_runtime(monkeypatch):
    settings = SimpleNamespace(
        hardware_profile=HardwareProfile.CPU_ONLY,
        gpu_backend=GPUBackend.ROCM,
    )
    monkeypatch.setattr(diarization_module, "get_settings", lambda: settings)

    assert diarization_module._should_retry_on_cpu_after_gpu_error(RuntimeError("MIOpen failure"))


def test_should_not_retry_on_cpu_after_gpu_error_for_strix_halo(monkeypatch):
    settings = SimpleNamespace(
        hardware_profile=HardwareProfile.STRIX_HALO,
        gpu_backend=GPUBackend.ROCM,
    )
    monkeypatch.setattr(diarization_module, "get_settings", lambda: settings)

    assert not diarization_module._should_retry_on_cpu_after_gpu_error(RuntimeError("HIP kernel failure"))
