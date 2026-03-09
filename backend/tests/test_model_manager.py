import sys
import types

import pytest

from app.config import GPUBackend, HardwareProfile, ModelLoadingStrategy
from app.core import model_manager as model_manager_module
from app.core.model_manager import ModelManager, _resolve_llama_gpu_layers, _torch_device


def test_resolve_llama_gpu_layers_defaults_to_cpu_on_cpu_backend():
    assert _resolve_llama_gpu_layers(GPUBackend.CPU, None) == 0


def test_resolve_llama_gpu_layers_uses_gpu_defaults_for_rocm():
    assert _resolve_llama_gpu_layers(GPUBackend.ROCM, None, rocm_safe_default=True) == -1
    assert _resolve_llama_gpu_layers(GPUBackend.ROCM, None, rocm_safe_default=False) == -1


def test_resolve_llama_gpu_layers_respects_explicit_override():
    assert _resolve_llama_gpu_layers(GPUBackend.ROCM, 32, rocm_safe_default=True) == 32


def test_torch_device_raises_when_strict_gpu_runtime_is_missing(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="embedding requires rocm acceleration"):
        _torch_device(GPUBackend.ROCM, strict=True, runtime_label="embedding")


def test_torch_device_returns_cuda_when_gpu_is_available(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _torch_device(GPUBackend.ROCM, strict=True, runtime_label="embedding") == "cuda"


@pytest.mark.asyncio
async def test_load_whisper_canary_failure_raises_without_fallback(monkeypatch, tmp_path):
    settings = types.SimpleNamespace(
        whisper_model="nvidia/canary-qwen-2.5b",
        gpu_backend=GPUBackend.CPU,
        granite_force_cpu=False,
        model_cache_dir=tmp_path,
        hardware_profile=HardwareProfile.CPU_ONLY,
        model_loading=ModelLoadingStrategy.SEQUENTIAL,
    )
    monkeypatch.setattr(model_manager_module, "get_settings", lambda: settings)

    class FakeSALM:
        @staticmethod
        def from_pretrained(_model_name):
            raise RuntimeError("canary load failed")

    monkeypatch.setitem(
        sys.modules,
        "nemo.collections.speechlm2.models",
        types.SimpleNamespace(SALM=FakeSALM),
    )

    manager = ModelManager()

    with pytest.raises(RuntimeError, match="canary load failed"):
        await manager._load_whisper()
