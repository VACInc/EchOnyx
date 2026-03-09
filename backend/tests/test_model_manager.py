import sys
import types

import pytest

from app.config import GPUBackend
from app.core.model_manager import _resolve_llama_gpu_layers, _torch_device


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
