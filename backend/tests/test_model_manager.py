import sys
import types

import pytest

from app.config import GPUBackend, HardwareProfile, ModelLoadingStrategy
from app.core import model_manager as model_manager_module
from app.core.model_manager import (
    ModelManager,
    _normalize_whisper_model_name,
    _offload_model_to_cpu,
    _resolve_llama_gpu_layers,
    _torch_device,
)


def test_resolve_llama_gpu_layers_defaults_to_cpu_on_cpu_backend():
    assert _resolve_llama_gpu_layers(GPUBackend.CPU, None) == 0


def test_resolve_llama_gpu_layers_uses_gpu_defaults_for_rocm():
    assert _resolve_llama_gpu_layers(GPUBackend.ROCM, None, rocm_safe_default=True) == -1
    assert _resolve_llama_gpu_layers(GPUBackend.ROCM, None, rocm_safe_default=False) == -1


def test_resolve_llama_gpu_layers_respects_explicit_override():
    assert _resolve_llama_gpu_layers(GPUBackend.ROCM, 32, rocm_safe_default=True) == 32


def test_normalize_whisper_model_name_uses_transformers_ids_for_rocm():
    assert _normalize_whisper_model_name("large-v3", backend=GPUBackend.ROCM) == "openai/whisper-large-v3"
    assert (
        _normalize_whisper_model_name("whisper-large-v3-turbo", backend=GPUBackend.ROCM)
        == "openai/whisper-large-v3-turbo"
    )


def test_normalize_whisper_model_name_uses_faster_whisper_aliases_for_cuda():
    assert _normalize_whisper_model_name("large-v3", backend=GPUBackend.CUDA) == "large-v3"
    assert _normalize_whisper_model_name("openai/whisper-large-v3", backend=GPUBackend.CUDA) == "large-v3"
    assert _normalize_whisper_model_name("whisper-large", backend=GPUBackend.CUDA) == "large"


def test_torch_device_raises_when_strict_gpu_runtime_is_missing(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="embedding requires rocm acceleration"):
        _torch_device(GPUBackend.ROCM, strict=True, runtime_label="embedding")


def test_torch_device_returns_cuda_when_gpu_is_available(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _torch_device(GPUBackend.ROCM, strict=True, runtime_label="embedding") == "cuda"


def test_torch_device_returns_specific_cuda_index_when_requested(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _torch_device(
        GPUBackend.CUDA,
        strict=True,
        runtime_label="embedding",
        device_index=5,
    ) == "cuda:5"


def test_offload_model_to_cpu_handles_nested_model_bundles():
    events: list[str] = []

    class CpuModel:
        def cpu(self):
            events.append("cpu")

    class ToModel:
        def to(self, _device):
            events.append("to")

    _offload_model_to_cpu({
        "primary": CpuModel(),
        "nested": {"secondary": ToModel()},
        "plain": object(),
    })

    assert events == ["cpu", "to"]


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
    monkeypatch.setattr(
        model_manager_module,
        "build_runtime_plan",
        lambda *_args, **_kwargs: types.SimpleNamespace(keep_resident_models=(), worker_model_loading=ModelLoadingStrategy.SEQUENTIAL),
    )
    monkeypatch.setattr(model_manager_module, "detect_gpu_info", lambda: {"nvidia_gpus": [], "amd_gpus": []})

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


@pytest.mark.asyncio
async def test_release_model_keeps_resident_models_loaded(monkeypatch, tmp_path):
    settings = types.SimpleNamespace(
        whisper_model="large-v3",
        gpu_backend=GPUBackend.CPU,
        granite_force_cpu=False,
        model_cache_dir=tmp_path,
        hardware_profile=HardwareProfile.CPU_ONLY,
        model_loading=ModelLoadingStrategy.SEQUENTIAL,
    )
    monkeypatch.setattr(model_manager_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        model_manager_module,
        "build_runtime_plan",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            keep_resident_models=("whisper",),
            worker_model_loading=ModelLoadingStrategy.SEQUENTIAL,
        ),
    )
    monkeypatch.setattr(model_manager_module, "detect_gpu_info", lambda: {"nvidia_gpus": [], "amd_gpus": []})

    manager = ModelManager()
    manager._loaded_models[model_manager_module.ModelType.WHISPER] = object()

    await manager.release_model(model_manager_module.ModelType.WHISPER)

    assert model_manager_module.ModelType.WHISPER in manager._loaded_models


@pytest.mark.asyncio
async def test_load_whisper_uses_faster_whisper_alias_on_cuda(monkeypatch, tmp_path):
    settings = types.SimpleNamespace(
        whisper_model="openai/whisper-large-v3",
        gpu_backend=GPUBackend.CUDA,
        granite_force_cpu=False,
        model_cache_dir=tmp_path,
        hardware_profile=HardwareProfile.MULTI_GPU,
        model_loading=ModelLoadingStrategy.PARALLEL,
    )
    runtime_plan = types.SimpleNamespace(
        keep_resident_models=(),
        worker_model_loading=ModelLoadingStrategy.PARALLEL,
        preferred_worker_device_indices=(5,),
        preferred_endpoint_device_indices=(),
    )
    monkeypatch.setattr(model_manager_module, "get_settings", lambda: settings)
    monkeypatch.setattr(model_manager_module, "detect_gpu_info", lambda: {"nvidia_gpus": [], "amd_gpus": []})
    monkeypatch.setattr(model_manager_module, "build_runtime_plan", lambda *_args, **_kwargs: runtime_plan)

    seen: dict[str, str] = {}

    async def fake_load_faster_whisper(model_name, *_args, **_kwargs):
        seen["model_name"] = model_name
        return object()

    monkeypatch.setattr(model_manager_module, "_load_faster_whisper", fake_load_faster_whisper)

    manager = ModelManager()
    await manager._load_whisper()

    assert seen["model_name"] == "large-v3"


@pytest.mark.asyncio
async def test_load_whisper_uses_transformers_repo_on_rocm(monkeypatch, tmp_path):
    settings = types.SimpleNamespace(
        whisper_model="large-v3",
        gpu_backend=GPUBackend.ROCM,
        granite_force_cpu=False,
        model_cache_dir=tmp_path,
        hardware_profile=HardwareProfile.STRIX_HALO,
        model_loading=ModelLoadingStrategy.SEQUENTIAL,
    )
    runtime_plan = types.SimpleNamespace(
        keep_resident_models=(),
        worker_model_loading=ModelLoadingStrategy.SEQUENTIAL,
        preferred_worker_device_indices=(0,),
        preferred_endpoint_device_indices=(),
    )
    monkeypatch.setattr(model_manager_module, "get_settings", lambda: settings)
    monkeypatch.setattr(model_manager_module, "detect_gpu_info", lambda: {"nvidia_gpus": [], "amd_gpus": []})
    monkeypatch.setattr(model_manager_module, "build_runtime_plan", lambda *_args, **_kwargs: runtime_plan)

    seen: dict[str, str] = {}

    async def fake_load_transformers_whisper(model_name, *_args, **_kwargs):
        seen["model_name"] = model_name
        return {"type": "whisper_transformers"}

    monkeypatch.setattr(model_manager_module, "_load_transformers_whisper", fake_load_transformers_whisper)

    manager = ModelManager()
    await manager._load_whisper()

    assert seen["model_name"] == "openai/whisper-large-v3"


def test_model_manager_uses_planner_gpu_indices_for_cuda_placement(monkeypatch, tmp_path):
    settings = types.SimpleNamespace(
        whisper_model="large-v3",
        gpu_backend=GPUBackend.CUDA,
        granite_force_cpu=False,
        model_cache_dir=tmp_path,
        hardware_profile=HardwareProfile.MULTI_GPU,
        model_loading=ModelLoadingStrategy.PARALLEL,
    )
    runtime_plan = types.SimpleNamespace(
        keep_resident_models=(),
        worker_model_loading=ModelLoadingStrategy.PARALLEL,
        preferred_worker_device_indices=(5,),
        preferred_endpoint_device_indices=(0, 3),
    )
    gpu_info = {
        "nvidia_gpus": [
            {"index": 0, "name": "RTX 3090", "vram_gb": 24.0, "free_vram_gb": 24.0},
            {"index": 3, "name": "RTX 3090", "vram_gb": 24.0, "free_vram_gb": 24.0},
            {"index": 5, "name": "RTX PRO 6000", "vram_gb": 96.0, "free_vram_gb": 96.0},
        ],
        "amd_gpus": [],
    }

    monkeypatch.setattr(model_manager_module, "get_settings", lambda: settings)
    monkeypatch.setattr(model_manager_module, "detect_gpu_info", lambda: gpu_info)
    monkeypatch.setattr(model_manager_module, "build_runtime_plan", lambda *_args, **_kwargs: runtime_plan)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        types.SimpleNamespace(
            LLAMA_SPLIT_MODE_NONE=0,
            LLAMA_SPLIT_MODE_LAYER=1,
        ),
    )

    manager = ModelManager()

    assert manager.worker_gpu_indices == (5,)
    assert manager.endpoint_gpu_indices == (0, 3)
    assert manager._torch_runtime_device(runtime_label="embedding", strict=False) == "cuda:5"
    assert manager._llama_cuda_kwargs({"main_gpu", "split_mode", "tensor_split"}, endpoint=True) == {
        "main_gpu": 0,
        "tensor_split": [0.5, 0.0, 0.0, 0.5],
        "split_mode": 1,
    }
