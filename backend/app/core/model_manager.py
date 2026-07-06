"""Model manager for loading/unloading models based on hardware profile."""

import asyncio
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from app.config import GPUBackend, HardwareProfile, ModelLoadingStrategy, detect_gpu_info, get_settings
from app.core.model_downloader import (
    is_model_cached,
    missing_model_download_message,
    model_download_target,
    resolve_local_model_path,
)
from app.runtime.planner import build_runtime_plan

logger = logging.getLogger(__name__)


def _is_granite_speech_model(model_name: str) -> bool:
    """Heuristic to detect Granite speech models."""
    return "granite-speech" in model_name.lower()


def _is_canary_model(model_name: str) -> bool:
    """Heuristic to detect NVIDIA Canary models."""
    lowered = model_name.lower()
    return "canary" in lowered or "nvidia/canary" in lowered


def _is_clap_model(model_name: str) -> bool:
    """Heuristic to detect CLAP audio models."""
    return "clap" in model_name.lower()


def _embedding_requires_trust_remote_code(model_name: str) -> bool:
    return model_name.lower().startswith("nomic-ai/")


def _normalize_whisper_model_name(
    model_name: str,
    *,
    backend: GPUBackend | None = None,
) -> str:
    """Normalize Whisper aliases for the selected runtime backend."""
    lower = model_name.lower()
    if backend == GPUBackend.CUDA:
        if lower in {"large-v3-turbo", "whisper-large-v3-turbo", "openai/whisper-large-v3-turbo"}:
            return "large-v3-turbo"
        if lower in {"large-v3", "whisper-large-v3", "openai/whisper-large-v3"}:
            return "large-v3"
        if lower in {"large", "whisper-large", "openai/whisper-large"}:
            return "large"
        if lower.startswith("openai/whisper-"):
            return lower.removeprefix("openai/whisper-")
        if lower.startswith("whisper-") and "/" not in model_name:
            return lower.removeprefix("whisper-")
        return model_name

    if lower in {"large-v3-turbo", "whisper-large-v3-turbo"}:
        return "openai/whisper-large-v3-turbo"
    if lower in {"large-v3", "whisper-large-v3"}:
        return "openai/whisper-large-v3"
    if lower == "whisper-large":
        return "openai/whisper-large"
    if "/" not in model_name and lower in {"tiny", "base", "small", "medium", "large"}:
        return f"openai/whisper-{lower}"
    if lower.startswith("whisper-") and "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


def _torch_dtype_for_backend(backend: GPUBackend):
    """Select a safer dtype for the given backend."""
    try:
        import torch
    except Exception:
        return None
    if backend == GPUBackend.ROCM:
        return torch.float32
    return torch.float16 if backend in {GPUBackend.CUDA, GPUBackend.METAL} else torch.float32


def _is_accelerated_torch_device_name(device: str) -> bool:
    return device == "mps" or device == "cuda" or device.startswith("cuda:")


def _cuda_local_device_index(host_index: int | None, *, runtime_label: str = "model") -> int | None:
    """Translate a planner host GPU index to this process's CUDA-local ordinal."""
    if host_index is None:
        return None

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return host_index

    visible_tokens = [part.strip() for part in visible.split(",") if part.strip()]
    try:
        visible_indices = [int(token) for token in visible_tokens]
    except ValueError:
        logger.warning(
            "CUDA_VISIBLE_DEVICES=%s is not a simple integer list; planner-selected GPU %s "
            "for %s cannot be translated, falling back to CUDA-local device 0.",
            visible,
            host_index,
            runtime_label,
        )
        return 0

    if host_index in visible_indices:
        return visible_indices.index(host_index)

    logger.warning(
        "Planner-selected GPU %s for %s is not visible in CUDA_VISIBLE_DEVICES=%s; "
        "falling back to CUDA-local device 0.",
        host_index,
        runtime_label,
        visible,
    )
    return 0


def _torch_device(
    backend: GPUBackend,
    *,
    strict: bool = False,
    runtime_label: str = "model",
    device_index: int | None = None,
) -> str:
    """Map backend choice to torch device name with safe fallback."""
    if backend.value in {"cuda", "rocm"}:
        try:
            import torch
            if torch.cuda.is_available():
                if device_index is not None:
                    return f"cuda:{device_index}"
                return "cuda"
        except Exception:
            if strict:
                raise RuntimeError(
                    f"{runtime_label} requires {backend.value} acceleration, but PyTorch GPU support is unavailable."
                ) from None
            logger.warning("Torch CUDA/ROCm not available, falling back to CPU.")
            return "cpu"
        if strict:
            raise RuntimeError(
                f"{runtime_label} requires {backend.value} acceleration, but no GPU device is available."
            )
        logger.warning("GPU backend requested but no CUDA/ROCm device found, using CPU.")
    if backend == GPUBackend.METAL:
        try:
            import torch

            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
        except Exception:
            if strict:
                raise RuntimeError(
                    f"{runtime_label} requires metal acceleration, but PyTorch MPS support is unavailable."
                ) from None
            logger.warning("PyTorch MPS not available, falling back to CPU.")
            return "cpu"
        if strict:
            raise RuntimeError(
                f"{runtime_label} requires metal acceleration, but no MPS device is available."
            )
        logger.warning("Metal backend requested but no MPS device found, using CPU.")
    return "cpu"


def _faster_whisper_device(backend: GPUBackend) -> str:
    """Map backend to faster-whisper device name."""
    # faster-whisper (CTranslate2) only supports NVIDIA CUDA for GPU
    if backend == GPUBackend.CUDA:
        return "cuda"
    return "cpu"


def _faster_whisper_compute_type(backend: GPUBackend) -> str:
    """Pick faster-whisper compute type based on backend."""
    if backend == GPUBackend.CUDA:
        return "float16"
    # Favor accuracy for CPU fallback
    return "float32"


def _resolve_llama_gpu_layers(
    backend: GPUBackend,
    configured_layers: int | None,
    *,
    rocm_safe_default: bool = False,
) -> int:
    """Resolve llama.cpp GPU layer count defaults."""
    if configured_layers is not None:
        return configured_layers
    if backend == GPUBackend.CPU:
        return 0
    return -1


def _offload_model_to_cpu(model: Any) -> None:
    """Best-effort move of a loaded model bundle back to CPU before deletion."""
    if isinstance(model, dict):
        for value in model.values():
            _offload_model_to_cpu(value)
        return

    if hasattr(model, "cpu"):
        try:
            model.cpu()
            return
        except Exception:
            logger.debug("Model.cpu() offload failed during unload.", exc_info=True)

    if hasattr(model, "to"):
        try:
            model.to("cpu")
            return
        except Exception:
            logger.debug("Model.to(cpu) offload failed during unload.", exc_info=True)


async def _load_faster_whisper(
    model_name: str,
    backend: GPUBackend,
    cache_dir: Path,
    *,
    device_index: int | None = None,
) -> Any:
    """Load a faster-whisper model by name."""
    from inspect import signature
    from faster_whisper import WhisperModel

    device = _faster_whisper_device(backend)
    compute_type = _faster_whisper_compute_type(backend)
    if backend in {GPUBackend.ROCM, GPUBackend.VULKAN, GPUBackend.METAL} and device == "cpu":
        logger.warning(
            "faster-whisper GPU acceleration is CUDA-only; using CPU for %s backend.",
            backend.value,
        )

    loop = asyncio.get_event_loop()
    init_params = {
        "model_size_or_path": model_name,
        "device": device,
        "compute_type": compute_type,
        "download_root": str(cache_dir),
    }
    if device == "cuda" and device_index is not None:
        param_names = set(signature(WhisperModel).parameters.keys())
        if "device_index" in param_names:
            init_params["device_index"] = device_index

    return await loop.run_in_executor(
        None,
        lambda: WhisperModel(**init_params),
    )


def _attn_implementation_for_backend(backend: GPUBackend) -> str | None:
    """Select a safer attention implementation for a backend if supported."""
    if backend == GPUBackend.ROCM:
        return "eager"
    return None


async def _load_transformers_whisper(
    model_name: str,
    backend: GPUBackend,
    cache_dir: Path,
    *,
    strict_accelerator: bool = False,
    device_index: int | None = None,
) -> Any:
    """Load a Whisper model via transformers for ROCm/CPU compatibility."""
    from inspect import signature
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch

    device = _torch_device(
        backend,
        strict=strict_accelerator,
        runtime_label="transcription",
        device_index=device_index,
    )
    dtype = _torch_dtype_for_backend(backend) or (
        torch.float16 if _is_accelerated_torch_device_name(device) else torch.float32
    )
    loop = asyncio.get_event_loop()

    def load_whisper():
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
        )
        model_kwargs = {
            "cache_dir": str(cache_dir),
        }
        param_names = set(signature(AutoModelForSpeechSeq2Seq.from_pretrained).parameters.keys())
        if "dtype" in param_names:
            model_kwargs["dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = dtype
        attn_impl = _attn_implementation_for_backend(backend)
        if attn_impl and "attn_implementation" in param_names:
            model_kwargs["attn_implementation"] = attn_impl
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            **model_kwargs,
        )
        if _is_accelerated_torch_device_name(device):
            model = model.to(torch.device(device))
        return {
            "type": "whisper_transformers",
            "model": model,
            "processor": processor,
            "device": device,
        }

    return await loop.run_in_executor(None, load_whisper)


class ModelType(str, Enum):
    """Types of models that can be loaded."""

    WHISPER = "whisper"
    DIARIZATION = "diarization"
    VISION = "vision"
    SUMMARIZATION = "summarization"
    EMBEDDING = "embedding"
    AUDIO_EVENT = "audio_event"


class ModelManager:
    """
    Manages model loading and unloading based on hardware constraints.

    For Strix Halo (sequential mode): loads one model at a time
    For Multi-GPU (parallel mode): keeps all models loaded
    """

    def __init__(self):
        self.settings = get_settings()
        self.gpu_info = detect_gpu_info()
        self.runtime_plan = build_runtime_plan(self.settings, self.gpu_info)
        self._loaded_models: dict[ModelType, Any] = {}
        self._lock = asyncio.Lock()
        self._loaded_state_path = self.settings.model_cache_dir / ".loaded_models.json"
        self._persist_loaded_models()

    @property
    def is_sequential(self) -> bool:
        """Check if we're in sequential loading mode."""
        return self.settings.model_loading == ModelLoadingStrategy.SEQUENTIAL

    @property
    def resident_model_types(self) -> set[ModelType]:
        resident: set[ModelType] = set()
        for model_name in self.runtime_plan.keep_resident_models:
            if model_name in ModelType._value2member_map_:
                resident.add(ModelType(model_name))
        return resident

    @property
    def requires_strict_accelerator(self) -> bool:
        """Whether this runtime must fail closed instead of using CPU fallback."""
        return (
            self.settings.hardware_profile == HardwareProfile.STRIX_HALO
            and self.settings.gpu_backend == GPUBackend.ROCM
        )

    @property
    def worker_gpu_indices(self) -> tuple[int, ...]:
        indices = tuple(getattr(self.runtime_plan, "preferred_worker_device_indices", ()) or ())
        if indices:
            return indices
        if self.settings.gpu_backend == GPUBackend.CUDA:
            visible_gpus = self.gpu_info.get("nvidia_gpus", [])
            if visible_gpus and visible_gpus[0].get("index") is not None:
                return (int(visible_gpus[0]["index"]),)
        return ()

    @property
    def endpoint_gpu_indices(self) -> tuple[int, ...]:
        indices = tuple(getattr(self.runtime_plan, "preferred_endpoint_device_indices", ()) or ())
        if indices:
            return indices
        if self.settings.gpu_backend == GPUBackend.CUDA:
            return self.worker_gpu_indices
        return ()

    def _torch_runtime_device(
        self,
        *,
        runtime_label: str,
        strict: bool,
        endpoint: bool = False,
    ) -> str:
        indices = self.endpoint_gpu_indices if endpoint else self.worker_gpu_indices
        device_index = self._cuda_local_device_index(
            indices[0] if indices else None,
            runtime_label=runtime_label,
        )
        return _torch_device(
            self.settings.gpu_backend,
            strict=strict,
            runtime_label=runtime_label,
            device_index=device_index,
        )

    def _cuda_local_device_index(
        self,
        host_index: int | None,
        *,
        runtime_label: str,
    ) -> int | None:
        if self.settings.gpu_backend != GPUBackend.CUDA:
            return host_index
        return _cuda_local_device_index(host_index, runtime_label=runtime_label)

    def _worker_cuda_local_device_index(self, *, runtime_label: str) -> int | None:
        return self._cuda_local_device_index(
            self.worker_gpu_indices[0] if self.worker_gpu_indices else None,
            runtime_label=runtime_label,
        )

    @property
    def model_auto_download_enabled(self) -> bool:
        return bool(getattr(self.settings, "model_auto_download", True))

    def _require_cached_or_auto_download(self, model_name: str, component: str) -> None:
        if self.model_auto_download_enabled:
            return
        if is_model_cached(
            model_name,
            self.settings.model_cache_dir,
            component=component,
            backend=self.settings.gpu_backend,
        ):
            return
        raise RuntimeError(missing_model_download_message(model_name))

    async def _ensure_local_gguf_model(self, configured_model: str, component: str, env_var: str) -> Path:
        model_path = resolve_local_model_path(configured_model, self.settings.model_cache_dir)
        if model_path is not None:
            return model_path

        model_name, target_dir = model_download_target(configured_model, self.settings.model_cache_dir)
        self._require_cached_or_auto_download(model_name, component)

        from app.core.model_downloader import download_model_async

        logger.info("%s model not found locally, downloading...", component.title())
        return await download_model_async(
            model_name,
            target_dir,
            component=component,
            backend=self.settings.gpu_backend,
            env_var=env_var,
        )

    def _llama_cuda_device_selection(self, *, endpoint: bool) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if self.settings.gpu_backend != GPUBackend.CUDA:
            return (), ()

        host_indices = self.endpoint_gpu_indices if endpoint else self.worker_gpu_indices
        if not host_indices:
            return (), ()

        explicit_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if explicit_visible:
            local_indices = tuple(
                _cuda_local_device_index(index, runtime_label="llama.cpp")
                for index in host_indices
            )
            return host_indices, tuple(index for index in local_indices if index is not None)

        if "llama_cpp" in sys.modules:
            logger.warning(
                "llama_cpp was imported before CUDA_VISIBLE_DEVICES could be narrowed; llama.cpp will use host GPU indices."
            )
            return host_indices, host_indices

        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in host_indices)
        logger.info("Planner narrowed llama.cpp CUDA visibility to GPUs: %s", os.environ["CUDA_VISIBLE_DEVICES"])
        return host_indices, tuple(range(len(host_indices)))

    def _llama_cuda_kwargs(self, param_names: set[str], *, endpoint: bool) -> dict[str, Any]:
        if self.settings.gpu_backend != GPUBackend.CUDA:
            return {}

        host_indices, local_indices = self._llama_cuda_device_selection(endpoint=endpoint)
        if not host_indices:
            return {}

        import llama_cpp as llama_cpp_module

        kwargs: dict[str, Any] = {}
        if "main_gpu" in param_names:
            kwargs["main_gpu"] = local_indices[0]

        if len(local_indices) == 1:
            split_none = getattr(llama_cpp_module, "LLAMA_SPLIT_MODE_NONE", None)
            if split_none is not None and "split_mode" in param_names:
                kwargs["split_mode"] = split_none
            return kwargs

        if "tensor_split" not in param_names:
            return kwargs

        gpu_free_by_index = {
            int(gpu["index"]): float(gpu.get("free_vram_gb", gpu.get("vram_gb", 0.0)) or 0.0)
            for gpu in self.gpu_info.get("nvidia_gpus", [])
            if gpu.get("index") is not None
        }
        total_free = sum(max(gpu_free_by_index.get(index, 0.0), 0.0) for index in host_indices)
        if total_free <= 0:
            return kwargs

        if local_indices == tuple(range(len(local_indices))):
            fractions = [max(gpu_free_by_index.get(index, 0.0), 0.0) / total_free for index in host_indices]
        else:
            fractions = [0.0] * (max(local_indices) + 1)
            for host_index, local_index in zip(host_indices, local_indices, strict=False):
                fractions[local_index] = max(gpu_free_by_index.get(host_index, 0.0), 0.0) / total_free
        kwargs["tensor_split"] = fractions

        split_layer = getattr(llama_cpp_module, "LLAMA_SPLIT_MODE_LAYER", None)
        if split_layer is not None and "split_mode" in param_names:
            kwargs["split_mode"] = split_layer

        return kwargs

    async def get_model(self, model_type: ModelType) -> Any:
        """
        Get a model, loading it if necessary.

        In sequential mode, this will unload other models first.
        """
        async with self._lock:
            if model_type in self._loaded_models:
                return self._loaded_models[model_type]

            # In sequential mode, unload other models first
            if self.is_sequential:
                await self._unload_for_request(model_type)

            # Load the requested model
            model = await self._load_model(model_type)
            self._loaded_models[model_type] = model
            self._persist_loaded_models()
            return model

    async def release_model(self, model_type: ModelType):
        """
        Release a model after use.

        In sequential mode, this triggers unloading.
        In parallel mode, models stay loaded.
        """
        if self.is_sequential:
            if model_type in self.resident_model_types:
                return
            async with self._lock:
                await self._unload_model(model_type)

    async def _load_model(self, model_type: ModelType) -> Any:
        """Load a specific model type."""
        logger.info(f"Loading model: {model_type.value}")

        if model_type == ModelType.WHISPER:
            return await self._load_whisper()
        elif model_type == ModelType.DIARIZATION:
            return await self._load_diarization()
        elif model_type == ModelType.VISION:
            return await self._load_vision()
        elif model_type == ModelType.SUMMARIZATION:
            return await self._load_summarization()
        elif model_type == ModelType.EMBEDDING:
            return await self._load_embedding()
        elif model_type == ModelType.AUDIO_EVENT:
            return await self._load_audio_event()

        raise ValueError(f"Unknown model type: {model_type}")

    async def _load_audio_event(self) -> Any:
        """Load an audio event classification model."""
        model_name = self.settings.audio_event_model
        self._require_cached_or_auto_download(model_name, "audio_event")

        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoProcessor, ClapModel
        import torch

        # CLAP on Apple Metal currently crashes inside MPS graph normalization,
        # so keep this optional stage on CPU there and let the main pipeline stay on Metal.
        if self.settings.gpu_backend == GPUBackend.METAL:
            device = "cpu"
            dtype = torch.float32
        else:
            device = self._torch_runtime_device(
                strict=self.requires_strict_accelerator,
                runtime_label="audio event classification",
            )
            # CLAP batch norm is not reliable in half precision on CUDA here.
            # Keep it in float32 and let this optional stage fail-soft only for other reasons.
            dtype = torch.float32 if _is_clap_model(model_name) else (_torch_dtype_for_backend(self.settings.gpu_backend) or torch.float32)
        cache_dir = self.settings.model_cache_dir
        loop = asyncio.get_event_loop()

        def load_audio_event():
            if _is_clap_model(model_name):
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir),
                )
                model = ClapModel.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir),
                    torch_dtype=dtype,
                )
                bundle_type = "audio_event_clap"
            else:
                processor = AutoFeatureExtractor.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir),
                )
                model = AutoModelForAudioClassification.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir),
                    torch_dtype=dtype,
                )
                bundle_type = "audio_event_classifier"
            if _is_accelerated_torch_device_name(device):
                model = model.to(torch.device(device))
            model.eval()
            return {
                "type": bundle_type,
                "model": model,
                "processor": processor,
                "device": device,
            }

        return await loop.run_in_executor(None, load_audio_event)

    async def _load_whisper(self) -> Any:
        """Load the transcription model (Whisper or Granite)."""
        model_name = self.settings.whisper_model
        resolved_name = _normalize_whisper_model_name(
            model_name,
            backend=self.settings.gpu_backend,
        )
        self._require_cached_or_auto_download(resolved_name, "asr")

        if _is_canary_model(model_name):
            from nemo.collections.speechlm2.models import SALM

            device = _torch_device(
                self.settings.gpu_backend,
                strict=self.requires_strict_accelerator and not self.settings.granite_force_cpu,
                runtime_label="Canary transcription",
                device_index=self._worker_cuda_local_device_index(runtime_label="Canary transcription"),
            )
            if self.settings.granite_force_cpu:
                device = "cpu"

            loop = asyncio.get_event_loop()

            def load_canary():
                model = SALM.from_pretrained(model_name)
                if _is_accelerated_torch_device_name(device):
                    import torch

                    model = model.to(torch.device(device))
                model.eval()
                return {
                    "type": "nemo_canary",
                    "model": model,
                    "device": device,
                }

            model_bundle = await loop.run_in_executor(None, load_canary)
            logger.info("Loaded Canary speech model: %s", model_name)
            return model_bundle

        if _is_granite_speech_model(model_name):
            from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
            from inspect import signature
            import torch

            device = _torch_device(
                self.settings.gpu_backend,
                strict=self.requires_strict_accelerator and not self.settings.granite_force_cpu,
                runtime_label="Granite transcription",
                device_index=self._worker_cuda_local_device_index(runtime_label="Granite transcription"),
            )
            if self.settings.granite_force_cpu:
                device = "cpu"
            dtype = _torch_dtype_for_backend(self.settings.gpu_backend) or (
                    torch.float16 if _is_accelerated_torch_device_name(device) else torch.float32
            )

            loop = asyncio.get_event_loop()

            def load_granite():
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=str(self.settings.model_cache_dir),
                )
                tokenizer = getattr(processor, "tokenizer", None)
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=str(self.settings.model_cache_dir),
                    )
                feature_extractor = AutoFeatureExtractor.from_pretrained(
                    model_name,
                    cache_dir=str(self.settings.model_cache_dir),
                )
                model_kwargs = {
                    "cache_dir": str(self.settings.model_cache_dir),
                }
                param_names = set(signature(AutoModelForSpeechSeq2Seq.from_pretrained).parameters.keys())
                if "dtype" in param_names:
                    model_kwargs["dtype"] = dtype
                else:
                    model_kwargs["torch_dtype"] = dtype
                attn_impl = _attn_implementation_for_backend(self.settings.gpu_backend)
                if attn_impl and "attn_implementation" in param_names:
                    model_kwargs["attn_implementation"] = attn_impl
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name,
                    **model_kwargs,
                )
                if _is_accelerated_torch_device_name(device):
                    model = model.to(torch.device(device))
                return {
                    "type": "granite",
                    "model": model,
                    "processor": processor,
                    "tokenizer": tokenizer,
                    "feature_extractor": feature_extractor,
                    "device": device,
                }

            model_bundle = await loop.run_in_executor(None, load_granite)
            logger.info(f"Loaded Granite speech model: {model_name}")
            return model_bundle

        if self.settings.gpu_backend in {GPUBackend.ROCM, GPUBackend.VULKAN, GPUBackend.METAL}:
            model = await _load_transformers_whisper(
                resolved_name,
                self.settings.gpu_backend,
                self.settings.model_cache_dir,
                strict_accelerator=self.requires_strict_accelerator,
                device_index=self._worker_cuda_local_device_index(runtime_label="transcription"),
            )
            logger.info("Loaded Whisper transformers model: %s", resolved_name)
            return model

        model = await _load_faster_whisper(
            resolved_name,
            self.settings.gpu_backend,
            self.settings.model_cache_dir,
            device_index=self._worker_cuda_local_device_index(runtime_label="transcription"),
        )
        logger.info("Loaded Whisper model: %s", resolved_name)
        return model

    async def _load_diarization(self) -> Any:
        """Load the pyannote diarization model."""
        model_name = self.settings.diarization_model
        self._require_cached_or_auto_download(model_name, "diarization")

        from pyannote.audio import Pipeline

        hf_token = self.settings.hf_token

        if not hf_token:
            raise ValueError("HF_TOKEN is required for pyannote models")

        loop = asyncio.get_event_loop()
        pipeline = await loop.run_in_executor(
            None,
            lambda: Pipeline.from_pretrained(
                model_name,
                token=hf_token,
            ),
        )

        # Move to GPU for CUDA/ROCm if available; Vulkan falls back to CPU.
        if self.settings.gpu_backend in {GPUBackend.CUDA, GPUBackend.ROCM, GPUBackend.METAL}:
            import torch
            device = self._torch_runtime_device(
                strict=self.requires_strict_accelerator,
                runtime_label="diarization",
            )
            if device != "cpu":
                try:
                    pipeline.to(torch.device(device))
                except Exception:
                    if self.requires_strict_accelerator:
                        raise
                    logger.warning("Diarization could not use %s; falling back to CPU.", device, exc_info=True)
            elif self.requires_strict_accelerator:
                raise RuntimeError(
                    "Diarization requires ROCm acceleration on Strix Halo, but no GPU device is available."
                )
        elif self.settings.gpu_backend == GPUBackend.VULKAN:
            logger.warning("Diarization running on CPU for %s backend.", self.settings.gpu_backend.value)

        logger.info(f"Loaded diarization model: {model_name}")
        return pipeline

    async def _load_vision(self) -> Any:
        """Load the vision model via llama.cpp."""
        self._llama_cuda_device_selection(endpoint=True)
        model_path = await self._ensure_local_gguf_model(
            self.settings.vision_model,
            "vision",
            "VISION_MODEL",
        )

        from inspect import signature
        from llama_cpp import Llama

        n_gpu_layers = _resolve_llama_gpu_layers(
            self.settings.gpu_backend,
            self.settings.vision_gpu_layers,
            rocm_safe_default=False,
        )

        loop = asyncio.get_event_loop()
        param_names = set(signature(Llama).parameters.keys())
        init_params = {
            "model_path": str(model_path),
            "n_ctx": 8192,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }
        init_params.update(self._llama_cuda_kwargs(param_names, endpoint=True))
        chat_handler = None
        if self.settings.vision_mmproj:
            mmproj_path = resolve_local_model_path(self.settings.vision_mmproj, self.settings.model_cache_dir)
            if mmproj_path is None:
                mmproj_name, mmproj_dir = model_download_target(
                    self.settings.vision_mmproj,
                    self.settings.model_cache_dir,
                )
                self._require_cached_or_auto_download(mmproj_name, "vision")
                try:
                    from app.core.model_downloader import download_model_async

                    mmproj_path = await download_model_async(
                        mmproj_name,
                        mmproj_dir,
                        component="vision",
                        backend=self.settings.gpu_backend,
                        env_var="VISION_MMPROJ",
                    )
                except Exception as exc:
                    logger.warning("Vision mmproj download failed: %s", exc)
            if mmproj_path is not None and not mmproj_path.exists():
                logger.warning("Vision mmproj path not found: %s", mmproj_path)

        if self.settings.vision_chat_format:
            chat_format = self.settings.vision_chat_format.lower()
            if chat_format in {"qwen3-vl", "qwen2.5-vl", "qwen25-vl"}:
                try:
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler

                    if self.settings.vision_mmproj and mmproj_path.exists():
                        chat_handler = Qwen25VLChatHandler(
                            clip_model_path=str(mmproj_path),
                            verbose=False,
                        )
                    else:
                        logger.warning("Vision chat handler requested but mmproj is missing.")
                except Exception as exc:
                    logger.warning("Failed to initialize Qwen VL chat handler: %s", exc)
            else:
                init_params["chat_format"] = self.settings.vision_chat_format

        if chat_handler is not None:
            init_params["chat_handler"] = chat_handler
        elif self.settings.vision_mmproj and "mmproj_path" in locals() and mmproj_path.exists():
            if "clip_model_path" in param_names:
                init_params["clip_model_path"] = str(mmproj_path)
            elif "mmproj" in param_names:
                init_params["mmproj"] = str(mmproj_path)
            elif "llava_projector_path" in param_names:
                init_params["llava_projector_path"] = str(mmproj_path)
            else:
                logger.warning(
                    "Vision mmproj configured but llama_cpp does not accept a projector parameter."
                )

        def _load_llama():
            return Llama(**init_params)

        try:
            model = await loop.run_in_executor(None, _load_llama)
        except Exception as exc:
            message = str(exc).lower()
            if "failed to load model from file" in message and model_path.exists():
                if not self.model_auto_download_enabled:
                    raise RuntimeError(
                        f"Model {self.settings.vision_model} failed to load from cache. "
                        "Download it in Settings or set MODEL_AUTO_DOWNLOAD=true."
                    ) from exc
                logger.warning(
                    "Vision model load failed, re-downloading %s and retrying.",
                    self.settings.vision_model,
                )
                try:
                    model_path.unlink()
                except Exception as unlink_exc:
                    logger.warning("Failed to remove corrupt vision model: %s", unlink_exc)
                from app.core.model_downloader import download_model_async

                model_name, target_dir = model_download_target(
                    self.settings.vision_model,
                    self.settings.model_cache_dir,
                )
                model_path = await download_model_async(
                    model_name,
                    target_dir,
                    component="vision",
                    backend=self.settings.gpu_backend,
                    env_var="VISION_MODEL",
                )
                init_params["model_path"] = str(model_path)
                model = await loop.run_in_executor(None, _load_llama)
            else:
                raise

        logger.info(f"Loaded vision model: {self.settings.vision_model}")
        return model

    async def _load_summarization(self) -> Any:
        """Load the summarization model via llama.cpp."""
        self._llama_cuda_device_selection(endpoint=True)
        model_path = await self._ensure_local_gguf_model(
            self.settings.summarization_model,
            "summarization",
            "SUMMARIZATION_MODEL",
        )

        from inspect import signature
        from llama_cpp import Llama

        n_gpu_layers = _resolve_llama_gpu_layers(
            self.settings.gpu_backend,
            self.settings.summarization_gpu_layers,
        )

        loop = asyncio.get_event_loop()
        param_names = set(signature(Llama).parameters.keys())
        init_params = {
            "model_path": str(model_path),
            "n_ctx": 32768,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }
        init_params.update(self._llama_cuda_kwargs(param_names, endpoint=True))
        model = await loop.run_in_executor(
            None,
            lambda: Llama(**init_params),
        )
        logger.info(f"Loaded summarization model: {self.settings.summarization_model}")
        return model

    async def _load_embedding(self) -> Any:
        """Load the embedding model."""
        model_name = self.settings.embedding_model
        self._require_cached_or_auto_download(model_name, "embedding")

        from sentence_transformers import SentenceTransformer

        device = self._torch_runtime_device(
            strict=self.requires_strict_accelerator,
            runtime_label="embedding",
        )

        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer(
                model_name,
                device=device,
                cache_folder=str(self.settings.model_cache_dir),
                trust_remote_code=_embedding_requires_trust_remote_code(model_name),
            ),
        )
        logger.info(f"Loaded embedding model: {model_name}")
        return model

    async def _unload_model(self, model_type: ModelType):
        """Unload a specific model to free memory."""
        if model_type not in self._loaded_models:
            return

        logger.info(f"Unloading model: {model_type.value}")
        model = self._loaded_models.pop(model_type)

        # Clear CUDA cache if applicable
        try:
            import torch
            if torch.cuda.is_available():
                _offload_model_to_cpu(model)
                del model
                gc.collect()
                if hasattr(torch.cuda, "synchronize"):
                    torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                _offload_model_to_cpu(model)
                del model
                gc.collect()
                torch.mps.empty_cache()
        except ImportError:
            del model

        # Force garbage collection
        gc.collect()
        self._persist_loaded_models()

    async def _unload_all(self):
        """Unload all loaded models."""
        for model_type in list(self._loaded_models.keys()):
            await self._unload_model(model_type)

    async def _unload_for_request(self, requested_type: ModelType) -> None:
        resident_models = self.resident_model_types
        for loaded_type in list(self._loaded_models.keys()):
            if loaded_type == requested_type:
                continue
            if loaded_type in resident_models:
                continue
            await self._unload_model(loaded_type)

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model types."""
        return [m.value for m in self._loaded_models.keys()]

    def _persist_loaded_models(self) -> None:
        """Persist loaded model types to a shared file for status reporting."""
        payload = {
            "models": [m.value for m in self._loaded_models.keys()],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._loaded_state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._loaded_state_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload), encoding="utf-8")
            tmp_path.replace(self._loaded_state_path)
        except Exception:
            logger.warning("Failed to persist loaded model state.", exc_info=True)


# Global model manager instance
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


async def reset_model_manager() -> None:
    """Drop the global model manager so new settings take effect immediately."""
    global _model_manager
    if _model_manager is not None:
        async with _model_manager._lock:
            await _model_manager._unload_all()
    _model_manager = None
