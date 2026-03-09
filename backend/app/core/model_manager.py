"""Model manager for loading/unloading models based on hardware profile."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from app.config import GPUBackend, ModelLoadingStrategy, get_settings

logger = logging.getLogger(__name__)


def _is_granite_speech_model(model_name: str) -> bool:
    """Heuristic to detect Granite speech models."""
    return "granite-speech" in model_name.lower()


def _is_canary_model(model_name: str) -> bool:
    """Heuristic to detect NVIDIA Canary models."""
    lowered = model_name.lower()
    return "canary" in lowered or "nvidia/canary" in lowered


def _normalize_whisper_model_name(model_name: str) -> str:
    """Normalize common Whisper aliases to Hugging Face model IDs."""
    lower = model_name.lower()
    if lower in {"large-v3-turbo", "whisper-large-v3-turbo"}:
        return "openai/whisper-large-v3-turbo"
    if lower in {"large-v3", "whisper-large-v3"}:
        return "openai/whisper-large-v3"
    if lower == "whisper-large":
        return "openai/whisper-large"
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
    return torch.float16 if backend == GPUBackend.CUDA else torch.float32


def _torch_device(backend: GPUBackend) -> str:
    """Map backend choice to torch device name with safe fallback."""
    if backend.value in {"cuda", "rocm"}:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            logger.warning("Torch CUDA/ROCm not available, falling back to CPU.")
            return "cpu"
        logger.warning("GPU backend requested but no CUDA/ROCm device found, using CPU.")
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
    """Resolve llama.cpp GPU layer count with ROCm-safe defaults."""
    if configured_layers is not None:
        return configured_layers
    if backend == GPUBackend.CPU:
        return 0
    if backend == GPUBackend.ROCM and rocm_safe_default:
        return 0
    return -1


async def _load_faster_whisper(
    model_name: str,
    backend: GPUBackend,
    cache_dir: Path,
) -> Any:
    """Load a faster-whisper model by name."""
    from faster_whisper import WhisperModel

    device = _faster_whisper_device(backend)
    compute_type = _faster_whisper_compute_type(backend)
    if backend in {GPUBackend.ROCM, GPUBackend.VULKAN} and device == "cpu":
        logger.warning(
            "faster-whisper GPU acceleration is CUDA-only; using CPU for %s backend.",
            backend.value,
        )

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(cache_dir),
        ),
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
) -> Any:
    """Load a Whisper model via transformers for ROCm/CPU compatibility."""
    from inspect import signature
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch

    device = _torch_device(backend)
    dtype = _torch_dtype_for_backend(backend) or (torch.float16 if device == "cuda" else torch.float32)
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
        if device == "cuda":
            model = model.to(torch.device("cuda"))
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
        self._loaded_models: dict[ModelType, Any] = {}
        self._lock = asyncio.Lock()
        self._loaded_state_path = self.settings.model_cache_dir / ".loaded_models.json"
        self._persist_loaded_models()

    @property
    def is_sequential(self) -> bool:
        """Check if we're in sequential loading mode."""
        return self.settings.model_loading == ModelLoadingStrategy.SEQUENTIAL

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
                await self._unload_all()

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
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        import torch

        model_name = self.settings.audio_event_model
        device = _torch_device(self.settings.gpu_backend)
        dtype = _torch_dtype_for_backend(self.settings.gpu_backend) or torch.float32
        cache_dir = self.settings.model_cache_dir
        loop = asyncio.get_event_loop()

        def load_audio_event():
            processor = AutoFeatureExtractor.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
            )
            model = AutoModelForAudioClassification.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                torch_dtype=dtype,
            )
            if device == "cuda":
                model = model.to(torch.device("cuda"))
            model.eval()
            return {
                "type": "audio_event",
                "model": model,
                "processor": processor,
                "device": device,
            }

        return await loop.run_in_executor(None, load_audio_event)

    async def _load_whisper(self) -> Any:
        """Load the transcription model (Whisper or Granite)."""
        model_name = self.settings.whisper_model
        normalized_name = _normalize_whisper_model_name(model_name)

        if _is_canary_model(model_name):
            from nemo.collections.speechlm2.models import SALM
            import torch

            device = _torch_device(self.settings.gpu_backend)
            if self.settings.granite_force_cpu:
                device = "cpu"

            loop = asyncio.get_event_loop()

            def load_canary():
                model = SALM.from_pretrained(model_name)
                if device == "cuda" and torch.cuda.is_available():
                    model = model.cuda()
                model.eval()
                return {
                    "type": "nemo_canary",
                    "model": model,
                    "device": device,
                }

            try:
                model_bundle = await loop.run_in_executor(None, load_canary)
                logger.info("Loaded Canary speech model: %s", model_name)
                return model_bundle
            except Exception:
                if self.settings.transcription_fallback_enabled:
                    fallback = self.settings.transcription_fallback_model
                    logger.exception(
                        "Canary load failed, falling back to Whisper model: %s",
                        fallback,
                    )
                    fallback_name = _normalize_whisper_model_name(fallback)
                    if self.settings.gpu_backend in {
                        GPUBackend.ROCM,
                        GPUBackend.VULKAN,
                    }:
                        model = await _load_transformers_whisper(
                            fallback_name,
                            self.settings.gpu_backend,
                            self.settings.model_cache_dir,
                        )
                    else:
                        model = await _load_faster_whisper(
                            fallback_name,
                            self.settings.gpu_backend,
                            self.settings.model_cache_dir,
                        )
                    logger.info("Loaded Whisper fallback model: %s", fallback)
                    return model
                raise

        if _is_granite_speech_model(model_name):
            from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
            from inspect import signature
            import torch

            device = _torch_device(self.settings.gpu_backend)
            if self.settings.granite_force_cpu:
                device = "cpu"
            dtype = _torch_dtype_for_backend(self.settings.gpu_backend) or (
                torch.float16 if device == "cuda" else torch.float32
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
                if device == "cuda":
                    model = model.to(torch.device("cuda"))
                return {
                    "type": "granite",
                    "model": model,
                    "processor": processor,
                    "tokenizer": tokenizer,
                    "feature_extractor": feature_extractor,
                    "device": device,
                }

            try:
                model_bundle = await loop.run_in_executor(None, load_granite)
                logger.info(f"Loaded Granite speech model: {model_name}")
                return model_bundle
            except Exception:
                if self.settings.transcription_fallback_enabled:
                    fallback = self.settings.transcription_fallback_model
                    logger.exception(
                        "Granite load failed, falling back to Whisper model: %s",
                        fallback,
                    )
                    fallback_name = _normalize_whisper_model_name(fallback)
                    if self.settings.gpu_backend in {
                        GPUBackend.ROCM,
                        GPUBackend.VULKAN,
                    }:
                        model = await _load_transformers_whisper(
                            fallback_name,
                            self.settings.gpu_backend,
                            self.settings.model_cache_dir,
                        )
                    else:
                        model = await _load_faster_whisper(
                            fallback_name,
                            self.settings.gpu_backend,
                            self.settings.model_cache_dir,
                        )
                    logger.info("Loaded Whisper fallback model: %s", fallback)
                    return model
                raise

        if self.settings.gpu_backend in {GPUBackend.ROCM, GPUBackend.VULKAN}:
            model = await _load_transformers_whisper(
                normalized_name,
                self.settings.gpu_backend,
                self.settings.model_cache_dir,
            )
            logger.info("Loaded Whisper transformers model: %s", normalized_name)
            return model

        model = await _load_faster_whisper(
            normalized_name,
            self.settings.gpu_backend,
            self.settings.model_cache_dir,
        )
        logger.info("Loaded Whisper model: %s", normalized_name)
        return model

    async def _load_diarization(self) -> Any:
        """Load the pyannote diarization model."""
        from pyannote.audio import Pipeline

        model_name = self.settings.diarization_model
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
        if self.settings.gpu_backend in {GPUBackend.CUDA, GPUBackend.ROCM}:
            import torch
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        elif self.settings.gpu_backend == GPUBackend.VULKAN:
            logger.warning("Diarization running on CPU for %s backend.", self.settings.gpu_backend.value)

        logger.info(f"Loaded diarization model: {model_name}")
        return pipeline

    async def _load_vision(self) -> Any:
        """Load the vision model via llama.cpp."""
        from inspect import signature
        from llama_cpp import Llama
        from app.core.model_downloader import download_model_async

        model_path = self.settings.model_cache_dir / self.settings.vision_model

        if not model_path.exists():
            logger.info(f"Vision model not found locally, downloading...")
            model_path = await download_model_async(
                self.settings.vision_model,
                self.settings.model_cache_dir
            )

        # Default local vision to CPU on ROCm unless explicitly overridden.
        n_gpu_layers = _resolve_llama_gpu_layers(
            self.settings.gpu_backend,
            self.settings.vision_gpu_layers,
            rocm_safe_default=True,
        )
        if self.settings.gpu_backend == GPUBackend.ROCM and self.settings.vision_gpu_layers is None:
            logger.warning(
                "ROCm vision model loads defaulting to CPU layers for stability. "
                "Set VISION_GPU_LAYERS to override."
            )

        loop = asyncio.get_event_loop()
        init_params = {
            "model_path": str(model_path),
            "n_ctx": 8192,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }
        chat_handler = None
        if self.settings.vision_mmproj:
            mmproj_path = Path(self.settings.vision_mmproj)
            if not mmproj_path.is_absolute():
                mmproj_path = self.settings.model_cache_dir / mmproj_path
            if not mmproj_path.exists():
                try:
                    mmproj_name = Path(self.settings.vision_mmproj).name
                    mmproj_path = await download_model_async(
                        mmproj_name,
                        self.settings.model_cache_dir,
                    )
                except Exception as exc:
                    logger.warning("Vision mmproj download failed: %s", exc)
            if not mmproj_path.exists():
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
            param_names = set(signature(Llama).parameters.keys())
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
                logger.warning(
                    "Vision model load failed, re-downloading %s and retrying.",
                    self.settings.vision_model,
                )
                try:
                    model_path.unlink()
                except Exception as unlink_exc:
                    logger.warning("Failed to remove corrupt vision model: %s", unlink_exc)
                model_path = await download_model_async(
                    self.settings.vision_model,
                    self.settings.model_cache_dir,
                )
                init_params["model_path"] = str(model_path)
                model = await loop.run_in_executor(None, _load_llama)
            else:
                raise

        logger.info(f"Loaded vision model: {self.settings.vision_model}")
        return model

    async def _load_summarization(self) -> Any:
        """Load the summarization model via llama.cpp."""
        from llama_cpp import Llama
        from app.core.model_downloader import download_model_async

        model_path = self.settings.model_cache_dir / self.settings.summarization_model

        if not model_path.exists():
            logger.info(f"Summarization model not found locally, downloading...")
            model_path = await download_model_async(
                self.settings.summarization_model,
                self.settings.model_cache_dir
            )

        n_gpu_layers = _resolve_llama_gpu_layers(
            self.settings.gpu_backend,
            self.settings.summarization_gpu_layers,
        )

        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            None,
            lambda: Llama(
                model_path=str(model_path),
                n_ctx=32768,  # Large context for long transcripts
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            ),
        )
        logger.info(f"Loaded summarization model: {self.settings.summarization_model}")
        return model

    async def _load_embedding(self) -> Any:
        """Load the embedding model."""
        from sentence_transformers import SentenceTransformer

        model_name = self.settings.embedding_model
        if self.settings.gpu_backend in {GPUBackend.CUDA, GPUBackend.ROCM}:
            device = "cuda"
        else:
            device = "cpu"

        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer(
                model_name,
                device=device,
                cache_folder=str(self.settings.model_cache_dir),
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
                del model
                torch.cuda.empty_cache()
        except ImportError:
            del model

        # Force garbage collection
        import gc
        gc.collect()
        self._persist_loaded_models()

    async def _unload_all(self):
        """Unload all loaded models."""
        for model_type in list(self._loaded_models.keys()):
            await self._unload_model(model_type)

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
