"""Runtime planning helpers for memory-aware model residency decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import ModelLoadingStrategy, Settings


WORKER_MODEL_ORDER = (
    "embedding",
    "whisper",
    "diarization",
    "audio_event",
    "vision",
    "summarization",
)


@dataclass(frozen=True)
class RuntimePlan:
    accelerator_count: int
    total_accelerator_memory_gb: float
    effective_memory_budget_gb: float
    placement_mode: str
    worker_model_loading: "ModelLoadingStrategy"
    keep_resident_models: tuple[str, ...]
    can_keep_all_worker_models_loaded: bool
    can_keep_endpoint_models_loaded: bool
    requires_endpoint_idle_teardown: bool
    endpoint_idle_timeout_recommendation_s: int
    estimated_memory_by_model_gb: dict[str, float]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["worker_model_loading"] = self.worker_model_loading.value
        payload["keep_resident_models"] = list(self.keep_resident_models)
        payload["notes"] = list(self.notes)
        return payload


def _round_gb(value: float) -> float:
    return round(max(float(value), 0.0), 2)


def _estimate_asr_memory_gb(model_name: str) -> float:
    lowered = model_name.lower()
    if "canary" in lowered:
        return 6.0
    if "granite-speech" in lowered:
        return 16.0
    if "large-v3-turbo" in lowered:
        return 3.0
    if "large-v3" in lowered or "whisper-large" in lowered:
        return 6.0
    if "medium" in lowered:
        return 1.5
    if "small" in lowered:
        return 1.0
    return 4.0


def _estimate_embedding_memory_gb(model_name: str) -> float:
    lowered = model_name.lower()
    if "qwen3-embedding-8b" in lowered:
        return 16.0
    if "nomic-embed" in lowered:
        return 0.6
    return 2.0


def _estimate_audio_event_memory_gb(model_name: str) -> float:
    lowered = model_name.lower()
    if "clap" in lowered:
        return 2.5
    if "ast-finetuned" in lowered:
        return 0.4
    return 1.0


def _estimate_endpoint_model_memory_gb(
    model_name: str,
    *,
    runtime_value: str,
) -> float:
    lowered = model_name.lower()
    if runtime_value == "vllm":
        if "30b" in lowered or "32b" in lowered:
            return 62.0
        if "70b" in lowered:
            return 110.0
        return 32.0

    if "120b" in lowered:
        return 72.0
    if "30b" in lowered or "32b" in lowered:
        if "q5" in lowered:
            return 28.0
        return 24.0
    if "8b" in lowered:
        return 8.0
    return 16.0


def estimate_model_memory_by_type_gb(settings: "Settings") -> dict[str, float]:
    return {
        "whisper": _estimate_asr_memory_gb(settings.whisper_model),
        "diarization": 2.0,
        "embedding": _estimate_embedding_memory_gb(settings.embedding_model),
        "audio_event": _estimate_audio_event_memory_gb(settings.audio_event_model),
        "vision": _estimate_endpoint_model_memory_gb(
            settings.vision_endpoint_model or settings.vision_model,
            runtime_value=getattr(settings.rocm_llm_runtime, "value", settings.rocm_llm_runtime),
        ),
        "summarization": _estimate_endpoint_model_memory_gb(
            settings.summarization_endpoint_model or settings.summarization_model,
            runtime_value=getattr(settings.rocm_llm_runtime, "value", settings.rocm_llm_runtime),
        ),
    }


def _resolve_total_accelerator_memory_gb(settings: "Settings", gpu_info: dict) -> float:
    if getattr(settings.hardware_profile, "value", settings.hardware_profile) == "strix_halo":
        unified = gpu_info.get("unified_memory_gb")
        if unified:
            return _round_gb(unified)
    return _round_gb(gpu_info.get("total_vram_gb", 0.0))


def _resolve_effective_budget_gb(settings: "Settings", total_memory_gb: float, notes: list[str]) -> float:
    if total_memory_gb <= 0:
        return 0.0

    if getattr(settings, "runtime_memory_ceiling_gb", None):
        requested = float(settings.runtime_memory_ceiling_gb or 0.0)
        if requested > total_memory_gb:
            notes.append(
                f"Runtime memory ceiling {requested:.1f} GB exceeds detected accelerator memory; clamping to {total_memory_gb:.1f} GB."
            )
        return _round_gb(min(requested, total_memory_gb))

    return _round_gb(total_memory_gb * float(settings.gpu_memory_fraction))


def _placement_mode(settings: "Settings", gpu_info: dict) -> str:
    backend_value = getattr(settings.gpu_backend, "value", settings.gpu_backend)
    if backend_value == "cpu":
        return "cpu_only"
    if getattr(settings.hardware_profile, "value", settings.hardware_profile) == "strix_halo":
        return "unified_memory_apu"
    nvidia_count = len(gpu_info.get("nvidia_gpus", []))
    amd_count = len(gpu_info.get("amd_gpus", []))
    if nvidia_count + amd_count > 1:
        return "multi_gpu"
    return "single_gpu"


def _uses_external_endpoints(settings: "Settings") -> bool:
    return bool(settings.vision_endpoint_url.strip() or settings.summarization_endpoint_url.strip())


def _select_resident_models(
    estimates: dict[str, float],
    *,
    budget_gb: float,
) -> tuple[str, ...]:
    selected: list[str] = []
    consumed = 0.0
    for key in WORKER_MODEL_ORDER:
        estimate = estimates.get(key, 0.0)
        if estimate <= 0:
            continue
        if consumed + estimate <= budget_gb:
            selected.append(key)
            consumed += estimate
    return tuple(selected)


def build_runtime_plan(settings: "Settings", gpu_info: dict) -> RuntimePlan:
    from app.config import ModelLoadingStrategy

    notes: list[str] = []
    estimates = estimate_model_memory_by_type_gb(settings)
    total_memory_gb = _resolve_total_accelerator_memory_gb(settings, gpu_info)
    budget_gb = _resolve_effective_budget_gb(settings, total_memory_gb, notes)
    placement = _placement_mode(settings, gpu_info)
    accelerator_count = len(gpu_info.get("nvidia_gpus", [])) + len(gpu_info.get("amd_gpus", []))

    endpoint_models = []
    worker_models = ["whisper", "diarization", "embedding", "audio_event", "vision", "summarization"]
    if _uses_external_endpoints(settings):
        endpoint_models = ["vision", "summarization"]
        worker_models = ["whisper", "diarization", "embedding", "audio_event"]

    endpoint_total = sum(estimates[key] for key in endpoint_models)
    max_endpoint = max((estimates[key] for key in endpoint_models), default=0.0)
    worker_total = sum(estimates[key] for key in worker_models)

    requires_idle_teardown = (
        getattr(settings.hardware_profile, "value", settings.hardware_profile) == "strix_halo"
        and getattr(settings.gpu_backend, "value", settings.gpu_backend) == "rocm"
        and getattr(settings.rocm_llm_runtime, "value", settings.rocm_llm_runtime) == "llama_server"
        and bool(endpoint_models)
    )

    can_keep_endpoint_models_loaded = False
    endpoint_reservation_gb = max_endpoint
    if endpoint_models:
        if requires_idle_teardown:
            notes.append(
                "ROCm llama_server endpoints stay on managed idle teardown on Strix Halo to avoid the post-job busy-state bug."
            )
        else:
            can_keep_endpoint_models_loaded = endpoint_total <= budget_gb
            endpoint_reservation_gb = endpoint_total if can_keep_endpoint_models_loaded else max_endpoint
            if not can_keep_endpoint_models_loaded:
                notes.append(
                    "Endpoint models exceed the active memory budget when kept hot together; plan reserves only one endpoint at a time."
                )

    if not endpoint_models:
        endpoint_reservation_gb = 0.0

    reserve_for_ephemeral_worker = 0.0
    available_for_worker_gb = max(budget_gb - endpoint_reservation_gb, 0.0)
    can_keep_all_worker_models_loaded = worker_total <= available_for_worker_gb
    worker_model_loading = (
        ModelLoadingStrategy.PARALLEL if can_keep_all_worker_models_loaded else ModelLoadingStrategy.SEQUENTIAL
    )

    if can_keep_all_worker_models_loaded:
        keep_resident_models = tuple(worker_models)
    else:
        reserve_for_ephemeral_worker = max((estimates[key] for key in worker_models), default=0.0)
        resident_budget = max(available_for_worker_gb - reserve_for_ephemeral_worker, 0.0)
        keep_resident_models = _select_resident_models(estimates, budget_gb=resident_budget)
        if keep_resident_models:
            notes.append(
                "Planner selected a hybrid resident set for worker-side models; larger models still load on demand."
            )

    if budget_gb <= 0:
        worker_model_loading = ModelLoadingStrategy.SEQUENTIAL
        keep_resident_models = ()
        notes.append("Detected accelerator memory is unavailable; planner falls back to sequential loading.")

    if placement == "multi_gpu":
        notes.append(
            "Multiple GPUs were detected. The planner records multi-GPU placement, but model splitting remains backend-specific work."
        )

    if settings.runtime_planner_enabled and can_keep_all_worker_models_loaded:
        notes.append("Worker-side models fit within the active memory budget and can stay resident.")

    endpoint_idle_timeout_recommendation_s = 120 if requires_idle_teardown else settings.rocm_llm_idle_timeout_s

    return RuntimePlan(
        accelerator_count=accelerator_count,
        total_accelerator_memory_gb=total_memory_gb,
        effective_memory_budget_gb=budget_gb,
        placement_mode=placement,
        worker_model_loading=worker_model_loading,
        keep_resident_models=keep_resident_models,
        can_keep_all_worker_models_loaded=can_keep_all_worker_models_loaded,
        can_keep_endpoint_models_loaded=can_keep_endpoint_models_loaded,
        requires_endpoint_idle_teardown=requires_idle_teardown,
        endpoint_idle_timeout_recommendation_s=endpoint_idle_timeout_recommendation_s,
        estimated_memory_by_model_gb={
            key: _round_gb(value) for key, value in estimates.items()
        },
        notes=tuple(notes),
    )
