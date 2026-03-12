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
    available_accelerator_memory_gb: float
    effective_memory_budget_gb: float
    placement_mode: str
    worker_model_loading: "ModelLoadingStrategy"
    keep_resident_models: tuple[str, ...]
    preferred_worker_device_indices: tuple[int, ...]
    preferred_worker_devices: tuple[str, ...]
    preferred_endpoint_device_indices: tuple[int, ...]
    preferred_endpoint_devices: tuple[str, ...]
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
        unified = gpu_info.get("unified_memory_gb") or gpu_info.get("system_memory_gb")
        if unified:
            return _round_gb(unified)
    return _round_gb(gpu_info.get("total_vram_gb", 0.0))


def _resolve_available_accelerator_memory_gb(settings: "Settings", gpu_info: dict) -> float:
    if getattr(settings.hardware_profile, "value", settings.hardware_profile) == "strix_halo":
        unified = gpu_info.get("unified_memory_gb") or gpu_info.get("system_memory_gb")
        if unified:
            return _round_gb(unified)
    return _round_gb(gpu_info.get("available_vram_gb", gpu_info.get("total_vram_gb", 0.0)))


def _resolve_effective_budget_gb(
    settings: "Settings",
    total_memory_gb: float,
    available_memory_gb: float,
    notes: list[str],
) -> float:
    base_memory_gb = available_memory_gb or total_memory_gb
    if base_memory_gb <= 0:
        return 0.0

    if getattr(settings, "runtime_memory_ceiling_gb", None):
        requested = float(settings.runtime_memory_ceiling_gb or 0.0)
        if requested > base_memory_gb:
            notes.append(
                f"Runtime memory ceiling {requested:.1f} GB exceeds currently available accelerator memory; clamping to {base_memory_gb:.1f} GB."
            )
        return _round_gb(min(requested, base_memory_gb))

    return _round_gb(base_memory_gb * float(settings.gpu_memory_fraction))


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


def _device_label(gpu: dict) -> str:
    index = gpu.get("index")
    name = gpu.get("name", "GPU")
    free = gpu.get("free_vram_gb")
    if index is None:
        return str(name)
    if free is None:
        return f"GPU{index} {name}"
    return f"GPU{index} {name} ({_round_gb(free)} GB free)"


def _free_capacity_gb(gpu: dict) -> float:
    return float(gpu.get("free_vram_gb", gpu.get("vram_gb", 0.0)) or 0.0)


def _pick_gpu_group(
    nvidia_gpus: list[dict],
    *,
    requirement_gb: float,
    prefer_nvlink: bool,
    topology: dict,
) -> tuple[dict, ...]:
    if not nvidia_gpus or requirement_gb <= 0:
        return ()

    sorted_gpus = sorted(nvidia_gpus, key=_free_capacity_gb, reverse=True)
    best = sorted_gpus[0]
    if _free_capacity_gb(best) >= requirement_gb:
        return (best,)

    if prefer_nvlink:
        for group in topology.get("nvlink_groups", []):
            candidates = [gpu for gpu in nvidia_gpus if gpu.get("index") in set(group)]
            if sum(_free_capacity_gb(gpu) for gpu in candidates) >= requirement_gb:
                return tuple(sorted(candidates, key=lambda item: item["index"]))

    selected: list[dict] = []
    remaining = requirement_gb
    for gpu in sorted_gpus:
        selected.append(gpu)
        remaining -= _free_capacity_gb(gpu)
        if remaining <= 0:
            break
    return tuple(selected)


def _device_indices(gpus: tuple[dict, ...]) -> tuple[int, ...]:
    return tuple(int(gpu["index"]) for gpu in gpus if gpu.get("index") is not None)


def _device_labels(gpus: tuple[dict, ...]) -> tuple[str, ...]:
    return tuple(_device_label(gpu) for gpu in gpus)


def build_runtime_plan(settings: "Settings", gpu_info: dict) -> RuntimePlan:
    from app.config import ModelLoadingStrategy

    notes: list[str] = []
    estimates = estimate_model_memory_by_type_gb(settings)
    total_memory_gb = _resolve_total_accelerator_memory_gb(settings, gpu_info)
    available_memory_gb = _resolve_available_accelerator_memory_gb(settings, gpu_info)
    budget_gb = _resolve_effective_budget_gb(settings, total_memory_gb, available_memory_gb, notes)
    placement = _placement_mode(settings, gpu_info)
    accelerator_count = len(gpu_info.get("nvidia_gpus", [])) + len(gpu_info.get("amd_gpus", []))
    if accelerator_count == 0 and getattr(settings.hardware_profile, "value", settings.hardware_profile) == "strix_halo":
        accelerator_count = 1
        if gpu_info.get("system_memory_gb") and not gpu_info.get("unified_memory_gb"):
            notes.append(
                "Planner inferred Strix Halo unified memory from host RAM because direct ROCm memory telemetry was unavailable."
            )

    endpoint_models = []
    worker_models = ["whisper", "diarization", "embedding", "audio_event", "vision", "summarization"]
    if _uses_external_endpoints(settings):
        endpoint_models = ["vision", "summarization"]
        worker_models = ["whisper", "diarization", "embedding", "audio_event"]

    endpoint_total = sum(estimates[key] for key in endpoint_models)
    max_endpoint = max((estimates[key] for key in endpoint_models), default=0.0)
    worker_total = sum(estimates[key] for key in worker_models)
    total_hot_set = worker_total + endpoint_total
    nvidia_gpus = list(gpu_info.get("nvidia_gpus", []))
    topology = gpu_info.get("nvidia_topology", {})

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

    preferred_worker_indices: tuple[int, ...] = ()
    preferred_worker_devices: tuple[str, ...] = ()
    preferred_endpoint_indices: tuple[int, ...] = ()
    preferred_endpoint_devices: tuple[str, ...] = ()
    if placement == "multi_gpu" and nvidia_gpus:
        if len({round(float(gpu.get("vram_gb", 0.0)), 1) for gpu in nvidia_gpus}) > 1:
            notes.append("Heterogeneous NVIDIA VRAM sizes detected; planner prefers the largest currently free GPU before spreading work.")

        sorted_gpus = sorted(nvidia_gpus, key=_free_capacity_gb, reverse=True)
        best_gpu = sorted_gpus[0]
        best_gpu_free = _free_capacity_gb(best_gpu)
        if best_gpu_free >= total_hot_set:
            placement = "single_large_gpu_preferred"
            preferred_worker_indices = _device_indices((best_gpu,))
            preferred_worker_devices = _device_labels((best_gpu,))
            preferred_endpoint_indices = _device_indices((best_gpu,))
            preferred_endpoint_devices = _device_labels((best_gpu,))
            notes.append(
                f"{_device_label(best_gpu)} can host the full active model set at current free memory."
            )
        else:
            preferred_worker_group = _pick_gpu_group(
                nvidia_gpus,
                requirement_gb=worker_total,
                prefer_nvlink=False,
                topology=topology,
            )
            preferred_worker_indices = _device_indices(preferred_worker_group)
            preferred_worker_devices = _device_labels(preferred_worker_group)
            endpoint_requirement = endpoint_total if can_keep_endpoint_models_loaded else max_endpoint
            preferred_endpoint_group = _pick_gpu_group(
                nvidia_gpus,
                requirement_gb=endpoint_requirement,
                prefer_nvlink=True,
                topology=topology,
            )
            preferred_endpoint_indices = _device_indices(preferred_endpoint_group)
            preferred_endpoint_devices = _device_labels(preferred_endpoint_group)
            if preferred_worker_devices:
                notes.append(f"Preferred worker placement: {', '.join(preferred_worker_devices)}.")
            if preferred_endpoint_devices:
                notes.append(f"Preferred endpoint placement: {', '.join(preferred_endpoint_devices)}.")
            if topology.get("nvlink_groups"):
                notes.append(
                    "Detected NVLink-connected 3090 pairs; use them first when a fallback multi-GPU endpoint placement is needed."
                )

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
        available_accelerator_memory_gb=available_memory_gb,
        effective_memory_budget_gb=budget_gb,
        placement_mode=placement,
        worker_model_loading=worker_model_loading,
        keep_resident_models=keep_resident_models,
        preferred_worker_device_indices=preferred_worker_indices,
        preferred_worker_devices=preferred_worker_devices,
        preferred_endpoint_device_indices=preferred_endpoint_indices,
        preferred_endpoint_devices=preferred_endpoint_devices,
        can_keep_all_worker_models_loaded=can_keep_all_worker_models_loaded,
        can_keep_endpoint_models_loaded=can_keep_endpoint_models_loaded,
        requires_endpoint_idle_teardown=requires_idle_teardown,
        endpoint_idle_timeout_recommendation_s=endpoint_idle_timeout_recommendation_s,
        estimated_memory_by_model_gb={
            key: _round_gb(value) for key, value in estimates.items()
        },
        notes=tuple(notes),
    )
