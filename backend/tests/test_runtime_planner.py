import types

from app.config import GPUBackend, HardwareProfile, ModelLoadingStrategy, ROCmLLMRuntime
from app.runtime.planner import build_runtime_plan


def _settings(**overrides):
    base = dict(
        hardware_profile=HardwareProfile.STRIX_HALO,
        gpu_backend=GPUBackend.ROCM,
        gpu_memory_fraction=0.75,
        runtime_planner_enabled=True,
        runtime_memory_ceiling_gb=None,
        rocm_llm_runtime=ROCmLLMRuntime.LLAMA_SERVER,
        rocm_llm_idle_timeout_s=120,
        whisper_model="nvidia/canary-qwen-2.5b",
        diarization_model="pyannote/speaker-diarization-community-1",
        embedding_model="Qwen/Qwen3-Embedding-8B",
        audio_event_model="laion/clap-htsat-fused",
        vision_model="Qwen3VL-32B-Instruct-Q4_K_M.gguf",
        vision_endpoint_model="Qwen3VL-32B-Instruct-Q4_K_M.gguf",
        vision_endpoint_url="http://vision",
        summarization_model="qwen3-30b-a3b-q4_k_m.gguf",
        summarization_endpoint_model="qwen3-30b-a3b-q4_k_m.gguf",
        summarization_endpoint_url="http://summary",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _gpu_info(**overrides):
    base = {
        "nvidia_gpus": [],
        "amd_gpus": [{"name": "AMD GPU", "vram_gb": 16.0}],
        "unified_memory_gb": 128.0,
        "total_vram_gb": 16.0,
    }
    base.update(overrides)
    return base


def test_runtime_plan_keeps_worker_models_hot_on_strix_halo_when_budget_allows():
    plan = build_runtime_plan(_settings(), _gpu_info())

    assert plan.worker_model_loading == ModelLoadingStrategy.PARALLEL
    assert plan.worker_execution_mode == "resident_all"
    assert plan.endpoint_model_loading == "sequential"
    assert plan.can_keep_all_worker_models_loaded is True
    assert set(plan.keep_resident_models) == {"whisper", "diarization", "embedding", "audio_event"}
    assert plan.available_accelerator_memory_gb == 128.0
    assert plan.requires_endpoint_idle_teardown is True
    assert plan.can_keep_endpoint_models_loaded is False


def test_runtime_plan_honors_explicit_memory_ceiling():
    plan = build_runtime_plan(
        _settings(runtime_memory_ceiling_gb=20.0),
        _gpu_info(),
    )

    assert plan.effective_memory_budget_gb == 20.0
    assert plan.worker_model_loading == ModelLoadingStrategy.SEQUENTIAL
    assert plan.worker_execution_mode == "stage_by_stage"
    assert "embedding" not in plan.keep_resident_models


def test_runtime_plan_marks_multi_gpu_placement():
    plan = build_runtime_plan(
        _settings(
            hardware_profile=HardwareProfile.MULTI_GPU,
            gpu_backend=GPUBackend.CUDA,
            rocm_llm_runtime=ROCmLLMRuntime.VLLM,
            vision_endpoint_url="",
            summarization_endpoint_url="",
        ),
        _gpu_info(
            nvidia_gpus=[
                {"name": "RTX 6000", "vram_gb": 24.0},
                {"name": "RTX 6000", "vram_gb": 24.0},
            ],
            amd_gpus=[],
            unified_memory_gb=None,
            total_vram_gb=48.0,
        ),
    )

    assert plan.placement_mode == "multi_gpu"
    assert plan.endpoint_model_loading == "none"
    assert any("Multiple GPUs were detected" in note for note in plan.notes)


def test_runtime_plan_prefers_single_large_gpu_when_it_fits_current_free_memory():
    plan = build_runtime_plan(
        _settings(
            hardware_profile=HardwareProfile.MULTI_GPU,
            gpu_backend=GPUBackend.CUDA,
            rocm_llm_runtime=ROCmLLMRuntime.LLAMA_SERVER,
            vision_endpoint_url="",
            summarization_endpoint_url="",
        ),
        _gpu_info(
            nvidia_gpus=[
                {"index": 0, "name": "RTX 3090", "vram_gb": 24.0, "free_vram_gb": 24.0},
                {"index": 1, "name": "RTX 3090", "vram_gb": 24.0, "free_vram_gb": 24.0},
                {"index": 5, "name": "RTX PRO 6000 Blackwell", "vram_gb": 97.0, "free_vram_gb": 97.0},
            ],
            amd_gpus=[],
            unified_memory_gb=None,
            total_vram_gb=145.0,
            available_vram_gb=145.0,
            nvidia_topology={"connections": {}, "nvlink_groups": [[0, 1]]},
        ),
    )

    assert plan.placement_mode == "single_large_gpu_preferred"
    assert plan.endpoint_model_loading == "none"
    assert plan.preferred_worker_device_indices == (5,)
    assert plan.preferred_worker_devices == ("GPU5 RTX PRO 6000 Blackwell (97.0 GB free)",)
    assert plan.preferred_endpoint_device_indices == (5,)
    assert plan.preferred_endpoint_devices == ("GPU5 RTX PRO 6000 Blackwell (97.0 GB free)",)
    assert plan.preferred_model_devices["worker"] == ("GPU5 RTX PRO 6000 Blackwell (97.0 GB free)",)
    assert any("can host the full active model set" in note for note in plan.notes)


def test_runtime_plan_falls_back_to_multi_gpu_groups_when_large_gpu_cannot_fit_hot_set():
    plan = build_runtime_plan(
        _settings(
            hardware_profile=HardwareProfile.MULTI_GPU,
            gpu_backend=GPUBackend.CUDA,
            rocm_llm_runtime=ROCmLLMRuntime.LLAMA_SERVER,
            vision_endpoint_url="http://vision",
            summarization_endpoint_url="http://summary",
        ),
        _gpu_info(
            nvidia_gpus=[
                {"index": 0, "name": "RTX 3090", "vram_gb": 24.0, "free_vram_gb": 24.0},
                {"index": 1, "name": "RTX 3090", "vram_gb": 24.0, "free_vram_gb": 24.0},
                {"index": 5, "name": "RTX PRO 6000 Blackwell", "vram_gb": 97.0, "free_vram_gb": 40.0},
            ],
            amd_gpus=[],
            unified_memory_gb=None,
            total_vram_gb=145.0,
            available_vram_gb=88.0,
            nvidia_topology={"connections": {}, "nvlink_groups": [[0, 1]]},
        ),
    )

    assert plan.placement_mode == "multi_gpu"
    assert plan.endpoint_model_loading == "parallel"
    assert plan.preferred_worker_device_indices == (1,)
    assert plan.preferred_worker_devices == ("GPU1 RTX 3090 (24.0 GB free)",)
    assert plan.preferred_endpoint_device_indices == (5, 0)
    assert plan.preferred_endpoint_devices == (
        "GPU5 RTX PRO 6000 Blackwell (40.0 GB free)",
        "GPU0 RTX 3090 (24.0 GB free)",
    )
    assert plan.preferred_model_devices["vision"] == ("GPU5 RTX PRO 6000 Blackwell (40.0 GB free)",)
    assert plan.preferred_model_devices["summarization"] == ("GPU0 RTX 3090 (24.0 GB free)",)
    assert any("Detected NVLink-connected 3090 pairs" in note for note in plan.notes)


def test_runtime_plan_falls_back_to_host_memory_for_strix_halo():
    plan = build_runtime_plan(
        _settings(),
        _gpu_info(
            amd_gpus=[],
            unified_memory_gb=None,
            total_vram_gb=0.0,
            system_memory_gb=128.0,
        ),
    )

    assert plan.accelerator_count == 1
    assert plan.total_accelerator_memory_gb == 128.0
    assert plan.effective_memory_budget_gb == 96.0
    assert plan.worker_model_loading == ModelLoadingStrategy.PARALLEL
    assert any("inferred Strix Halo unified memory from host RAM" in note for note in plan.notes)


def test_runtime_plan_uses_stage_by_stage_endpoints_on_single_small_gpu():
    plan = build_runtime_plan(
        _settings(
            hardware_profile=HardwareProfile.RTX_5090,
            gpu_backend=GPUBackend.CUDA,
            rocm_llm_runtime=ROCmLLMRuntime.LLAMA_SERVER,
            vision_endpoint_url="http://vision",
            summarization_endpoint_url="http://summary",
            runtime_memory_ceiling_gb=24.0,
        ),
        _gpu_info(
            nvidia_gpus=[
                {"index": 0, "name": "RTX 4090", "vram_gb": 24.0, "free_vram_gb": 24.0},
            ],
            amd_gpus=[],
            unified_memory_gb=None,
            total_vram_gb=24.0,
            available_vram_gb=24.0,
        ),
    )

    assert plan.worker_model_loading == ModelLoadingStrategy.SEQUENTIAL
    assert plan.worker_execution_mode == "stage_by_stage"
    assert plan.endpoint_model_loading == "sequential"
    assert plan.shutdown_endpoint_after_request is True
    assert any("unload each endpoint after its stage completes" in note for note in plan.notes)
