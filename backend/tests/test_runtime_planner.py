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
    assert plan.can_keep_all_worker_models_loaded is True
    assert set(plan.keep_resident_models) == {"whisper", "diarization", "embedding", "audio_event"}
    assert plan.requires_endpoint_idle_teardown is True
    assert plan.can_keep_endpoint_models_loaded is False


def test_runtime_plan_honors_explicit_memory_ceiling():
    plan = build_runtime_plan(
        _settings(runtime_memory_ceiling_gb=20.0),
        _gpu_info(),
    )

    assert plan.effective_memory_budget_gb == 20.0
    assert plan.worker_model_loading == ModelLoadingStrategy.SEQUENTIAL
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
    assert any("Multiple GPUs were detected" in note for note in plan.notes)


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
