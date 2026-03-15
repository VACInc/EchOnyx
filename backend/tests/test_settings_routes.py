import os
import types

import pytest

from app.api.routes import settings as settings_module
from app.config import DuplicateHandlingPolicy, GPUBackend, HardwareProfile, ModelLoadingStrategy


@pytest.mark.asyncio
async def test_get_current_settings_exposes_asr_and_audio_event_models(monkeypatch):
    runtime_plan = {
        "accelerator_count": 1,
        "total_accelerator_memory_gb": 124.94,
        "available_accelerator_memory_gb": 124.94,
        "effective_memory_budget_gb": 93.7,
        "placement_mode": "unified_memory_apu",
        "worker_execution_mode": "resident_all",
        "worker_model_loading": "parallel",
        "endpoint_model_loading": "sequential",
        "keep_resident_models": ["whisper", "diarization", "embedding", "audio_event"],
        "preferred_worker_devices": ["APU unified memory"],
        "preferred_endpoint_devices": ["APU unified memory"],
        "preferred_model_devices": {
            "worker": ["APU unified memory"],
            "vision": ["APU unified memory"],
            "summarization": ["APU unified memory"],
        },
        "can_keep_all_worker_models_loaded": True,
        "can_keep_endpoint_models_loaded": False,
        "requires_endpoint_idle_teardown": True,
        "endpoint_idle_timeout_recommendation_s": 120,
        "shutdown_endpoint_after_request": False,
        "estimated_memory_by_model_gb": {
            "whisper": 6.0,
            "diarization": 2.0,
            "embedding": 16.0,
            "audio_event": 2.5,
            "vision": 24.0,
            "summarization": 24.0,
        },
        "notes": ["planner note"],
    }
    fake_settings = types.SimpleNamespace(
        hardware_profile=HardwareProfile.STRIX_HALO,
        gpu_backend=GPUBackend.ROCM,
        model_loading=ModelLoadingStrategy.SEQUENTIAL,
        whisper_model="nvidia/canary-qwen-2.5b",
        granite_force_cpu=False,
        diarization_model="pyannote/speaker-diarization-community-1",
        vision_model="vision.gguf",
        vision_mmproj="vision.mmproj",
        vision_chat_format="qwen3-vl",
        vision_endpoint_url="http://vision",
        vision_endpoint_model="vision-endpoint",
        summarization_model="summary.gguf",
        summarization_endpoint_url="http://summary",
        summarization_endpoint_model="summary-endpoint",
        embedding_model="Qwen/Qwen3-Embedding-8B",
        audio_event_model="laion/clap-htsat-fused",
        runtime_planner_enabled=True,
        gpu_memory_fraction=0.75,
        runtime_memory_ceiling_gb=None,
        rocm_llm_runtime="llama_server",
        rocm_llm_idle_timeout_s=120,
        duplicate_detection_policy=DuplicateHandlingPolicy.COLLAPSE_EXACT,
        duplicate_exact_threshold=0.95,
        duplicate_probable_threshold=0.85,
        action_items_enabled=True,
        max_video_length_hours=4,
        keyframe_extraction_interval=5,
        frame_persistence_seconds=3.0,
        frame_change_threshold=12.0,
        frame_stability_threshold=6.0,
        frame_dedupe_threshold=4.0,
        frame_resize_width=320,
        max_keyframes=0,
        min_speech_duration=0.5,
        batch_concurrent_jobs=1,
        summary_chunk_minutes=6.0,
        summary_chunk_overlap_minutes=0.6,
    )
    monkeypatch.setattr(settings_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(
        settings_module,
        "get_hardware_info",
        lambda: {
                "detected_gpus": {"nvidia": [], "amd": [{"name": "AMD GPU", "vram_gb": 16.0}]},
                "unified_memory_gb": 128.0,
                "total_vram_gb": 16.0,
                "available_vram_gb": 16.0,
                "active_profile": "strix_halo",
                "active_backend": "rocm",
                "whisper_backend": "rocm",
            "asr_family": "canary",
            "model_loading_strategy": "parallel",
            "rocm_llm_runtime": "llama_server",
            "rocm_llm_idle_timeout_s": 120,
            "runtime_planner_enabled": True,
            "runtime_memory_ceiling_gb": None,
            "gpu_memory_fraction": 0.75,
            "runtime_plan": runtime_plan,
        },
    )

    response = await settings_module.get_current_settings()

    assert response.models.asr_model == "nvidia/canary-qwen-2.5b"
    assert response.models.asr_family == "canary"
    assert response.models.audio_event_model == "laion/clap-htsat-fused"
    assert response.models.rocm_llm_runtime == "llama_server"
    assert response.models.rocm_llm_idle_timeout_s == 120
    assert response.runtime_planner.accelerator_count == 1
    assert response.runtime_planner.total_accelerator_memory_gb == 124.94
    assert response.runtime_planner.available_accelerator_memory_gb == 124.94
    assert response.runtime_planner.worker_execution_mode == "resident_all"
    assert response.runtime_planner.worker_model_loading == "parallel"
    assert response.runtime_planner.endpoint_model_loading == "sequential"
    assert response.runtime_planner.preferred_worker_devices == ["APU unified memory"]
    assert response.runtime_planner.preferred_endpoint_devices == ["APU unified memory"]
    assert response.runtime_planner.preferred_model_devices["worker"] == ["APU unified memory"]
    assert response.duplicates.policy == "collapse_exact"
    assert response.duplicates.exact_threshold == 0.95
    assert response.duplicates.probable_threshold == 0.85
    assert response.action_items.enabled is True
    assert "transcription_fallback_model" not in response.models.model_dump()
    assert "transcription_fallback_enabled" not in response.models.model_dump()


@pytest.mark.asyncio
async def test_list_available_models_uses_asr_key():
    response = await settings_module.list_available_models()

    assert "asr" in response
    assert "whisper" not in response
    assert response["asr"][0]["name"] == "nvidia/canary-qwen-2.5b"
    assert response["audio_event"][0]["name"] == "laion/clap-htsat-fused"
    assert any(model["name"] == "Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf" for model in response["vision"])
    assert any(model["name"] == "Qwen2.5-3B-Instruct.Q4_K_M.gguf" for model in response["summarization"])
    assert any(model["name"] == "Qwen3VL-32B-Instruct-Q4_K_M.gguf" for model in response["vision"])
    assert any(model["name"] == "Qwen3-30B-A3B-Q4_K_M.gguf" for model in response["summarization"])


@pytest.mark.asyncio
async def test_verify_model_candidate_accepts_catalog_model():
    response = await settings_module.verify_model_candidate(
        settings_module.ModelVerifyRequest(component="embedding", model_name="Qwen/Qwen3-Embedding-8B")
    )

    assert response.exists is True
    assert response.source == "catalog"


@pytest.mark.asyncio
async def test_verify_model_candidate_checks_huggingface(monkeypatch):
    async def fake_exists(_model_name: str) -> bool:
        return True

    monkeypatch.setattr(settings_module, "_huggingface_model_exists", fake_exists)

    response = await settings_module.verify_model_candidate(
        settings_module.ModelVerifyRequest(component="embedding", model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    assert response.exists is True
    assert response.source == "huggingface"


@pytest.mark.asyncio
async def test_verify_model_candidate_rejects_unknown_gguf_name():
    response = await settings_module.verify_model_candidate(
        settings_module.ModelVerifyRequest(component="vision", model_name="unknown.gguf")
    )

    assert response.exists is False
    assert response.source == "unsupported"


@pytest.mark.asyncio
async def test_update_settings_persists_asr_and_planner_fields(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("WHISPER_MODEL=large-v3\nGPU_MEMORY_FRACTION=0.75\n", encoding="utf-8")
    captured = {}

    async def fake_reload():
        captured["reloaded"] = True

    monkeypatch.setattr(settings_module, "_resolve_env_file_path", lambda: env_path)
    monkeypatch.setattr(settings_module, "_reload_runtime_state", fake_reload)
    async def fake_get_current_settings():
        return types.SimpleNamespace(
            hardware_profile="strix_halo",
            gpu_backend="rocm",
            model_loading="parallel",
            models=types.SimpleNamespace(asr_model="nvidia/canary-qwen-2.5b"),
            runtime_planner=types.SimpleNamespace(worker_model_loading="parallel"),
            duplicates=types.SimpleNamespace(policy="collapse_probable"),
            action_items=types.SimpleNamespace(enabled=False),
            processing=types.SimpleNamespace(max_video_length_hours=4),
        )

    monkeypatch.setattr(settings_module, "get_current_settings", fake_get_current_settings)

    await settings_module.update_settings(
        settings_module.SettingsUpdate(
            asr_model="nvidia/canary-qwen-2.5b",
            runtime_planner_enabled=True,
            gpu_memory_fraction=0.6,
            runtime_memory_ceiling_gb=96,
            duplicate_detection_policy=DuplicateHandlingPolicy.COLLAPSE_PROBABLE,
            duplicate_exact_threshold=0.96,
            duplicate_probable_threshold=0.88,
            action_items_enabled=False,
        )
    )

    persisted = env_path.read_text(encoding="utf-8")
    assert "WHISPER_MODEL=nvidia/canary-qwen-2.5b" in persisted
    assert "RUNTIME_PLANNER_ENABLED=true" in persisted
    assert "GPU_MEMORY_FRACTION=0.6" in persisted
    assert "RUNTIME_MEMORY_CEILING_GB=96.0" in persisted
    assert "DUPLICATE_DETECTION_POLICY=collapse_probable" in persisted
    assert "DUPLICATE_EXACT_THRESHOLD=0.96" in persisted
    assert "DUPLICATE_PROBABLE_THRESHOLD=0.88" in persisted
    assert "ACTION_ITEMS_ENABLED=false" in persisted
    assert captured["reloaded"] is True


@pytest.mark.asyncio
async def test_update_settings_removes_nullable_env_fields(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "WHISPER_MODEL=nvidia/canary-qwen-2.5b\nRUNTIME_MEMORY_CEILING_GB=20\n",
        encoding="utf-8",
    )

    async def fake_reload():
        return None

    async def fake_get_current_settings():
        return types.SimpleNamespace(
            hardware_profile="strix_halo",
            gpu_backend="rocm",
            model_loading="parallel",
            models=types.SimpleNamespace(asr_model="nvidia/canary-qwen-2.5b"),
            runtime_planner=types.SimpleNamespace(worker_model_loading="parallel"),
            duplicates=types.SimpleNamespace(policy="collapse_exact"),
            action_items=types.SimpleNamespace(enabled=True),
            processing=types.SimpleNamespace(max_video_length_hours=4),
        )

    monkeypatch.setattr(settings_module, "_resolve_env_file_path", lambda: env_path)
    monkeypatch.setattr(settings_module, "_reload_runtime_state", fake_reload)
    monkeypatch.setattr(settings_module, "get_current_settings", fake_get_current_settings)

    os.environ["RUNTIME_MEMORY_CEILING_GB"] = "20"

    await settings_module.update_settings(
        settings_module.SettingsUpdate(runtime_memory_ceiling_gb=None)
    )

    persisted = env_path.read_text(encoding="utf-8")
    assert "RUNTIME_MEMORY_CEILING_GB" not in persisted
    assert "RUNTIME_MEMORY_CEILING_GB" not in os.environ


def test_write_env_updates_preserves_comments(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("# comment\nWHISPER_MODEL=large-v3\n", encoding="utf-8")

    settings_module._write_env_updates(env_path, {
        "WHISPER_MODEL": "medium",
        "GPU_MEMORY_FRACTION": 0.5,
    })

    payload = env_path.read_text(encoding="utf-8")
    assert "# comment" in payload
    assert "WHISPER_MODEL=medium" in payload
    assert "GPU_MEMORY_FRACTION=0.5" in payload


def test_write_env_updates_removes_none_values(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\nWHISPER_MODEL=large-v3\nRUNTIME_MEMORY_CEILING_GB=20\n",
        encoding="utf-8",
    )

    settings_module._write_env_updates(env_path, {
        "RUNTIME_MEMORY_CEILING_GB": None,
    })

    payload = env_path.read_text(encoding="utf-8")
    assert "# comment" in payload
    assert "WHISPER_MODEL=large-v3" in payload
    assert "RUNTIME_MEMORY_CEILING_GB" not in payload
