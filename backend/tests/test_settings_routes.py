import types

import pytest

from app.api.routes import settings as settings_module
from app.config import GPUBackend, HardwareProfile, ModelLoadingStrategy


@pytest.mark.asyncio
async def test_get_current_settings_exposes_asr_and_audio_event_models(monkeypatch):
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

    response = await settings_module.get_current_settings()

    assert response.models.asr_model == "nvidia/canary-qwen-2.5b"
    assert response.models.asr_family == "canary"
    assert response.models.audio_event_model == "laion/clap-htsat-fused"
    assert "transcription_fallback_model" not in response.models.model_dump()
    assert "transcription_fallback_enabled" not in response.models.model_dump()


@pytest.mark.asyncio
async def test_list_available_models_uses_asr_key():
    response = await settings_module.list_available_models()

    assert "asr" in response
    assert "whisper" not in response
    assert response["asr"][0]["name"] == "nvidia/canary-qwen-2.5b"
    assert response["audio_event"][0]["name"] == "laion/clap-htsat-fused"
