import sys
import types

import numpy as np

import pytest

from app.core import transcription


class DummyManager:
    def __init__(self, model):
        self.model = model
        self.released = False

    async def get_model(self, _model_type):
        return self.model

    async def release_model(self, _model_type):
        self.released = True


@pytest.mark.asyncio
async def test_transcribe_audio_does_not_fallback_when_primary_model_errors(monkeypatch, tmp_path):
    manager = DummyManager({"type": "granite"})
    fallback_called = False

    async def fail_primary(*_args, **_kwargs):
        raise RuntimeError("primary transcription failed")

    async def unexpected_fallback(*_args, **_kwargs):
        nonlocal fallback_called
        fallback_called = True
        return {}

    monkeypatch.setattr(transcription, "get_model_manager", lambda: manager)
    monkeypatch.setattr(transcription, "_transcribe_transformers_asr", fail_primary)
    monkeypatch.setattr(transcription, "_transcribe_with_faster_whisper", unexpected_fallback)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"placeholder")

    with pytest.raises(RuntimeError, match="primary transcription failed"):
        await transcription.transcribe_audio(audio_path)

    assert fallback_called is False
    assert manager.released is True


@pytest.mark.asyncio
async def test_transcribe_nemo_canary_uses_settings(monkeypatch, tmp_path):
    class FakeTokenizer:
        def ids_to_text(self, _ids):
            return "Budget review due Friday."

    class FakeModel:
        audio_locator_tag = "<audio>"
        tokenizer = FakeTokenizer()

        def generate(self, prompts, max_new_tokens: int):
            assert prompts
            assert max_new_tokens == 256
            return [np.array([1, 2, 3])]

    fake_sf = types.SimpleNamespace(
        read=lambda _path: (np.array([0.1, 0.2, 0.3], dtype=np.float32), 16000),
        write=lambda _path, _audio, _sample_rate: None,
    )

    monkeypatch.setattr(
        transcription,
        "get_settings",
        lambda: types.SimpleNamespace(asr_chunk_length_s=30.0, asr_chunk_overlap_s=2.0),
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"placeholder")

    result = await transcription._transcribe_nemo_canary(
        {"model": FakeModel()},
        audio_path,
    )

    assert result["text"] == "Budget review due Friday."
    assert result["language"] == "en"


def test_cuda_device_id_supports_indexed_cuda_devices():
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))

    assert transcription._cuda_device_id("cuda:5", fake_torch) == 5
