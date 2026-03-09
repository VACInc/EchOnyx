import types

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
