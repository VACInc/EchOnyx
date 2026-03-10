import sys
import types
from pathlib import Path

import numpy as np
import pytest


class FakeTensor:
    def __init__(self, data):
        self.data = np.array(data)

    @property
    def ndim(self):
        return self.data.ndim

    def mean(self, dim=None, keepdim=False):
        return FakeTensor(self.data.mean(axis=dim, keepdims=keepdim))

    def squeeze(self, dim=None):
        axis = dim
        return FakeTensor(np.squeeze(self.data, axis=axis))

    def numpy(self):
        return self.data

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def to(self, _device):
        return self

    def tolist(self):
        return self.data.tolist()

    def item(self):
        return float(self.data.item())

    def numel(self):
        return int(self.data.size)

    def __add__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data + other_data)

    def __truediv__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data / other_data)

    def __getitem__(self, item):
        return FakeTensor(self.data[item])


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_softmax(tensor, dim=-1):
    data = tensor.data
    shifted = data - np.max(data, axis=dim, keepdims=True)
    exp = np.exp(shifted)
    return FakeTensor(exp / exp.sum(axis=dim, keepdims=True))


def _fake_topk(tensor, k):
    flat = tensor.data.reshape(-1)
    indices = np.argsort(flat)[::-1][:k]
    scores = flat[indices]
    return FakeTensor(scores), FakeTensor(indices)


fake_torch = types.SimpleNamespace(
    Tensor=FakeTensor,
    float32="float32",
    ones=lambda shape, dtype=None: FakeTensor(np.ones(shape, dtype=float)),
    zeros=lambda shape, dtype=None: FakeTensor(np.zeros(shape, dtype=float)),
    softmax=_fake_softmax,
    topk=_fake_topk,
    no_grad=lambda: FakeNoGrad(),
    device=lambda name: name,
)

fake_torchaudio = types.SimpleNamespace(
    load=lambda _path, frame_offset=0, num_frames=0: (FakeTensor(np.ones((1, num_frames))), 48_000),
    info=lambda _path: types.SimpleNamespace(num_frames=96_000, sample_rate=48_000),
    transforms=types.SimpleNamespace(Resample=lambda _src, _dst: (lambda waveform: waveform)),
)

sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("torchaudio", fake_torchaudio)

from app.core import audio_classification


class DummyManager:
    def __init__(self, bundle):
        self.bundle = bundle
        self.released = False

    async def get_model(self, _model_type):
        return self.bundle

    async def release_model(self, _model_type):
        self.released = True


class FakeClapProcessor:
    feature_extractor = types.SimpleNamespace(sampling_rate=48_000)

    def __call__(self, text=None, audios=None, sampling_rate=None, return_tensors=None, padding=None):
        assert sampling_rate == 48_000
        assert text is not None
        assert audios is not None
        assert return_tensors == "pt"
        assert padding is True
        return {
            "input_ids": FakeTensor(np.ones((len(text), 2))),
            "attention_mask": FakeTensor(np.ones((len(text), 2))),
            "input_features": FakeTensor(np.ones((1, 8))),
        }


class FakeBroadcastClapModel:
    def __call__(self, **_kwargs):
        logits = np.zeros((1, len(audio_classification.CLAP_HINT_CANDIDATES)))
        logits[0, 1] = 5.0
        logits[0, 4] = 2.0
        return types.SimpleNamespace(logits_per_audio=FakeTensor(logits))


class FakeDirectSpeechClapModel:
    def __call__(self, **_kwargs):
        logits = np.zeros((1, len(audio_classification.CLAP_HINT_CANDIDATES)))
        logits[0, 3] = 5.0
        logits[0, 0] = 4.95
        logits[0, 4] = 4.6
        return types.SimpleNamespace(logits_per_audio=FakeTensor(logits))


@pytest.mark.asyncio
async def test_classify_audio_events_uses_clap_candidates(monkeypatch):
    settings = types.SimpleNamespace(
        audio_event_sample_seconds=1.0,
        audio_event_num_samples=2,
        audio_event_min_score=0.15,
        audio_event_debug=False,
    )
    manager = DummyManager(
        {
            "type": "audio_event_clap",
            "model": FakeBroadcastClapModel(),
            "processor": FakeClapProcessor(),
            "device": "cpu",
        }
    )

    monkeypatch.setattr(audio_classification, "get_settings", lambda: settings)
    monkeypatch.setattr(audio_classification, "get_model_manager", lambda: manager)
    monkeypatch.setattr(
        audio_classification.torchaudio,
        "info",
        lambda _path: types.SimpleNamespace(num_frames=96_000, sample_rate=48_000),
    )
    monkeypatch.setattr(
        audio_classification,
        "_load_audio_segment",
        lambda _audio_path, _offset, _num_frames: (FakeTensor(np.ones((1, 48_000))), 48_000),
    )

    result = await audio_classification.classify_audio_events(Path("/tmp/sample.wav"))

    assert result["top_labels"][0]["label"] == "broadcast playback"
    assert result["primary_context"]["label"] == "broadcast or TV playback"
    assert result["primary_context"]["confidence"] == "high"
    assert result["tv_score"] > result["speech_score"]
    assert any("television or broadcast playback" in hint for hint in result["hints"])
    assert "Primary audio context: broadcast or TV playback" in result["summary_context"]
    assert manager.released is True


@pytest.mark.asyncio
async def test_classify_audio_events_handles_missing_torchaudio_info(monkeypatch):
    settings = types.SimpleNamespace(
        audio_event_sample_seconds=1.0,
        audio_event_num_samples=1,
        audio_event_min_score=0.15,
        audio_event_debug=False,
    )
    manager = DummyManager(
        {
            "type": "audio_event_clap",
            "model": FakeBroadcastClapModel(),
            "processor": FakeClapProcessor(),
            "device": "cpu",
        }
    )

    def fake_load(_path, frame_offset=0, num_frames=0):
        frames = num_frames or 48_000
        return FakeTensor(np.ones((1, frames))), 48_000

    monkeypatch.setattr(audio_classification, "get_settings", lambda: settings)
    monkeypatch.setattr(audio_classification, "get_model_manager", lambda: manager)
    monkeypatch.delattr(audio_classification.torchaudio, "info", raising=False)
    monkeypatch.setattr(audio_classification.torchaudio, "load", fake_load)

    result = await audio_classification.classify_audio_events(Path("/tmp/sample.wav"))

    assert result["top_labels"][0]["label"] == "broadcast playback"
    assert result["primary_context"]["label"] == "broadcast or TV playback"
    assert any("television or broadcast playback" in hint for hint in result["hints"])
    assert "Primary audio context: broadcast or TV playback" in result["summary_context"]
    assert manager.released is True


@pytest.mark.asyncio
async def test_classify_audio_events_collapses_direct_speech_overlap(monkeypatch):
    settings = types.SimpleNamespace(
        audio_event_sample_seconds=1.0,
        audio_event_num_samples=2,
        audio_event_min_score=0.15,
        audio_event_debug=False,
    )
    manager = DummyManager(
        {
            "type": "audio_event_clap",
            "model": FakeDirectSpeechClapModel(),
            "processor": FakeClapProcessor(),
            "device": "cpu",
        }
    )

    monkeypatch.setattr(audio_classification, "get_settings", lambda: settings)
    monkeypatch.setattr(audio_classification, "get_model_manager", lambda: manager)
    monkeypatch.setattr(
        audio_classification.torchaudio,
        "info",
        lambda _path: types.SimpleNamespace(num_frames=96_000, sample_rate=48_000),
    )
    monkeypatch.setattr(
        audio_classification,
        "_load_audio_segment",
        lambda _audio_path, _offset, _num_frames: (FakeTensor(np.ones((1, 48_000))), 48_000),
    )

    result = await audio_classification.classify_audio_events(Path("/tmp/sample.wav"))

    assert result["primary_context"]["key"] == "software_demo_direct_speech"
    assert result["primary_context"]["label"] == "direct software-demo narration"
    assert result["speech_score"] > result["tv_score"]
    assert any(item["key"] == "music_heavy" for item in result["supporting_contexts"])
    assert any("noticeable music" in hint for hint in result["hints"])
    assert "Supporting audio cues: noticeable music bed or soundtrack." in result["summary_context"]
    assert manager.released is True
