import json
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
            "prompt_texts": list(text),
        }


class PromptMappedClapModel:
    def __init__(self, prompt_scores, support_scores_by_call=None):
        self.prompt_scores = prompt_scores
        self.support_scores_by_call = support_scores_by_call or []
        self.call_count = 0

    def __call__(self, prompt_texts=None, **_kwargs):
        self.call_count += 1
        logits = np.zeros((1, len(prompt_texts)))
        support_override = {}
        if self.support_scores_by_call and self.call_count - 1 < len(self.support_scores_by_call):
            support_override = self.support_scores_by_call[self.call_count - 1]
        for idx, prompt in enumerate(prompt_texts):
            lowered = str(prompt).lower()
            score = 0.0
            for key, value in support_override.items():
                if key in lowered:
                    score = value
                    break
            else:
                for key, value in self.prompt_scores.items():
                    if key in lowered:
                        score = value
                        break
            logits[0, idx] = score
        return types.SimpleNamespace(logits_per_audio=FakeTensor(logits))


def _base_settings(tmp_path: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        audio_event_sample_seconds=1.0,
        audio_event_num_samples=2,
        audio_event_min_score=0.15,
        audio_event_debug=False,
        audio_event_calibration_path=tmp_path / "audio_event_calibration.json",
    )


@pytest.mark.asyncio
async def test_classify_audio_events_uses_clap_candidates(monkeypatch, tmp_path):
    settings = _base_settings(tmp_path)
    model = PromptMappedClapModel(
        {
            "meeting room": 0.4,
            "television news": 5.0,
            "produced podcast": 0.3,
            "screen recording": 0.2,
            "background music": 0.5,
            "audience applause": 0.1,
        }
    )
    manager = DummyManager(
        {
            "type": "audio_event_clap",
            "model": model,
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
async def test_classify_audio_events_handles_missing_torchaudio_info(monkeypatch, tmp_path):
    settings = _base_settings(tmp_path)
    model = PromptMappedClapModel(
        {
            "meeting room": 0.3,
            "television news": 4.5,
            "produced podcast": 0.2,
            "screen recording": 0.1,
            "background music": 0.4,
            "audience applause": 0.1,
        }
    )
    manager = DummyManager(
        {
            "type": "audio_event_clap",
            "model": model,
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
async def test_classify_audio_events_collapses_direct_speech_overlap(monkeypatch, tmp_path):
    settings = _base_settings(tmp_path)
    model = PromptMappedClapModel(
        {
            "meeting room": 4.95,
            "television news": 0.1,
            "produced podcast": 0.3,
            "screen recording": 5.0,
            "background music": 2.8,
            "audience applause": 0.2,
        }
    )
    manager = DummyManager(
        {
            "type": "audio_event_clap",
            "model": model,
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
    assert "noticeable music bed or soundtrack" in result["summary_context"]
    assert manager.released is True


@pytest.mark.asyncio
async def test_classify_audio_events_detects_intermittent_music_with_supporting_aggregation(
    monkeypatch,
    tmp_path,
):
    settings = _base_settings(tmp_path)
    settings.audio_event_num_samples = 3
    model = PromptMappedClapModel(
        {
            "meeting room": 0.6,
            "television news": 0.1,
            "produced podcast": 0.2,
            "screen recording": 4.7,
            "background music": 0.0,
            "audience applause": 0.0,
        },
        support_scores_by_call=[
            {},
            {"background music": 3.6},
            {},
            {"background music": 0.5},
            {},
            {"background music": 3.2},
        ],
    )
    manager = DummyManager(
        {
            "type": "audio_event_clap",
            "model": model,
            "processor": FakeClapProcessor(),
            "device": "cpu",
        }
    )

    monkeypatch.setattr(audio_classification, "get_settings", lambda: settings)
    monkeypatch.setattr(audio_classification, "get_model_manager", lambda: manager)
    monkeypatch.setattr(
        audio_classification,
        "_load_audio_segment",
        lambda _audio_path, _offset, _num_frames: (FakeTensor(np.ones((1, 48_000))), 48_000),
    )

    result = await audio_classification.classify_audio_events(Path("/tmp/sample.wav"))

    assert result["primary_context"]["label"] == "direct software-demo narration"
    assert any(item["key"] == "music_heavy" for item in result["supporting_contexts"])
    assert result["supporting_contexts"][0]["score"] >= 0.15
    assert "noticeable music bed or soundtrack" in result["summary_context"]


def test_load_clap_runtime_profile_merges_override(monkeypatch, tmp_path):
    packaged_path = tmp_path / "packaged_audio_event_calibration.json"
    packaged_path.write_text(
        json.dumps(
            {
                "primary_prompts": {
                    "podcast_voiceover": "packaged narration baseline",
                },
                "supporting_prompts": {
                    "music_heavy": ["packaged soundtrack baseline"],
                },
                "supporting_rules": {
                    "music_heavy": {
                        "aggregation": "top2_mean",
                        "absolute_min_score": 0.07,
                        "relative_ratio": 0.12,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calibration_path = tmp_path / "audio_event_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "primary_prompts": {
                    "software_demo": "custom software walkthrough narration",
                },
                "supporting_prompts": {
                    "music_heavy": ["custom faint background music prompt"],
                },
                "supporting_rules": {
                    "music_heavy": {
                        "aggregation": "max",
                        "absolute_min_score": 0.03,
                        "relative_ratio": 0.05,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    settings = types.SimpleNamespace(
        audio_event_min_score=0.15,
        audio_event_calibration_path=calibration_path,
    )

    monkeypatch.setattr(audio_classification, "PACKAGED_CLAP_RUNTIME_PROFILE_PATH", packaged_path)
    profile = audio_classification.load_clap_runtime_profile(settings)

    assert profile["primary_prompts"]["software_demo"] == "custom software walkthrough narration"
    assert profile["supporting_prompts"]["music_heavy"] == ["custom faint background music prompt"]
    assert profile["supporting_rules"]["music_heavy"]["aggregation"] == "max"
    assert profile["supporting_rules"]["music_heavy"]["absolute_min_score"] == pytest.approx(0.03)
    assert profile["primary_prompts"]["podcast_voiceover"] == "packaged narration baseline"


def test_load_clap_runtime_profile_uses_packaged_baseline_when_override_missing(monkeypatch, tmp_path):
    packaged_path = tmp_path / "packaged_audio_event_calibration.json"
    packaged_path.write_text(
        json.dumps(
            {
                "supporting_prompts": {
                    "music_heavy": ["packaged soundtrack baseline"],
                },
                "supporting_rules": {
                    "music_heavy": {
                        "aggregation": "mean",
                        "absolute_min_score": 0.03,
                        "relative_ratio": 0.05,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    settings = types.SimpleNamespace(
        audio_event_min_score=0.15,
        audio_event_calibration_path=tmp_path / "missing_profile.json",
    )

    monkeypatch.setattr(audio_classification, "PACKAGED_CLAP_RUNTIME_PROFILE_PATH", packaged_path)
    profile = audio_classification.load_clap_runtime_profile(settings)

    assert profile["supporting_prompts"]["music_heavy"] == ["packaged soundtrack baseline"]
    assert profile["supporting_rules"]["music_heavy"]["aggregation"] == "mean"
