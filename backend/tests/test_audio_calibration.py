import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


class FakeTensor:
    def __init__(self, data):
        self.data = np.array(data)

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

    def item(self):
        return float(self.data.item())

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


fake_torch = types.SimpleNamespace(
    Tensor=FakeTensor,
    float32="float32",
    softmax=_fake_softmax,
    no_grad=lambda: FakeNoGrad(),
    device=lambda name: name,
)

fake_torchaudio = types.SimpleNamespace(
    load=lambda _path, frame_offset=0, num_frames=0: (FakeTensor(np.ones((1, num_frames or 1))), 48_000),
    info=lambda _path: types.SimpleNamespace(num_frames=96_000, sample_rate=48_000),
    transforms=types.SimpleNamespace(Resample=lambda _src, _dst: (lambda waveform: waveform)),
)

sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("torchaudio", fake_torchaudio)

from app.core import audio_calibration


def _fixture_observations() -> list[dict]:
    return [
        {
            "label": "voiceover_music",
            "media_path": "/tmp/voiceover_music.mp4",
            "expected_primary_key": "podcast_voiceover",
            "expected_supporting_keys": ["music_heavy"],
            "primary_window_scores": {
                "meeting_room_speech": [0.05, 0.04],
                "broadcast_playback": [0.01, 0.01],
                "podcast_voiceover": [0.92, 0.94],
                "software_demo": [0.22, 0.2],
            },
            "supporting_prompt_scores": {
                "music_heavy": {
                    "spoken narration with background music or soundtrack": [0.62, 0.66],
                    "corporate explainer narration with light underscore music": [0.81, 0.84],
                    "video intro or outro with speech over music": [0.23, 0.25],
                    "spoken presentation with faint background music": [0.77, 0.79],
                },
                "crowd_applause": {
                    "audience applause or crowd reaction": [0.01, 0.01],
                    "live event cheering or clapping audience": [0.01, 0.01],
                    "room applause after a talk or presentation": [0.01, 0.01],
                },
            },
        },
        {
            "label": "voiceover_clean",
            "media_path": "/tmp/voiceover_clean.mp4",
            "expected_primary_key": "podcast_voiceover",
            "expected_supporting_keys": [],
            "primary_window_scores": {
                "meeting_room_speech": [0.06, 0.05],
                "broadcast_playback": [0.01, 0.01],
                "podcast_voiceover": [0.91, 0.9],
                "software_demo": [0.2, 0.18],
            },
            "supporting_prompt_scores": {
                "music_heavy": {
                    "spoken narration with background music or soundtrack": [0.04, 0.05],
                    "corporate explainer narration with light underscore music": [0.02, 0.03],
                    "video intro or outro with speech over music": [0.01, 0.01],
                    "spoken presentation with faint background music": [0.03, 0.03],
                },
                "crowd_applause": {
                    "audience applause or crowd reaction": [0.01, 0.01],
                    "live event cheering or clapping audience": [0.01, 0.01],
                    "room applause after a talk or presentation": [0.01, 0.01],
                },
            },
        },
        {
            "label": "meeting_clean",
            "media_path": "/tmp/meeting_clean.mp4",
            "expected_primary_key": "meeting_room_speech",
            "expected_supporting_keys": [],
            "primary_window_scores": {
                "meeting_room_speech": [0.88, 0.9],
                "broadcast_playback": [0.01, 0.01],
                "podcast_voiceover": [0.08, 0.08],
                "software_demo": [0.22, 0.21],
            },
            "supporting_prompt_scores": {
                "music_heavy": {
                    "spoken narration with background music or soundtrack": [0.02, 0.03],
                    "corporate explainer narration with light underscore music": [0.02, 0.02],
                    "video intro or outro with speech over music": [0.01, 0.01],
                    "spoken presentation with faint background music": [0.02, 0.02],
                },
                "crowd_applause": {
                    "audience applause or crowd reaction": [0.01, 0.01],
                    "live event cheering or clapping audience": [0.01, 0.01],
                    "room applause after a talk or presentation": [0.01, 0.01],
                },
            },
        },
    ]


def test_load_audio_calibration_manifest_resolves_relative_media_paths(tmp_path):
    fixture_path = tmp_path / "fixtures" / "voiceover.mp4"
    fixture_path.parent.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "media_path": "fixtures/voiceover.mp4",
                        "expected_primary_key": "podcast_voiceover",
                        "expected_supporting_keys": ["music_heavy"],
                        "label": "voiceover_music",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    fixtures = audio_calibration.load_audio_calibration_manifest(manifest_path)

    assert fixtures[0].media_path == fixture_path.resolve()
    assert fixtures[0].expected_primary_key == "podcast_voiceover"
    assert fixtures[0].expected_supporting_keys == ("music_heavy",)
    assert fixtures[0].use_for_calibration is True


def test_load_audio_calibration_manifest_preserves_exploratory_fixtures(tmp_path):
    fixture_path = tmp_path / "fixtures" / "meeting.wav"
    fixture_path.parent.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "media_path": "fixtures/meeting.wav",
                        "expected_primary_key": "meeting_room_speech",
                        "expected_supporting_keys": [],
                        "label": "meeting_room_real",
                        "use_for_calibration": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    fixtures = audio_calibration.load_audio_calibration_manifest(manifest_path)

    assert fixtures[0].media_path == fixture_path.resolve()
    assert fixtures[0].use_for_calibration is False


def test_select_calibration_fixtures_skips_exploratory_entries():
    fixtures = [
        audio_calibration.AudioCalibrationFixture(
            media_path=Path("/tmp/voiceover.wav"),
            expected_primary_key="podcast_voiceover",
            expected_supporting_keys=(),
            label="voiceover",
            use_for_calibration=True,
        ),
        audio_calibration.AudioCalibrationFixture(
            media_path=Path("/tmp/meeting.wav"),
            expected_primary_key="meeting_room_speech",
            expected_supporting_keys=(),
            label="meeting_room_real",
            use_for_calibration=False,
        ),
    ]

    selected = audio_calibration.select_calibration_fixtures(fixtures)

    assert [fixture.label for fixture in selected] == ["voiceover"]


def test_select_calibration_fixtures_requires_at_least_one_enabled_fixture():
    fixtures = [
        audio_calibration.AudioCalibrationFixture(
            media_path=Path("/tmp/meeting.wav"),
            expected_primary_key="meeting_room_speech",
            expected_supporting_keys=(),
            label="meeting_room_real",
            use_for_calibration=False,
        )
    ]

    with pytest.raises(ValueError, match="use_for_calibration=true"):
        audio_calibration.select_calibration_fixtures(fixtures)


def test_calibrate_clap_profile_from_observations_selects_music_prompt_and_rule(monkeypatch):
    monkeypatch.setattr(
        audio_calibration,
        "get_settings",
        lambda: types.SimpleNamespace(audio_event_min_score=0.15),
    )

    profile = audio_calibration.calibrate_clap_profile_from_observations(_fixture_observations())

    assert profile["primary_prompts"]["podcast_voiceover"]
    assert "music_heavy" in profile["supporting_prompts"]
    assert len(profile["supporting_prompts"]["music_heavy"]) >= 1
    assert "corporate explainer narration with light underscore music" in profile["supporting_prompts"]["music_heavy"]
    assert profile["supporting_rules"]["music_heavy"]["aggregation"] in {"mean", "max", "top2_mean"}
    assert profile["supporting_rules"]["music_heavy"]["absolute_min_score"] in audio_calibration.ABSOLUTE_THRESHOLD_GRID
    assert profile["metrics"]["fixtures_evaluated"] == 3


@pytest.mark.asyncio
async def test_calibrate_audio_events_manifest_writes_profile(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "media_path": "/tmp/voiceover_music.mp4",
                        "expected_primary_key": "podcast_voiceover",
                        "expected_supporting_keys": ["music_heavy"],
                        "label": "voiceover_music",
                        "use_for_calibration": True,
                    },
                    {
                        "media_path": "/tmp/exploratory_meeting.wav",
                        "expected_primary_key": "meeting_room_speech",
                        "expected_supporting_keys": [],
                        "label": "meeting_room_real",
                        "use_for_calibration": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "audio_event_calibration.json"

    monkeypatch.setattr(
        audio_calibration,
        "collect_clap_fixture_observations",
        lambda fixtures, scratch_dir=None: _async_result(_fixture_observations()),
    )
    monkeypatch.setattr(
        audio_calibration,
        "get_settings",
        lambda: types.SimpleNamespace(audio_event_min_score=0.15),
    )

    profile = await audio_calibration.calibrate_audio_events_manifest(manifest_path, output_path)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert profile["supporting_rules"]["music_heavy"] == written["supporting_rules"]["music_heavy"]
    assert written["metrics"]["fixtures_evaluated"] == 3


@pytest.mark.asyncio
async def test_materialize_audio_path_transcodes_non_wav(monkeypatch, tmp_path):
    media_path = tmp_path / "fixture.mp3"
    media_path.write_bytes(b"placeholder")
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    captured = {}

    async def fake_extract_audio(input_path, output_path):
        captured["input_path"] = input_path
        captured["output_path"] = output_path
        output_path.write_bytes(b"wav")
        return output_path

    monkeypatch.setattr(audio_calibration, "extract_audio", fake_extract_audio)

    result = await audio_calibration._materialize_audio_path(media_path, scratch_dir)

    assert result == scratch_dir / "fixture.wav"
    assert captured["input_path"] == media_path
    assert captured["output_path"] == result


def test_repo_audio_calibration_fixture_pack_is_checked_in():
    manifest_path = Path(__file__).resolve().parent / "fixtures" / "audio_calibration" / "manifest.json"

    fixtures = audio_calibration.load_audio_calibration_manifest(manifest_path)

    assert [fixture.label for fixture in fixtures] == [
        "voiceover_no_music",
        "voiceover_with_music",
        "broadcast_weather_radio",
        "applause_real",
        "meeting_room_real",
        "software_demo_real",
    ]
    assert [fixture.label for fixture in audio_calibration.select_calibration_fixtures(fixtures)] == [
        "voiceover_no_music",
        "voiceover_with_music",
        "broadcast_weather_radio",
        "applause_real",
    ]
    for fixture in fixtures:
        assert fixture.media_path.exists()


def test_packaged_audio_calibration_profile_stays_conservative_default():
    profile_path = Path(__file__).resolve().parents[1] / "app" / "assets" / "audio_event_calibration.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert profile["metrics"]["fixtures_evaluated"] == 2
    assert profile["metrics"]["labels"] == ["voiceover_no_music", "voiceover_with_music"]
    assert "music_heavy" in profile["supporting_prompts"]
    assert "corporate explainer narration with light underscore music" in profile["supporting_prompts"]["music_heavy"]
    assert profile["supporting_rules"]["music_heavy"]["absolute_min_score"] == pytest.approx(0.03)


async def _async_result(value):
    return value
