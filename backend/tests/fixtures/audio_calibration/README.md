This fixture pack contains a small CLAP calibration baseline covering all
current primary classes plus the two supporting cues. The current mix is:

- `meeting_room_speech.wav`: short meeting-style speech
- `meeting_with_applause.wav`: meeting-style speech with applause support
- `broadcast_playback.wav`: broadcast-style announcement playback
- `software_demo_narration.wav`: narration-driven software walkthrough
- `voiceover_no_music.wav`: produced narration with no supporting soundtrack cue
- `voiceover_with_music.wav`: produced narration with a light music bed

They are intentionally small and are used to:

- validate the calibration CLI against real media artifacts
- provide a checked-in baseline profile for soundtrack sensitivity
- broaden the calibration set beyond only voice-over examples
- keep the end-user upload flow fully automatic

Regenerate the baseline profile with:

```bash
python -m app.core.audio_calibration \
  --manifest backend/tests/fixtures/audio_calibration/manifest.json \
  --output backend/app/assets/audio_event_calibration.json
```
