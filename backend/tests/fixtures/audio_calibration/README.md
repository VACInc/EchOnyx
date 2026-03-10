This fixture pack contains two short CLAP calibration clips captured from the
validated Strix Halo benchmark runs on March 10, 2026:

- `voiceover_no_music.wav`: produced narration with no supporting soundtrack cue
- `voiceover_with_music.wav`: produced narration with a light music bed

They are intentionally small and are used to:

- validate the calibration CLI against real media artifacts
- provide a checked-in baseline profile for soundtrack sensitivity
- keep the end-user upload flow fully automatic

Regenerate the baseline profile with:

```bash
python -m app.core.audio_calibration \
  --manifest backend/tests/fixtures/audio_calibration/manifest.json \
  --output backend/app/assets/audio_event_calibration.json
```
