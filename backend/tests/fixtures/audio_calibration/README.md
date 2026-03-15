This fixture pack now separates validated calibration fixtures from exploratory
real-world clips.

Validated calibration fixtures (`use_for_calibration: true` in the manifest):

- `voiceover_no_music.wav`: controlled clean narration baseline
- `voiceover_with_music.wav`: controlled narration with a light music bed
- `broadcast_weather_radio.wav`: real NOAA weather-radio clip
- `applause_real.wav`: real applause-only crowd cue

Exploratory real fixtures (`use_for_calibration: false`):

- `meeting_room_real.wav`: public-domain congressional hearing clip
- `software_demo_real.wav`: Wikimedia content-translation screencast narration

Why the split:

- the real weather-radio and applause clips survived live Strix Halo validation
- the real meeting and software-demo clips still collapsed toward produced narration in raw CLAP audio-only classification
- the exploratory clips remain checked in so future prompt/model work can benchmark against them without distorting the default calibration path
- primary prompt calibration now scores the actual prompt variants; the packaged baseline was regenerated from the four validated fixtures after fixing that bug

Source and license notes:

- `broadcast_weather_radio.wav`
  - source: <https://commons.wikimedia.org/wiki/File:NOAA_Weather_Radio_WXL40.ogg>
  - license: `CC BY-SA 3.0`
- `applause_real.wav`
  - source: <https://commons.wikimedia.org/wiki/File:Applause.ogg>
  - license: `CC BY-SA 3.0`
- `meeting_room_real.wav`
  - source: <https://commons.wikimedia.org/wiki/File:Rep._Greg_Walden_Opens_a_House_Energy_and_Commerce_Committee_Hearing_on_Facebook.ogg>
  - license: public domain
- `software_demo_real.wav`
  - source: <https://commons.wikimedia.org/wiki/File:Content_Translation_Screencast_(English).webm>
  - license: `CC BY-SA 4.0`

Regenerate the baseline profile with:

```bash
python -m app.core.audio_calibration \
  --manifest backend/tests/fixtures/audio_calibration/manifest.json \
  --output backend/app/assets/audio_event_calibration.json
```
