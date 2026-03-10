# Project Status And Backlog

## Current Operating Position

- Strix Halo is required to run ROCm-only for all inference stages. CPU fallback is not an acceptable normal path there.
- The live Strix Halo default should remain `llama_server` for now.
- The `vllm` runtime stays in the repo as an opt-in ROCm backend for future work and for later NVIDIA comparison.
- Live validation on the Strix Halo remains part of the acceptance bar for runtime changes.

## Validated Findings And Constraints

- Strix Halo currently behaves best with the managed ROCm `llama_server` path plus idle teardown.
- GPU idle must be validated with clocks and power, not only `rocm-smi --showuse`.
- The original "GPU pinned at 100%" issue on Strix Halo was real with persistent ROCm model-server processes; managed teardown resolved it for the `llama_server` path.
- `vllm` on Strix Halo `gfx1151` is not a good default today:
  - the FP8 model path is not viable on this GPU family
  - the BF16 path can boot on ROCm, but startup is much slower
  - large vision and summarization engines contend heavily for memory
  - child engine processes must be torn down as a process group or VRAM remains pinned
- On Strix Halo, `vllm` currently looks more like an experimental path than a production default.
- For this hardware, the practical tradeoff today is:
  - `llama_server`: slower cold loads, but operationally reliable
  - `vllm`: promising future path, but currently too heavy and fragile for default use
- The CLAP audio-event path is now live and is producing a single structured primary context for summarization instead of a noisy flat hint list.
- Live Strix Halo benchmark on March 10, 2026:
  - CLAP classified a narrated demo clip as `produced narration or voice-over` with high confidence
  - the summary with audio context added explicit narration context and one extra evidence-based key point compared with the no-audio version
  - the same summary call took about `27.3s` with audio context versus `13.7s` without it on a warm summarization endpoint
- Current CLAP limitation from that benchmark:
  - a light synthetic music bed did not clear the supporting-cue threshold, so soundtrack sensitivity still needs calibration
- Follow-up live Strix Halo benchmark on March 10, 2026 after the automatic support-scoring pass:
  - the same narrated benchmark clip now emits both `produced narration or voice-over` and `noticeable music bed or soundtrack`
  - the saved audio-event artifact shows soundtrack support scoring around `0.94` for that clip
  - the post-run GPU idle check still returned to `0%` use at about `607-609 MHz`
- The repo now includes a checked-in CLAP baseline fixture pack and packaged baseline profile:
  - fixture manifest: `backend/tests/fixtures/audio_calibration/manifest.json`
  - packaged runtime baseline: `backend/app/assets/audio_event_calibration.json`
  - custom `AUDIO_EVENT_CALIBRATION_PATH` still overrides the packaged baseline when present
- The checked-in CLAP fixture pack has now been expanded to cover:
  - `meeting_room_speech`
  - `meeting_with_applause`
  - `broadcast_playback`
  - `software_demo_narration`
  - `voiceover_no_music`
  - `voiceover_with_music`
- Live Strix Halo validation of the expanded synthetic pack showed a hard limitation:
  - raw CLAP-only classification still collapsed the synthetic meeting, broadcast, and software-demo clips toward `podcast_voiceover`
  - the broader auto-generated candidate profile did not outperform the conservative voice-over baseline enough to become the default packaged profile
  - the larger pack is still useful as a benchmark set and for future model/prompt work
- Post-benchmark idle validation remained clean on Strix Halo:
  - after processing and two direct summary comparisons, the GPU returned to `0%` use at roughly `608-609 MHz`

## High Priority Requirements

- Keep AMD/Strix Halo fully functional on ROCm unless CPU execution is benchmarked to be equally fast for the exact stage and model in question.
- Standardize the Strix Halo host baseline around a known-good ROCm/kernel combination and verify it during live acceptance so ROCm regressions are caught outside the app code too.
- Replace the current fixed model-loading behavior with a dynamic residency planner that:
  - detects GPU count, memory, and topology automatically
  - determines whether models can remain resident instead of unloading between stages
  - respects a configurable memory ceiling so the runtime stays under a user-defined VRAM or unified-memory budget
  - decides when models should be pinned to one GPU, shared across GPUs, or split across GPUs
- Reduce AMD cold-start overhead for large embedding models so the final embedding stage does not dominate batch tail latency.
- Keep live validation on the Strix Halo as part of the acceptance bar, including small real video fixtures.

## Active Functional Work

- Replace automatic transcription fallback with an explicit ASR model switcher or selector so one chosen ASR path runs deterministically instead of silently falling back to Whisper.
- Add full NVIDIA runtime support after the AMD ROCm path is solid.
- Add macOS Metal support after NVIDIA.
- Turn Ask into a true chat workflow using the summarization model as the response engine.
- Add model management UX in Settings:
  - dropdown model selection
  - Hugging Face existence verification before add or save
  - add-model flow with an explicit validation action

## Product And Pipeline Enhancements

- Replace or supplement the synthetic CLAP fixture pack with cleaner real-world labeled audio clips for meeting-room speech, broadcast playback, and software-demo narration so the benchmark reflects actual deployment acoustics.
- Let the summarization model reconcile CLAP-derived audio hints with transcript, slides, and vision context, but do not rely on the transcription model alone for audio-scene hints because ASR text drops nonverbal audio information.
- Keep benchmarking whether CLAP-derived audio context materially improves summaries enough to justify the extra prompt and latency cost on Strix Halo and NVIDIA.
- Incremental recomputation with step-level caching and lineage or versioning for transcripts, frames, vision metadata, and summaries.
- Multi-index retrieval routing (transcript, slides, vision) with reranking for better long-video QA.
- Optional live transcription plus audio capture or mixing mode for real-time meetings and demos.
- Log clarity: show the actual ASR backend instead of the generic `whisper` label.
- Ensure UI and logs reflect Canary when the ASR backend is Canary.
