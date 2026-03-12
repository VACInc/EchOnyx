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
- The checked-in CLAP fixture pack now distinguishes validated calibration fixtures from exploratory real clips:
  - validated calibration path:
    - `voiceover_no_music`
    - `voiceover_with_music`
    - `broadcast_weather_radio`
    - `applause_real`
  - exploratory fixtures kept out of default calibration for now:
    - `meeting_room_real`
    - `software_demo_real`
- Live Strix Halo validation on March 10, 2026 established:
  - the real NOAA weather-radio clip separated correctly as `broadcast playback`
  - the real applause clip separated correctly as a `crowd_applause` supporting cue
  - the current real meeting and software-demo clips still collapsed toward produced narration in raw CLAP audio-only classification
  - those exploratory real clips remain useful for benchmarking and future model or prompt work, but should not tune the default calibration path yet
- Post-benchmark idle validation remained clean on Strix Halo:
  - after processing and two direct summary comparisons, the GPU returned to `0%` use at roughly `608-609 MHz`
- Live Strix Halo duplicate handling is now in place:
  - duplicate policy and thresholds are configurable through `/api/settings` and the Settings UI
  - completed videos reject accidental reruns unless `force=true` is used explicitly
  - exact duplicate uploads are marked in `videos.duplicate_info`, skip standalone indexing, and are suppressed from default search results
  - explicit `video_id` or `video_ids` search and ask requests still allow targeting suppressed duplicates directly
- Live Strix Halo validation on March 12, 2026 confirmed:
  - two repeated uploads of the same spoken probe were both classified as `exact_duplicate` with `score: 1.0`
  - default search for that probe content returned only the representative video after suppression
  - `/api/videos/{id}` now reports an active rerun as `queued` or `processing` instead of incorrectly preferring an older completed job
- Live `ai-server` planner validation on March 12, 2026 confirmed:
  - the machine exposes `1x RTX PRO 6000 Blackwell Workstation Edition` plus `6x RTX 3090`
  - the 6000 had about `97 GB` free and is the correct first-choice placement when the current model set fits on one GPU
  - the planner now uses current free memory and `nvidia-smi topo -m` data, not only static total VRAM
  - CUDA worker-side models now honor the planner-selected device index, and local `llama.cpp` models can use the planner-selected CUDA main GPU or tensor split
  - the NVIDIA Docker path now has a dedicated CUDA backend image with CUDA PyTorch wheels, CUDA `llama.cpp`, and NeMo enabled by default
  - `CUDA_VISIBLE_DEVICES` should stay blank by default on NVIDIA so the planner can see all visible GPUs
  - the CUDA backend image now smoke-builds successfully on that host after switching the image to a venv install, linking against CUDA driver stubs during `docker build`, and setting `CUDA_ARCHITECTURES=86;120`
  - the three NVLink-connected 3090 pairs are now recorded as fallback placement candidates when the 6000 cannot fit the active set
  - current accelerator sizing guidance for the shipped model set is:
    - rough floor: about `24 GB` free
    - practical single-accelerator target: about `32 GB` free
    - warm worker models plus one local endpoint: about `50.5 GB` of budget
    - single-accelerator fully resident target: about `100 GB` free at the default memory fraction
  - full live CUDA deployment and end-to-end acceptance are still pending, but the backend CUDA image build itself is now validated

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

- Keep sourcing better real meeting-room and software-demo fixtures until CLAP can separate them cleanly enough to promote them from exploratory to validated calibration inputs.
- Let the summarization model reconcile CLAP-derived audio hints with transcript, slides, and vision context, but do not rely on the transcription model alone for audio-scene hints because ASR text drops nonverbal audio information.
- Keep benchmarking whether CLAP-derived audio context materially improves summaries enough to justify the extra prompt and latency cost on Strix Halo and NVIDIA.
- Incremental recomputation with step-level caching and lineage or versioning for transcripts, frames, vision metadata, and summaries.
- Multi-index retrieval routing (transcript, slides, vision) with reranking for better long-video QA.
- Optional live transcription plus audio capture or mixing mode for real-time meetings and demos.
- Log clarity: show the actual ASR backend instead of the generic `whisper` label.
- Ensure UI and logs reflect Canary when the ASR backend is Canary.
