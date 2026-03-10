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

- Replace the current audio-event classifier with a CLAP-based raw-audio hint stage so meeting, broadcast, podcast, and demo-style cues come from the audio itself rather than only a closed AudioSet label map.
- Let the summarization model reconcile CLAP-derived audio hints with transcript, slides, and vision context, but do not rely on the transcription model alone for audio-scene hints because ASR text drops nonverbal audio information.
- If audio-event classification stays, benchmark candidate models on Strix Halo ROCm and NVIDIA, including accuracy on small real video fixtures plus latency and VRAM impact.
- Incremental recomputation with step-level caching and lineage or versioning for transcripts, frames, vision metadata, and summaries.
- Multi-index retrieval routing (transcript, slides, vision) with reranking for better long-video QA.
- Optional live transcription plus audio capture or mixing mode for real-time meetings and demos.
- Log clarity: show the actual ASR backend instead of the generic `whisper` label.
- Ensure UI and logs reflect Canary when the ASR backend is Canary.
