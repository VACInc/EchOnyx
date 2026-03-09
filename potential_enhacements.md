# Active Requirements And Backlog

## High Priority

- Make AMD/Strix Halo fully functional on ROCm unless CPU execution is benchmarked to be equally fast for that exact stage and model.
- Standardize the Strix Halo host baseline around a known-good ROCm/kernel combination and verify it during live acceptance so ROCm regressions are caught outside the app code too.
- Replace the current fixed model-loading behavior with a dynamic residency planner that:
  - detects GPU count, memory, and topology automatically
  - determines whether models can remain resident instead of unloading between stages
  - respects a configurable memory ceiling so the runtime stays under a user-defined VRAM/unified-memory budget
  - decides when models should be pinned to one GPU, shared across GPUs, or split across GPUs
- Reduce AMD cold-start overhead for large embedding models so the final embedding stage does not dominate batch tail latency.
- Keep live validation on the Strix Halo as part of the acceptance bar, including small real video fixtures.

## Next Functional Work

- Replace automatic transcription fallback with an explicit ASR model switcher/selector so one chosen ASR path runs deterministically instead of silently falling back to Whisper.
- Add full NVIDIA runtime support after the AMD/ROCm path is solid.
- Add macOS/Metal support after NVIDIA.
- Turn Ask into a true chat workflow using the summarization model as the response engine.
- Add model management UX in Settings:
  - dropdown model selection
  - Hugging Face existence verification before add/save
  - add-model flow with an explicit validation action

## Product And Pipeline Enhancements

- Replace the current audio-event classifier with a CLAP-based raw-audio hint stage so meeting/broadcast/podcast/demo-style cues come from the audio itself rather than only a closed AudioSet label map.
- Let the summarization model reconcile CLAP-derived audio hints with transcript/slides/vision context, but do not rely on the transcription model alone for audio-scene hints because ASR text drops nonverbal audio information.
- If audio-event classification stays, benchmark candidate models on Strix Halo ROCm and NVIDIA, including accuracy on small real video fixtures plus latency/VRAM impact.
- Incremental recomputation with step-level caching and lineage/versioning for transcripts, frames, vision metadata, and summaries.
- Multi-index retrieval routing (transcript + slides + vision) with reranking for better long-video QA.
- Optional live transcription + audio capture/mixing mode for real-time meetings and demos.
- Log clarity: show the actual ASR backend (for example Canary) instead of the generic "whisper" label.
- Ensure UI/logs reflect Canary when ASR backend is Canary.
