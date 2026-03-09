# Active Requirements And Backlog

## High Priority

- Make AMD/Strix Halo fully functional on ROCm unless CPU execution is benchmarked to be equally fast for that exact stage and model.
- Replace the current fixed model-loading behavior with a dynamic residency planner that:
  - detects GPU count, memory, and topology automatically
  - determines whether models can remain resident instead of unloading between stages
  - respects a configurable memory ceiling so the runtime stays under a user-defined VRAM/unified-memory budget
  - decides when models should be pinned to one GPU, shared across GPUs, or split across GPUs
- Reduce AMD cold-start overhead for large embedding models so the final embedding stage does not dominate batch tail latency.
- Keep live validation on the Strix Halo as part of the acceptance bar, including small real video fixtures.

## Next Functional Work

- Add full NVIDIA runtime support after the AMD/ROCm path is solid.
- Add macOS/Metal support after NVIDIA.
- Turn Ask into a true chat workflow using the summarization model as the response engine.
- Add model management UX in Settings:
  - dropdown model selection
  - Hugging Face existence verification before add/save
  - add-model flow with an explicit validation action

## Product And Pipeline Enhancements

- Incremental recomputation with step-level caching and lineage/versioning for transcripts, frames, vision metadata, and summaries.
- Multi-index retrieval routing (transcript + slides + vision) with reranking for better long-video QA.
- Optional live transcription + audio capture/mixing mode for real-time meetings and demos.
- Log clarity: show the actual ASR backend (for example Canary) instead of the generic "whisper" label.
- Ensure UI/logs reflect Canary when ASR backend is Canary.
