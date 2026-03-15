# EchOnyx Backend

Backend service for the video summarization system.

## Features

- Video transcription with selectable ASR models
- Speaker diarization with pyannote-audio when `HF_TOKEN` is configured
- Vision analysis with ROCm-backed OpenAI-compatible endpoints
- Summarization with Qwen3
- Vector search with ChromaDB
- Duplicate detection and suppression metadata during processing

## Development

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn app.main:app --reload

# Run Celery worker
uv run celery -A app.workers.celery_app worker --loglevel=info
```

## Notes

- Completed videos require an explicit forced rerun; reset/reprocess is not implicit anymore.
- Duplicate scoring is computed after summarization and before embedding so suppressed duplicates do not get indexed as separate search representatives.
- If `HF_TOKEN` is unset, diarization is skipped and the pipeline continues without speaker labels instead of failing the whole job.
- The default Compose stack keeps PostgreSQL and Redis internal-only; they are not published on host ports unless you add an override.
- The runtime planner now uses current free GPU memory plus NVIDIA topology data when choosing preferred worker and endpoint placement.
- CUDA worker-side models now load on the planner-selected device, and local `llama.cpp` endpoint models use the planner-selected CUDA placement when possible.
- The NVIDIA Docker path now builds from `backend/Dockerfile.cuda` with CUDA PyTorch wheels, CUDA `llama.cpp`, and NeMo enabled by default.
- The NVIDIA Compose override now uses `gpus: all` so backend and worker actually see every visible NVIDIA GPU under normal Docker Compose.
- The NVIDIA worker currently uses Celery `--pool=solo`, and the NVIDIA Compose override now routes vision/summarization through managed endpoint runtimes instead of the in-process worker path.
- On NVIDIA, leave `CUDA_VISIBLE_DEVICES` unset unless you intentionally want to hide GPUs from the planner; an empty string hides all CUDA devices. When it is unset, local `llama.cpp` loads now narrow visibility to the planner-selected GPUs before first import.
- On heterogeneous NVIDIA hosts, `NVIDIA_VISION_VISIBLE_DEVICES` and `NVIDIA_SUMMARIZATION_VISIBLE_DEVICES` can pin the endpoint runtimes to specific cards. When they are unset, the managed NVIDIA endpoints auto-pick GPUs from current free memory and fall back to stage-by-stage endpoint loading on single smaller GPUs.
- The CUDA image now smoke-builds on the live `ai-server`; when `docker build` has no visible GPU, set `CUDA_ARCHITECTURES` explicitly for your target cards. The live `3090 + RTX PRO 6000 Blackwell` mix was validated with `86;120`.
- The CUDA image now uses the `llama-cpp-python` vendored `llama.cpp` by default; only set an external `LLAMA_CPP_REPO` / `LLAMA_CPP_REF` when intentionally testing a specific upstream checkout.
- Current live NVIDIA split: summarization stays on CUDA `llama.cpp` and runs on a pinned `3090`; vision uses official `vLLM 0.11.2` on the pinned `RTX PRO 6000` because direct CUDA `llama.cpp` on Blackwell was not stable enough for `Qwen3VL`.
- Live `ai-server` acceptance now covers single upload, batch upload, summary retrieval, search, ask, and similar on that mixed NVIDIA split.
- Summary generation and `ask` answers now strip `<think>...</think>` reasoning blocks before persistence or API response.
- The audio-event step now reads extracted WAV files directly before CLAP scoring, so CUDA deployments do not depend on `torchaudio` + `torchcodec` just to build summary-side audio hints.
- Audio-event classification is fail-soft: if it breaks, summarization continues with empty audio context instead of failing the whole job.
- For the current model set, plan around `24 GB` free accelerator memory as the rough floor, `32 GB` free as the practical single-accelerator target, about `50.5 GB` of budget for warm worker models plus one endpoint, and about `100 GB` free if you expect the whole stack to stay resident at the default memory fraction.
