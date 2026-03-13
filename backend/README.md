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
- On NVIDIA, leave `CUDA_VISIBLE_DEVICES` unset unless you intentionally want to hide GPUs from the planner; an empty string hides all CUDA devices.
- The CUDA image now smoke-builds on the live `ai-server`; when `docker build` has no visible GPU, set `CUDA_ARCHITECTURES` explicitly for your target cards. The live `3090 + RTX PRO 6000 Blackwell` mix was validated with `86;120`.
- For the current model set, plan around `24 GB` free accelerator memory as the rough floor, `32 GB` free as the practical single-accelerator target, about `50.5 GB` of budget for warm worker models plus one endpoint, and about `100 GB` free if you expect the whole stack to stay resident at the default memory fraction.
