# EchOnyx Backend

Backend service for the video summarization system.

## Features

- Video transcription with selectable ASR models
- Speaker diarization with pyannote-audio
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
- The runtime planner now uses current free GPU memory plus NVIDIA topology data when choosing preferred worker and endpoint placement.
- For the current model set, plan around `24 GB` free as the rough NVIDIA floor, `32 GB` free as the practical single-GPU target, and about `100 GB` free if you expect the whole stack to stay resident on one GPU.
