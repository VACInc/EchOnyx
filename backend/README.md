# EchOnyx Backend

Backend service for the video summarization system.

## Features

- Video transcription with faster-whisper
- Speaker diarization with pyannote-audio
- Vision analysis with Qwen3-Omni
- Summarization with Qwen3
- Vector search with ChromaDB

## Development

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn app.main:app --reload

# Run Celery worker
uv run celery -A app.workers.celery worker --loglevel=info
```
