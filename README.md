> ⚠️ **Work in Progress** — EchOnyx is under active development and not yet ready for general use. Expect breaking changes, incomplete features, and rough edges.

<p align="center">
  <img src="img/echonyx.png" alt="EchOnyx" width="720" />
</p>
<p align="center">
  <strong>EchOnyx:</strong> Local, privacy-first video and presentation intelligence that runs entirely on your hardware. Designed for long-form meetings, demos, and reviews where details matter.
</p>

## Features

- Speech-to-text transcription (high-accuracy models, configurable)
- Speaker diarization (who spoke when)
- Scene/keyframe extraction and visual analysis (slides + screen content)
- Structured summaries (executive summary, key points, actions, decisions)
- Semantic search + question answering across all content
- User labels (tags) and label-scoped search/ask
- Retry (resume) and Reset (reprocess) pipeline controls

## Quick Start (Docker)

### Prereqs

- Docker + Docker Compose
- 32GB RAM minimum (128GB recommended for Strix Halo)
- Hugging Face account + accepted pyannote terms (for diarization)

### 1) Configure

```bash
git clone <repository-url>
cd EchOnyx
cp .env.example .env
```

Update `.env` (at minimum):

```
HF_TOKEN=hf_your_token_here
```

### 2) Run

**AMD Strix Halo / ROCm:**
```bash
docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d
```

**NVIDIA:**
```bash
docker compose up -d
```

### 3) Access

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- FastAPI docs: `http://localhost:8000/docs`

## Configuration (Key Env Vars)

Set these in `.env` as needed:

- `GPU_BACKEND`: `cuda` | `vulkan` | `rocm` | `cpu`
- `MODEL_LOADING`: `sequential` (low memory) or `parallel`
- `HF_TOKEN`: required for pyannote diarization models
- `VISION_ENDPOINT_URL`, `VISION_ENDPOINT_MODEL`: use an external VL server
- `SUMMARIZATION_ENDPOINT_URL`, `SUMMARIZATION_ENDPOINT_MODEL`: use an external LLM server
- `ROCM_LLM_RUNTIME`: `llama_server` (managed idle teardown) or `vllm`
- `ROCM_LLM_IDLE_TIMEOUT_S`: idle shutdown for ROCm `llama_server` endpoints
- `INSTALL_VLLM=1`: opt-in build flag for the heavier ROCm `vLLM` image path
- `VLLM_INSTALL_METHOD`: `wheel` (official ROCm wheel index) or `source`
- `VISION_VLLM_MODEL_ID`, `SUMMARIZATION_VLLM_MODEL_ID`: Hugging Face model ids for the `vLLM` runtime
- `EMBEDDING_MODEL`: embedding model id (HF)
- `UPLOAD_DIR`, `MODEL_CACHE_DIR`: storage locations

## Models

Defaults are configured in `.env.example`. All models are swappable:

- **Transcription**: `WHISPER_MODEL` (explicit ASR selector; no silent fallback)
- **Diarization**: `DIARIZATION_MODEL` (pyannote)
- **Vision**: `VISION_MODEL` (GGUF) or `VISION_ENDPOINT_*`
- **Summarization**: `SUMMARIZATION_MODEL` (GGUF) or `SUMMARIZATION_ENDPOINT_*`
- **Embeddings**: `EMBEDDING_MODEL` (HF)
- **Audio Hints**: `AUDIO_EVENT_MODEL` (defaults to CLAP for raw-audio source cues)

GGUF models can be downloaded automatically via the built-in model downloader when needed.

## Operations Guide (Operator Focus)

### Upload + Processing
- Upload a video in the UI and processing starts immediately.
- Processing steps: audio extraction → transcription → diarization → transcript merge → frame extraction → vision analysis → summarization → embedding.

### Retry vs Reset
- **Retry**: resumes from the last successful step (idempotent).
- **Reset**: restarts the entire pipeline from scratch.

### Labels (Tags)
- Add labels on a video’s detail page.
- Use labels as filters in Search/Ask to target only those videos.

### Delete Videos
- Use the Delete button on a video detail page to remove the video and all associated data (artifacts + embeddings).

## Search & Q/A

Use the Search page to:

- Search transcripts and summaries
- Ask natural-language questions
- Apply label filters to narrow the scope

## Hardware Support

| Profile | Description | Model Loading |
|---------|-------------|---------------|
| Strix Halo | AMD APU with 128GB unified memory | Sequential (current default) |
| RTX 5090 | Single high-VRAM NVIDIA GPU (32GB+) | Parallel |
| Multi-GPU | Multiple NVIDIA GPUs | Parallel |
| CPU Only | Fallback for systems without GPU | Sequential |

Current AMD note:
- Strix Halo is treated as a ROCm-only profile; Vulkan and CPU fallbacks are rejected.
- The AMD Docker override now supports two ROCm LLM endpoint paths behind the same OpenAI-compatible URLs:
  - `llama_server`: AMD ROCm `llama.cpp`, managed with idle teardown
  - `vllm`: vLLM OpenAI server for ROCm (opt-in image build)
- The `vllm` path can load Hugging Face model ids directly while still serving the existing endpoint model names expected by the backend.
- The AMD Docker override still uses AMD's published ROCm `llama.cpp` server artifact for `gfx115X` and fails closed if ROCm cannot enumerate a supported device.
- Current AMD defaults target ROCm `7.2` for both the backend wheels and the dedicated GGUF server image.

## Active Requirements

- Strix Halo and other AMD systems must be fully functional on ROCm unless CPU execution is proven to be equally fast for the same stage/model.
- Model residency must become dynamic instead of hard-coded:
  - detect available GPUs, VRAM/unified memory, and topology automatically
  - determine whether all models can stay resident without unloading
  - support a configurable memory ceiling so the runtime keeps itself under a user-defined budget
  - decide whether models should be isolated, shared, or split across GPUs when the hardware supports it
- Cold-start penalties for large embedding models need to be reduced on AMD so batch tails do not stall behind first-load startup costs.

## Architecture (Brief)

```
Frontend (Next.js)
        |
Backend (FastAPI)
        |
Redis (queue) + Postgres (metadata) + ChromaDB (embeddings)
        |
Worker (Celery)
        |
Models (ASR, diarization, vision, LLM, embeddings)
```

## Troubleshooting (Short)

- **Diarization fails**: confirm `HF_TOKEN` and accepted pyannote terms.
- **Out of memory**: use `MODEL_LOADING=sequential`, smaller models, or external endpoints.
- **Model download errors**: verify model IDs/filenames in `.env` and registry.
- **Jobs stuck**: restart workers; stale job recovery will requeue.

## Development (Minimal)

For local development, you can still run via Docker. If you run services directly:

- Backend: `backend/` (FastAPI + Celery)
- Frontend: `frontend/` (Next.js)

Tests:
```bash
pytest backend/tests
```

## License

MIT License - see LICENSE.

## Acknowledgments

- faster-whisper
- pyannote-audio
- llama.cpp / llama-cpp-python
- Qwen models
