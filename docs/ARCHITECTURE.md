# Architecture

EchOnyx is a local-first video and presentation intelligence app. The default deployment keeps media, metadata, embeddings, and model files on the operator's machine or self-managed hosts.

## Components

| Component | Role |
|---|---|
| Next.js frontend | Browser UI for sign-in, uploads, videos, summaries, search/ask, todos, jobs, batches, and settings. |
| FastAPI backend | API surface, auth/session handling, settings, upload validation, metadata reads/writes, and orchestration entry points. |
| Celery worker | Long-running video processing pipeline and batch execution. |
| PostgreSQL | Durable relational metadata for videos, jobs, batches, sessions, audit logs, and action items. |
| Redis | Celery broker/result backend and rate-limit state when configured. |
| ChromaDB | Local vector persistence for transcript, summary, slide, and vision retrieval. |
| Managed model endpoints | Local OpenAI-compatible vision and summarization runtimes for ROCm or NVIDIA Compose paths. |

## Processing Pipeline

Uploads are stored under the configured upload directory and then processed asynchronously by Celery.

```text
upload validation
  -> audio extraction
  -> ASR transcription
  -> optional diarization
  -> transcript merge
  -> scene/keyframe/slide extraction
  -> vision analysis
  -> optional audio-event classification
  -> summarization
  -> duplicate evaluation
  -> embedding/indexing
```

If `HF_TOKEN` is missing, diarization is skipped and the rest of the pipeline continues. Audio-event classification is supporting context only; if it fails, summarization continues with empty audio context. Duplicate suppression can keep exact duplicates out of default search results while still allowing direct video-targeted search and Ask requests.

## Runtime Planner

The runtime planner builds a plan from the hardware profile, visible accelerators, current free memory, configured memory fraction, and optional memory ceiling. Settings exposes installed accelerator memory separately from the effective free-memory budget.

The plan controls:

- worker model loading: sequential or parallel
- endpoint loading: resident, one-at-a-time, or idle teardown
- preferred worker devices
- preferred endpoint devices
- per-model placement for vision and summarization
- whether the current hardware can keep worker models or endpoint models resident

Profile behavior is intentionally conservative:

- Strix Halo resolves to ROCm and rejects CPU/Vulkan fallback for that profile.
- Apple Silicon resolves to Metal and uses small host-run defaults with sequential loading.
- CPU-only stays sequential and is intended for small models, short clips, and functional tests.
- NVIDIA uses current `nvidia-smi` free-memory data and topology hints when available.

## Managed Endpoint Runtimes

### ROCm

The AMD Compose override runs internal-only OpenAI-compatible endpoint containers for vision and summarization.

- Default: `ROCM_LLM_RUNTIME=llama_server`, backed by AMD ROCm `llama.cpp`, with managed idle teardown.
- Experimental: `ROCM_LLM_RUNTIME=vllm` with `INSTALL_VLLM=1`, backed by vLLM's OpenAI server path.

The backend talks to those containers through `VISION_ENDPOINT_URL` and `SUMMARIZATION_ENDPOINT_URL` inside the Compose network.

### NVIDIA

The NVIDIA Compose override uses a split managed endpoint design:

- Vision runs through an official vLLM OpenAI container.
- Summarization runs through CUDA `llama.cpp` behind the same managed runtime wrapper.

When role-specific pins such as `NVIDIA_VISION_VISIBLE_DEVICES` and `NVIDIA_SUMMARIZATION_VISIBLE_DEVICES` are unset, the endpoint runtime can choose devices from current free-memory data. On smaller single-GPU systems, endpoints can switch to stage-by-stage loading instead of keeping both large models hot.

## Auth Model

EchOnyx uses a single-admin auth model for 1.0.

- First-run password setup is allowed only from localhost unless `AUTH_PASSWORD_HASH` is preseeded or OIDC is configured.
- Sessions are cookie-based.
- Mutating routes require a matching CSRF header.
- OIDC login can create the same local session used by password login.
- Reverse-proxy headers are ignored unless `TRUST_PROXY_HEADERS=true` and `TRUSTED_PROXY_CIDRS` are configured.
- Non-loopback HTTP auth is blocked by default.

Public health and readiness routes stay available for deployment checks. `/ready` returns 503 for
database, Redis, or Chroma failures; a missing worker heartbeat is reported as degraded with HTTP 200
by default so read-only API traffic can still be served. Use `/ready?strict=1` for "ready to process"
checks where any failing component, including the worker, must return 503.

## Data On Disk

Docker defaults:

- `./data:/data` stores uploads and local app artifacts.
- `model_cache:/data/models` stores downloaded models.
- `postgres_data` stores PostgreSQL data.
- `redis_data` stores Redis data.
- `vllm_cache` stores vLLM/Hugging Face cache data for managed endpoint images.
- `CHROMA_PERSIST_DIR=/data/chroma` stores ChromaDB data.

Host-run paths depend on the shell's working directory and environment variables. The Apple Silicon guide uses repo-local paths such as:

```bash
UPLOAD_DIR=$PWD/data/uploads
MODEL_CACHE_DIR=$PWD/data/models
CHROMA_PERSIST_DIR=$PWD/data/chroma
```

API responses should not expose absolute server paths. Summary slide images are returned as filenames, and duplicate metadata is sanitized before response.
