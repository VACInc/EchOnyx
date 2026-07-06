> ⚠️ **Work in Progress** — EchOnyx is under active development and not yet ready for general use. Expect breaking changes, incomplete features, and rough edges.

<p align="center">
  <img src="img/echonyx.png" alt="EchOnyx" width="720" />
</p>
<p align="center">
  <strong>EchOnyx:</strong> Local, privacy-first video and presentation intelligence that runs entirely on your hardware. Designed for long-form meetings, demos, and reviews where details matter.
</p>

## What EchOnyx Does

EchOnyx turns videos and recorded presentations into searchable, private knowledge. It runs locally by default, keeps your source media and derived artifacts on your own machine, and uses local or self-managed model endpoints unless you explicitly configure external OpenAI-compatible endpoints.

Current features:

- Transcribes speech with selectable ASR models.
- Adds speaker diarization when `HF_TOKEN` is configured and pyannote terms are accepted.
- Extracts scenes, frames, slides, and screen content for visual analysis.
- Produces structured summaries with key points, decisions, and action items.
- Supports semantic search and grounded question answering across uploaded videos.
- Keeps Ask-mode follow-up threads local to the browser session.
- Supports labels and label-scoped search/ask.
- Provides first-class todos/action items linked to source videos.
- Detects duplicates and can suppress exact duplicates from default search results.
- Provides retry, reset, cancel, and batch-processing controls.
- Exports summaries as Markdown, PDF, or JSON.
- Includes Settings-based model selection, verification, recommendations, and downloads.

## Quickstart

### Prerequisites

- Docker and Docker Compose for the Docker paths.
- `git`, `curl`, and `ffmpeg` for local checks and host-run paths.
- A Hugging Face account if you want gated models or pyannote diarization.
- Enough disk space for model downloads. The default GPU model set is multi-tens of GB.

### Tier 1: Docker + AMD ROCm / Strix Halo

This is the reference platform from the 1.0 readiness matrix.

```bash
git clone <repository-url>
cd EchOnyx
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d
```

The AMD override targets ROCm 7.2, sets `HARDWARE_PROFILE=strix_halo`, uses `GPU_BACKEND=rocm`, and runs managed OpenAI-compatible vision and summarization endpoints inside the Compose network. Strix Halo is treated as a ROCm-only profile; CPU and Vulkan fallbacks are rejected for that profile.

If your host uses different `/dev/dri` group ids, set `VIDEO_GROUP_ID` and `RENDER_GROUP_ID` in `.env`.

### Tier 1: Docker + NVIDIA CUDA

```bash
git clone <repository-url>
cd EchOnyx
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d
```

The NVIDIA override exposes all visible NVIDIA GPUs to the backend and worker, runs the worker with Celery `--pool=solo`, and uses managed endpoint services for vision and summarization. If you build on a host without a visible GPU during `docker build`, set `CUDA_ARCHITECTURES` for the target cards. The validation host used `86;120` for an `RTX 3090 + RTX PRO 6000 Blackwell` split.

By default, NVIDIA vision uses the official vLLM image and summarization uses CUDA `llama.cpp`. The runtime planner can auto-pick devices from live `nvidia-smi` free-memory data when explicit NVIDIA endpoint pins are unset.

### Tier 2: Apple Silicon Host Run

Docker does not expose Metal to the Linux containers used by this repo. Run the backend and worker directly on the Mac host.

Use small sequential defaults on smaller unified-memory Macs:

```bash
cd backend
export HARDWARE_PROFILE=apple_silicon
export GPU_BACKEND=metal
export MODEL_LOADING=sequential
export WHISPER_MODEL=small
export EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
export VISION_MODEL=Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf
export VISION_MMPROJ=Qwen2.5-VL-3B-Instruct.mmproj-fp16.gguf
export VISION_CHAT_FORMAT=qwen2.5-vl
export SUMMARIZATION_MODEL=Qwen2.5-3B-Instruct.Q4_K_M.gguf
```

See [backend/README.md](backend/README.md) for the host-run install, API, and Celery commands. The Apple path is meant for functional use with smaller models, not the full high-capacity default stack.

### Tier 2: CPU-Only Docker

CPU-only is supported for functional testing, short clips, and small models. It is not a practical way to run the default large model set.

```bash
git clone <repository-url>
cd EchOnyx
cp .env.example .env
HARDWARE_PROFILE=cpu_only GPU_BACKEND=cpu docker compose up -d
```

Before processing real media, open Settings -> Model Downloads and choose the small recommended set. Keep `MODEL_LOADING=sequential`. Expect long runtimes, especially for transcription, vision, and summarization.

### Tier 2: External Endpoints

You can keep EchOnyx local-first while offloading vision and/or summarization to OpenAI-compatible endpoints that you manage.

```env
VISION_ENDPOINT_URL=http://host.docker.internal:8081/v1
VISION_ENDPOINT_MODEL=Qwen3VL-32B-Instruct-Q4_K_M.gguf
SUMMARIZATION_ENDPOINT_URL=http://host.docker.internal:8080/v1
SUMMARIZATION_ENDPOINT_MODEL=Qwen3-30B-A3B-Q4_K_M.gguf
```

Use HTTPS or trusted private networking for remote endpoints. Settings rejects unsafe public HTTP endpoints and path-like custom model names.

### Experimental

These paths are available for testing but are not default support targets:

- ROCm vLLM runtime: set `ROCM_LLM_RUNTIME=vllm` and build with `INSTALL_VLLM=1`. Current Strix Halo findings show slow startup and teardown hazards, so `llama_server` remains the AMD default.
- NeMo / Canary / Granite ASR: set `INSTALL_NEMO=1` and select a Canary or Granite ASR model. These alternatives are not part of the default path.
- Vulkan: the enum remains for compatibility, but Strix Halo rejects Vulkan as unsupported.

## First Run

### Guided Setup

The setup script walks through the recommended path and points model setup through Settings:

```bash
scripts/setup.sh
```

It checks the environment, prepares local configuration, and prints health/readiness self-checks.

### Manual Setup

```bash
git clone <repository-url>
cd EchOnyx
cp .env.example .env
# Edit .env for your hardware, auth, model, and token choices.
docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d
```

For NVIDIA, replace the AMD override with `docker-compose.nvidia.yml`. For CPU-only, use the base Compose file with `HARDWARE_PROFILE=cpu_only` and `GPU_BACKEND=cpu`.

Open:

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Readiness: `http://localhost:8000/ready`

`/health` confirms the API process is alive. `/ready` checks database, Redis, and Chroma persistence dependencies and returns `503` with failing component names when a dependency is not ready.

### Sign-In Bootstrap

EchOnyx uses a single-admin auth model for 1.0.

- First local use creates the admin password through the sign-in gate or `POST /api/auth/setup`.
- First-run password creation is localhost-only by default.
- For remote first-run installs, preseed `AUTH_PASSWORD_HASH` or configure OIDC before opening the service remotely.
- OIDC uses the same local session cookies after login. Set `OIDC_ENABLED=true` plus the issuer, client id, and client secret values.
- Non-loopback HTTP auth is blocked by default. Use HTTPS for remote auth, or set `ALLOW_INSECURE_AUTH_HTTP=true` only for temporary development.

### Model Downloads

Open Settings -> Model Downloads after the app starts.

- The recommendations endpoint chooses a small or default set based on the current hardware profile and runtime plan.
- `HF_TOKEN` is optional for basic use, but required for pyannote diarization and gated Hugging Face models.
- For pyannote, create a Hugging Face token and accept the model agreements for `pyannote/speaker-diarization-community-1` and `pyannote/segmentation-3.0`.
- `MODEL_AUTO_DOWNLOAD=true` lets workers and managed endpoints fetch missing models implicitly on first use.
- `MODEL_AUTO_DOWNLOAD=false` makes missing models fail clearly until you explicitly download them from Settings.

## Common Tasks

### Upload And Batch

Use the upload control for one or more videos. Single uploads start processing immediately. Batch uploads create a batch record, enqueue each file, and expose batch status and cancellation.

The processing pipeline is:

```text
upload -> audio extraction -> transcription -> diarization -> transcript merge
  -> frame/slide extraction -> vision analysis -> summarization -> embeddings
```

If `HF_TOKEN` is absent, diarization is skipped and the rest of the pipeline continues.

### Search And Ask

Use Search to query transcripts, summaries, slides, and vision-derived context. Ask mode answers natural-language questions from retrieved local context and can use recent turns for follow-up questions.

### Labels

Add labels on videos and use them to filter search, Ask, and todos. The backend exposes existing labels with counts so the UI can suggest labels already in use.

### Todos

Promote summary action items into todos, add manual todos, complete them, remove them, and filter the global todo list by text, completion state, or video labels.

### Duplicates

Duplicate policy is configurable in Settings. Exact duplicates can be collapsed out of default search while still remaining accessible directly by `video_id` or explicit selection.

### Retry, Reset, And Cancel

- Retry resumes from the last failed step when possible.
- Reset/reprocess starts a fresh pipeline run.
- Completed videos do not rerun by accident; a forced reset or reprocess is required.
- Job and batch cancellation revoke queued or running Celery work when possible.

### Exports

Video summaries can be exported as Markdown, PDF, or JSON from the summary export endpoint and UI surfaces that call it.

### Delete

Deleting a video removes its database record, stored artifacts, and embeddings for that video.

## Configuration Reference

Set these in `.env`. Blank values generally mean "auto-detect" or "disabled" as noted.

### Core, Storage, And Services

| Variable | Meaning |
|---|---|
| `POSTGRES_USER` | PostgreSQL user for Compose. |
| `POSTGRES_PASSWORD` | PostgreSQL password for Compose. |
| `POSTGRES_DB` | PostgreSQL database name for Compose. |
| `DATABASE_URL` | Backend database URL; Compose sets this to the internal Postgres service. |
| `REDIS_URL` | Redis URL for Celery and rate-limit state; Compose sets this internally. |
| `UPLOAD_DIR` | Where uploaded media and derived upload artifacts live. |
| `MODEL_CACHE_DIR` | Where model files are cached. |
| `CHROMA_PERSIST_DIR` | Where ChromaDB vector data is persisted. |
| `HF_TOKEN` | Hugging Face token for gated downloads and pyannote diarization. |

### Hardware And Planning

| Variable | Meaning |
|---|---|
| `HARDWARE_PROFILE` | Optional profile override: `strix_halo`, `apple_silicon`, `rtx_5090`, `multi_gpu`, or `cpu_only`. |
| `GPU_BACKEND` | Optional backend override: `cuda`, `metal`, `vulkan`, `rocm`, or `cpu`. |
| `MODEL_LOADING` | `sequential` loads/unloads stages for lower memory; `parallel` keeps models warm when memory allows. |
| `GPU_MEMORY_FRACTION` | Fraction of accelerator memory treated as usable budget by the planner. |
| `RUNTIME_PLANNER_ENABLED` | Enables free-memory-based runtime placement when true. |
| `RUNTIME_MEMORY_CEILING_GB` | Optional memory ceiling for planner decisions. |
| `CUDA_VISIBLE_DEVICES` | CUDA visibility override; leave unset unless intentionally hiding devices. An empty value hides all CUDA devices. |
| `CUDA_ARCHITECTURES` | CUDA arch list for image builds, such as `86;120` for a mixed 3090 and RTX PRO 6000 host. |
| `VULKAN_DEVICE` | Vulkan device index for the deprecated/experimental Vulkan backend. |
| `VIDEO_GROUP_ID` | Host `video` group id used by the AMD Docker override for `/dev/dri`. |
| `RENDER_GROUP_ID` | Host `render` group id used by the AMD Docker override for `/dev/dri`. |

### Auth, Browser Origins, And OIDC

| Variable | Meaning |
|---|---|
| `CORS_ALLOWED_ORIGINS` | Comma-separated explicit browser origins to trust. |
| `CORS_ALLOW_ORIGIN_REGEX` | Override for the default local/private-network browser-origin regex. |
| `AUTH_REQUIRED` | Keep true unless intentionally running an unauthenticated local dev instance. |
| `AUTH_PASSWORD_HASH` | Optional preseeded local admin password hash for remote bootstrap or managed deployments. |
| `AUTH_SESSION_COOKIE_NAME` | Session cookie name. |
| `AUTH_CSRF_COOKIE_NAME` | CSRF cookie name. |
| `AUTH_SESSION_TTL_HOURS` | Local session lifetime. |
| `TRUST_PROXY_HEADERS` | Trust `X-Forwarded-*` headers only behind a trusted reverse proxy. |
| `TRUSTED_PROXY_CIDRS` | CIDRs allowed to provide trusted proxy headers. |
| `ALLOW_INSECURE_AUTH_HTTP` | Dev/emergency override for non-loopback HTTP auth; leave false in real deployments. |
| `OIDC_ENABLED` | Enables external OIDC login. |
| `OIDC_PROVIDER_NAME` | Display name for the OIDC provider. |
| `OIDC_ISSUER_URL` | OIDC issuer URL. |
| `OIDC_CLIENT_ID` | OIDC client id. |
| `OIDC_CLIENT_SECRET` | OIDC client secret. |
| `OIDC_SCOPES` | OIDC scopes, defaulting to `openid profile email`. |
| `OIDC_ALLOWED_EMAILS` | Optional email allowlist for OIDC logins. |
| `OIDC_ALLOWED_GROUPS` | Optional group allowlist for OIDC logins. |
| `OIDC_REDIRECT_URI` | Optional callback override when the default API callback is not correct. |
| `OIDC_FRONTEND_REDIRECT_URL` | Optional frontend redirect override after OIDC login. |
| `LOGIN_RATE_LIMIT_ATTEMPTS` | Login attempts allowed per window. |
| `LOGIN_RATE_LIMIT_WINDOW_SECONDS` | Login rate-limit window. |
| `WRITE_RATE_LIMIT_REQUESTS` | Mutating API requests allowed per window. |
| `WRITE_RATE_LIMIT_WINDOW_SECONDS` | Mutating API rate-limit window. |
| `UPLOAD_RATE_LIMIT_REQUESTS` | Upload requests allowed per window. |
| `UPLOAD_RATE_LIMIT_WINDOW_SECONDS` | Upload rate-limit window. |
| `MAX_JSON_REQUEST_BYTES` | Request-size ceiling for JSON write routes. |
| `AUDIT_LOG_RETENTION_DAYS` | Audit-log retention period. |

### Models

| Variable | Meaning |
|---|---|
| `WHISPER_MODEL` | Explicit ASR selector; there is no silent fallback. |
| `TRANSCRIPTION_FALLBACK_ENABLED` | Deprecated compatibility field; automatic ASR fallback is disabled. |
| `TRANSCRIPTION_FALLBACK_MODEL` | Deprecated compatibility field for the old fallback model. |
| `GRANITE_FORCE_CPU` | Forces Granite ASR to CPU, useful only if accelerator kernels crash. |
| `DIARIZATION_MODEL` | pyannote diarization model. |
| `VISION_MODEL` | Vision GGUF filename in `MODEL_CACHE_DIR`. |
| `VISION_MMPROJ` | Vision multimodal projector filename for local llama.cpp use. |
| `VISION_CHAT_FORMAT` | Chat format override for local vision models. |
| `VISION_ENDPOINT_URL` | External or managed OpenAI-compatible vision endpoint URL. |
| `VISION_ENDPOINT_MODEL` | Model name sent to the vision endpoint. |
| `VISION_ENDPOINT_API_KEY` | API key for the vision endpoint when needed. |
| `VISION_ENDPOINT_TIMEOUT_S` | Vision endpoint timeout. |
| `VISION_DEBUG` | Enables extra vision debug logging. |
| `VISION_MODEL_PATH` | Model path used by managed vision endpoint containers. |
| `VISION_MMPROJ_PATH` | Multimodal projector path used by managed vision endpoint containers. |
| `VISION_VLLM_MODEL_ID` | Hugging Face model id for vLLM vision runtime. |
| `VISION_CONTEXT_SIZE` | Context size for managed vision endpoint runtime. |
| `VISION_GPU_LAYERS` | GPU layer count for local or managed llama.cpp vision runtime. |
| `VISION_LLAMA_SERVER_EXTRA_ARGS` | Extra args for managed llama.cpp vision server. |
| `VISION_VLLM_EXTRA_ARGS` | Extra args for managed vLLM vision server. |
| `SUMMARIZATION_MODEL` | Summarization GGUF filename in `MODEL_CACHE_DIR`. |
| `SUMMARIZATION_ENDPOINT_URL` | External or managed OpenAI-compatible summarization endpoint URL. |
| `SUMMARIZATION_ENDPOINT_MODEL` | Model name sent to the summarization endpoint. |
| `SUMMARIZATION_ENDPOINT_API_KEY` | API key for the summarization endpoint when needed. |
| `SUMMARIZATION_ENDPOINT_TIMEOUT_S` | Summarization endpoint timeout. |
| `SUMMARIZATION_MODEL_PATH` | Model path used by managed summarization endpoint containers. |
| `SUMMARIZATION_VLLM_MODEL_ID` | Hugging Face model id for vLLM summarization runtime. |
| `SUMMARIZATION_CONTEXT_SIZE` | Context size for managed summarization endpoint runtime. |
| `SUMMARIZATION_GPU_LAYERS` | GPU layer count for local or managed llama.cpp summarization runtime. |
| `SUMMARIZATION_LLAMA_SERVER_EXTRA_ARGS` | Extra args for managed llama.cpp summarization server. |
| `SUMMARIZATION_VLLM_EXTRA_ARGS` | Extra args for managed vLLM summarization server. |
| `EMBEDDING_MODEL` | Hugging Face embedding model id. |
| `MODEL_AUTO_DOWNLOAD` | Allows implicit model downloads when true; set false to require Settings downloads first. |

### Audio Events

| Variable | Meaning |
|---|---|
| `AUDIO_EVENT_MODEL` | Audio hint model, defaulting to CLAP for raw-audio source cues. |
| `AUDIO_EVENT_CALIBRATION_PATH` | Optional JSON profile overriding CLAP prompts and thresholds. |
| `AUDIO_EVENT_SAMPLE_SECONDS` | Duration of each sampled audio window. |
| `AUDIO_EVENT_NUM_SAMPLES` | Number of audio windows sampled per video. |
| `AUDIO_EVENT_MIN_SCORE` | Minimum score for audio event hints. |
| `AUDIO_EVENT_DEBUG` | Enables audio-event debug logging. |

### Processing, Duplicates, And Actions

| Variable | Meaning |
|---|---|
| `MAX_VIDEO_LENGTH_HOURS` | Maximum accepted video duration. |
| `KEYFRAME_EXTRACTION_INTERVAL` | Keyframe extraction interval in seconds. |
| `VISUAL_CONTEXT_MAX_FRAMES` | Maximum frames included in visual context. |
| `VISUAL_CONTEXT_MAX_CHARS` | Character budget for visual context. |
| `MIN_SPEECH_DURATION` | Minimum speech segment duration for diarization. |
| `BATCH_CONCURRENT_JOBS` | Concurrent batch jobs; increase only when hardware can support it. |
| `MAX_UPLOAD_SIZE_GB` | Maximum upload size. |
| `JOB_STALE_MINUTES` | Age after which processing jobs are considered stale for recovery. |
| `DUPLICATE_DETECTION_POLICY` | Duplicate policy: `off`, `warn`, `collapse_exact`, or `collapse_probable`. |
| `DUPLICATE_EXACT_THRESHOLD` | Similarity threshold for exact duplicates. |
| `DUPLICATE_PROBABLE_THRESHOLD` | Similarity threshold for probable duplicates. |
| `ACTION_ITEMS_ENABLED` | Enables action-item/todo APIs and UI. |

### ROCm Runtime

| Variable | Meaning |
|---|---|
| `ROCM_LLM_RUNTIME` | `llama_server` for managed ROCm llama.cpp or `vllm` for experimental ROCm vLLM. |
| `ROCM_LLM_IDLE_TIMEOUT_S` | Idle teardown timeout for ROCm `llama_server` endpoints. |
| `INSTALL_VLLM` | Builds the heavier ROCm vLLM path into the AMD endpoint image. |
| `VLLM_INSTALL_METHOD` | ROCm vLLM install method: `wheel` or `source`. |
| `VLLM_WHEEL_EXTRA_INDEX_URL` | Extra ROCm wheel index for vLLM. |
| `ROCM_WHL_URL` | ROCm PyTorch wheel source for AMD builds. |
| `ROCM_TORCH_VERSION` | Torch version for AMD builds. |
| `ROCM_TORCHAUDIO_VERSION` | Torchaudio version for AMD builds. |
| `ROCM_TORCHVISION_VERSION` | Torchvision version for AMD builds. |
| `ROCM_DEV_IMAGE` | ROCm development image used to build managed endpoint runtime. |
| `AMD_LLAMA_PACKAGE_URL` | AMD ROCm llama.cpp server package used by managed endpoints. |
| `HSA_OVERRIDE_GFX_VERSION` | ROCm gfx override commonly needed on Strix Halo. |

### NVIDIA Runtime

| Variable | Meaning |
|---|---|
| `CUDA_WHL_URL` | CUDA PyTorch wheel index for NVIDIA builds. |
| `CUDA_TORCH_VERSION` | Torch version for NVIDIA builds. |
| `CUDA_TORCHAUDIO_VERSION` | Torchaudio version for NVIDIA builds. |
| `CUDA_TORCHVISION_VERSION` | Torchvision version for NVIDIA builds. |
| `LLAMA_BUILD_CUDA` | Enables CUDA llama.cpp builds in the NVIDIA backend image. |
| `INSTALL_NEMO` | Installs NeMo so Canary ASR can run in the NVIDIA image. |
| `NVIDIA_VISION_VISIBLE_DEVICES` | Optional role-specific GPU pin for the NVIDIA vision endpoint. |
| `NVIDIA_SUMMARIZATION_VISIBLE_DEVICES` | Optional role-specific GPU pin for the NVIDIA summarization endpoint. |
| `NVIDIA_ENDPOINT_IDLE_TIMEOUT_SECONDS` | Idle teardown timeout for managed NVIDIA endpoints. |
| `NVIDIA_VISION_VLLM_IMAGE` | vLLM image used for the NVIDIA vision endpoint. |
| `VISION_VLLM_GPU_MEMORY_UTILIZATION` | vLLM GPU memory utilization for the NVIDIA vision endpoint. |
| `VISION_ENDPOINT_ESTIMATED_MEMORY_GB` | Planner estimate for vision endpoint memory. |
| `SUMMARIZATION_ENDPOINT_ESTIMATED_MEMORY_GB` | Planner estimate for summarization endpoint memory. |
| `NVIDIA_ENDPOINT_HOT_SET_MEMORY_GB` | Planner estimate for keeping NVIDIA endpoint hot sets resident. |

## Hardware Sizing

Plan against free accelerator memory, not only installed VRAM or unified memory. Settings shows installed accelerator memory separately from the active free-memory budget.

Current sizing guidance for the shipped model set:

- Rough floor: about `24 GB` free to run the largest current stage sequentially.
- Practical single-accelerator target: about `32 GB` free.
- Warm worker-side models: about `26.5 GB` of budget.
- Warm worker-side models plus one local endpoint at a time: about `50.5 GB` of budget.
- Whole current stack resident on one accelerator: about `74.5 GB` of budget, or about `100 GB` free at `GPU_MEMORY_FRACTION=0.75`.

On multi-GPU systems, the planner prefers the emptiest accelerator that can fit the requested model set, then falls back to topology-aware spread.

## Acceptance Checks

Use `scripts/acceptance.sh` for repeatable API-level checks. It covers health, settings, hardware, model status, upload/batch, summary, search, ask, similar videos, and action-item CRUD/filter behavior.

```bash
# Local Apple Silicon functional pass
ECHONYX_PASSWORD='<admin-password>' \
scripts/acceptance.sh \
  --base-url http://127.0.0.1:8000 \
  --primary-fixture /Users/you/EchOnyx/tmp/mac-smoke/budget.mp4 \
  --secondary-fixture /Users/you/EchOnyx/tmp/mac-smoke/probe.mp4 \
  --search-query "budget review" \
  --ask-question "When is the budget review due?" \
  --ask-expects "Friday" \
  --run-batch

# NVIDIA validation host mixed-GPU pass
ECHONYX_PASSWORD='<admin-password>' \
scripts/acceptance.sh \
  --base-url http://192.0.2.147:8000 \
  --primary-fixture /srv/echonyx/tmp/live-fixtures/probe1.mp4 \
  --secondary-fixture /srv/echonyx/tmp/live-fixtures/probe2.mp4 \
  --search-query "budget review due Friday" \
  --ask-question "When is the budget review due?" \
  --ask-expects "Friday" \
  --run-batch

# Strix Halo non-disruptive health/models check
scripts/acceptance.sh --base-url http://192.0.2.178:8000 --read-only
```

Use `ECHONYX_PASSWORD` or `--password` for secured deployments. Do not put real passwords into shell history, docs, commits, or screenshots.

## Audio Calibration

Use fixture-driven calibration to generate an `AUDIO_EVENT_CALIBRATION_PATH` profile:

```bash
cd backend
uv run python -m app.core.audio_calibration \
  --manifest tests/fixtures/audio_calibration/manifest.json \
  --output app/assets/audio_event_calibration.json
```

Manifest entries can point to audio or video files:

```json
{
  "fixtures": [
    {
      "media_path": "/abs/path/to/demo-with-music.mp4",
      "expected_primary_key": "podcast_voiceover",
      "expected_supporting_keys": ["music_heavy"],
      "label": "demo_with_music",
      "use_for_calibration": true
    }
  ]
}
```

Relative `media_path` values resolve from the manifest directory. Set `use_for_calibration=false` for exploratory fixtures that should remain in the pack without tuning the packaged baseline.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Sign-in setup is blocked remotely | First-run password setup is localhost-only. Use local setup, preseed `AUTH_PASSWORD_HASH`, or configure OIDC. |
| Diarization is missing | Set `HF_TOKEN` and accept the pyannote model agreements. Without it, processing continues without speaker labels. |
| Model download fails | Confirm model names in Settings, free disk space under `MODEL_CACHE_DIR`, and Hugging Face token/license access. |
| CPU-only processing is too slow | Use Settings -> Model Downloads small recommendations, shorter clips, and `MODEL_LOADING=sequential`. |
| GPU out of memory | Use smaller models, external endpoints, lower memory ceilings, sequential loading, or dedicated endpoint pins. |
| NVIDIA build cannot detect GPU arch | Set `CUDA_ARCHITECTURES` explicitly for the target cards. |
| Managed endpoint stays in loading | Check endpoint logs, model cache contents, endpoint model names, and available accelerator memory. Startup retries are bounded and should fail rather than hang forever. |
| Jobs appear stuck | Check `/api/jobs`, worker logs, Redis connectivity, and stale job recovery. Use retry, reset, cancel, or cancel-orphaned controls as appropriate. |
| `/ready` returns 503 | Read the failing component names in the response and check DB, Redis, or Chroma persistence. |

## Security Notes

- EchOnyx is local-first, but browser access is not wildcard-open by default. CORS trusts explicit origins plus local/private-network browser origins unless overridden.
- Protected routes require session auth; mutating routes require a matching CSRF token.
- Auth attempts, uploads, and mutating API operations are rate-limited and audit-logged.
- Uploads enforce size limits while streaming and reject files that do not probe as valid video media.
- Summary slide paths are scrubbed to filenames before API response.
- Settings validates custom model names and endpoint URLs to avoid unsafe public HTTP endpoints and path-like model names.
- Keep secrets out of docs, commits, command output, screenshots, and uploaded artifacts.

## More Documentation

- [Backend host-run notes](backend/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Changelog](CHANGELOG.md)
- [Roadmap](ROADMAP.md)
- [1.0 readiness tracker](docs/1.0-READINESS.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT License - see [LICENSE](LICENSE).

## Acknowledgments

- faster-whisper
- pyannote-audio
- llama.cpp / llama-cpp-python
- Qwen models
- ChromaDB
- FastAPI, Celery, PostgreSQL, Redis, and Next.js
