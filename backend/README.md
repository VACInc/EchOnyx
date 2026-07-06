# EchOnyx Backend

Backend service for the video summarization system.

## Features

- Video transcription with selectable ASR models
- Speaker diarization with pyannote-audio when `HF_TOKEN` is configured
- Vision analysis with ROCm-backed OpenAI-compatible endpoints
- Summarization with Qwen3
- Vector search with ChromaDB
- Chat-style follow-up Q&A on top of retrieval
- Duplicate detection and suppression metadata during processing
- Settings-side model verification and add flow for built-in entries and Hugging Face model ids

## Development

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn app.main:app --reload

# Run Celery worker
uv run celery -A app.workers.celery_app worker --loglevel=info
```

## Auth

- The backend now expects a single local admin password.
- First-use setup happens through `POST /api/auth/setup` or the frontend sign-in gate.
- First-use setup is localhost-only by default. Remote bootstrap should use a preseeded `AUTH_PASSWORD_HASH` or OIDC config.
- Later access uses `POST /api/auth/login`, `POST /api/auth/logout`, and `POST /api/auth/password`.
- Session auth is cookie-based; mutating requests require the matching CSRF header.
- Non-loopback HTTP auth is blocked by default. Use HTTPS, or set `ALLOW_INSECURE_AUTH_HTTP=true` only for temporary dev/test environments.
- OIDC is also supported for providers like Authentik.
- Set `OIDC_ENABLED=true` plus `OIDC_ISSUER_URL`, `OIDC_CLIENT_ID`, and `OIDC_CLIENT_SECRET` to turn it on.
- Optional `OIDC_ALLOWED_EMAILS` and `OIDC_ALLOWED_GROUPS` restrict which IdP users can create a local session.
- The backend exposes `GET /api/auth/oidc/login` and `GET /api/auth/oidc/callback`; the frontend sign-in gate uses those automatically when OIDC is enabled.
- Only enable `TRUST_PROXY_HEADERS=true` when the app is actually behind a trusted reverse proxy and `TRUSTED_PROXY_CIDRS` is set correctly.

## Apple Silicon / Metal Bring-Up

Docker does not expose Metal to the Linux containers used by this repo, so Apple Silicon runs are host-only for now.

Use a smaller sequential stack on a `16 GB` Mac:

```bash
export HARDWARE_PROFILE=apple_silicon
export GPU_BACKEND=metal
export MODEL_LOADING=sequential
export WHISPER_MODEL=small
export EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
export VISION_MODEL=Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf
export VISION_MMPROJ=Qwen2.5-VL-3B-Instruct.mmproj-fp16.gguf
export VISION_CHAT_FORMAT=qwen2.5-vl
export SUMMARIZATION_MODEL=Qwen2.5-3B-Instruct.Q4_K_M.gguf
export UPLOAD_DIR=$PWD/data/uploads
export MODEL_CACHE_DIR=$PWD/data/models
export CHROMA_PERSIST_DIR=$PWD/data/chroma
```

Install `llama-cpp-python` with Metal enabled before you start the backend or worker:

```bash
CMAKE_ARGS="-DGGML_METAL=on" uv sync
```

Start the host worker with `--pool=solo` on Apple Silicon:

```bash
uv run celery -A app.workers.celery_app worker --pool=solo --concurrency=1 --loglevel=info
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
- Apple Silicon host runs currently need Celery `--pool=solo`; the default prefork worker path stalled during the first real Metal validation run.
- On NVIDIA, leave `CUDA_VISIBLE_DEVICES` unset unless you intentionally want to hide GPUs from the planner; an empty string hides all CUDA devices. When it is unset, local `llama.cpp` loads now narrow visibility to the planner-selected GPUs before first import.
- On heterogeneous NVIDIA hosts, `NVIDIA_VISION_VISIBLE_DEVICES` and `NVIDIA_SUMMARIZATION_VISIBLE_DEVICES` are role-specific pins. A summary service now ignores the vision pin and vice versa. When they are unset, the managed NVIDIA endpoints auto-pick the emptiest GPU that can fit the requested model and fall back to stage-by-stage endpoint loading on single smaller GPUs.
- The runtime planner reports installed accelerator memory separately from the current free-memory budget; the budget is intentionally based on currently free memory, not raw installed VRAM.
- Vision and summarization endpoint warmup is now bounded; if an endpoint keeps returning `503 Loading model` for too long, the worker stops waiting and surfaces the failure instead of hanging for many minutes.
- The CUDA image now smoke-builds on the NVIDIA validation host; when `docker build` has no visible GPU, set `CUDA_ARCHITECTURES` explicitly for your target cards. The validated `3090 + RTX PRO 6000 Blackwell` mix used `86;120`.
- The CUDA image now uses the `llama-cpp-python` vendored `llama.cpp` by default; only set an external `LLAMA_CPP_REPO` / `LLAMA_CPP_REF` when intentionally testing a specific upstream checkout.
- Current live NVIDIA split: summarization stays on CUDA `llama.cpp` and runs on a pinned `3090`; vision uses official `vLLM` on the pinned `RTX PRO 6000` because direct CUDA `llama.cpp` on Blackwell was not stable enough for `Qwen3VL`. The default NVIDIA vision image now tracks `v0.17.1` so newer Qwen families like `Qwen3.5` are recognized.
- Live NVIDIA-host acceptance now covers single upload, batch upload, summary retrieval, search, ask, and similar on that mixed NVIDIA split.
- `scripts/acceptance.sh` is the operator entry point for repeatable API-level acceptance on live targets and the local Mac path; it now also checks runtime settings/hardware endpoints and the action-items CRUD/filter flow.
- Summary generation and `ask` answers now strip `<think>...</think>` reasoning blocks before persistence or API response.
- `/api/search/ask` now accepts optional conversation history so follow-up questions can reuse prior turns while staying grounded in retrieved context.
- `/api/settings/models/verify` checks a candidate model against the built-in catalog, GGUF registry, or Hugging Face before the UI adds it to a selector.
- `/api/settings/models/recommendations` and `/api/settings/models/download` support guided model setup; set `MODEL_AUTO_DOWNLOAD=false` to make workers fail clearly until models are downloaded from Settings.
- `/api/action-items` now provides first-class todo CRUD with video-label filters so summary action items and manual follow-ups can be managed outside the raw summary payload.
- The security layer now includes single-admin auth, CSRF on mutating routes, upload/write/login rate limits, JSON body caps, endpoint/model validation, and audit-log retention cleanup.
- Browser-origin access is no longer wildcard-open by default: CORS and job WebSocket access now trust explicit origins plus local/private-network browser origins unless you override that in env.
- Single uploads now enforce the configured size cap while streaming and reject media that fails ffprobe validation instead of storing arbitrary blobs on disk.
- Summary responses now expose only slide image filenames, not absolute server filesystem paths.
- The audio-event step now reads extracted WAV files directly before CLAP scoring, so CUDA deployments do not depend on `torchaudio` + `torchcodec` just to build summary-side audio hints.
- Audio-event classification is fail-soft: if it breaks, summarization continues with empty audio context instead of failing the whole job.
- Similar-video reranking now leans more heavily on transcript and key-point overlap so generic narrated videos do not outrank truly related ones as easily.
- Embedding indexing now sanitizes slide/topic metadata to Chroma-safe scalar values before insert so one malformed payload does not fail the job at the final embedding stage.
- For the current model set, plan around `24 GB` free accelerator memory as the rough floor, `32 GB` free as the practical single-accelerator target, about `50.5 GB` of budget for warm worker models plus one endpoint, and about `100 GB` free if you expect the whole stack to stay resident at the default memory fraction.
- `scripts/acceptance.sh` now supports secured deployments via `ECHONYX_PASSWORD` or `--password`.
