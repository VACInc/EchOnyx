# Changelog

Reverse-chronological project milestones and validation notes. Operational plans and future work live in [ROADMAP.md](ROADMAP.md); support status lives in [docs/1.0-READINESS.md](docs/1.0-READINESS.md).

## 2026-07-06

- Added the 1.0 readiness audit, support matrix, and launch checklist.
- Confirmed support tiers: AMD Strix Halo ROCm Docker and NVIDIA CUDA Docker as Tier 1; Apple Silicon host-run, CPU-only, and external endpoints as Tier 2; ROCm vLLM, NeMo/Canary ASR, and Vulkan as experimental.
- Added deterministic backend test setup, dependency locks, CI, `/ready`, model-download guidance, and compose/default alignment work as part of the 1.0 push.

## 2026-03-16

- Added single-admin auth with bootstrap setup, login/logout, password rotation, local session cookies, and CSRF protection on mutating routes.
- Added base OIDC support for providers such as Authentik, including auth-code login, callback handling, allowlists, and reuse of local session cookies.
- Added auth, upload, and write rate limits; JSON request-size ceilings; settings-side custom endpoint/model validation; audit logging with retention cleanup.
- Hardened remote auth behavior: forwarded-header trust is opt-in, localhost bootstrap does not trust spoofed `X-Forwarded-For`, cross-origin public auth setup posts are rejected, and non-loopback HTTP auth is blocked by default.
- Stopped publishing AMD managed vision and summarization runtime ports on the host by default.

## 2026-03-15

- Narrowed browser CORS from wildcard-open to explicit origins plus local/private-network browser origins.
- Applied the same browser-origin check to job WebSocket access.
- Enforced upload size limits while streaming and rejected uploads that fail media probing.
- Scrubbed summary slide image responses so API clients receive filenames instead of absolute server filesystem paths.
- Added the Apple Silicon host-run path with smaller Metal defaults, repo-local `data/` paths, `soundfile`, and Celery `--pool=solo` guidance.
- Added Ask-mode follow-up chat, action-item todos, label-aware todo filters, and Settings model selectors with verify/add support.

## 2026-03-14

- Validated the mixed NVIDIA path on a dedicated NVIDIA host with single upload, batch upload, summary retrieval, search, ask, and similar-video checks.
- Added bounded endpoint startup retries so repeated `503 Loading model` responses fail clearly instead of hanging jobs.
- Sanitized Chroma metadata before embedding insert so nested slide/topic metadata cannot fail a job at the final indexing step.
- Stripped `<think>...</think>` reasoning blocks from generated summaries and Ask responses before persistence or API return.
- Exposed runtime plan details for worker execution mode, endpoint loading mode, per-model placement, and endpoint idle unloading.

## 2026-03-13

- Confirmed CUDA Whisper loads `Systran/faster-whisper-large-v3` correctly instead of the incompatible raw OpenAI snapshot layout.
- Made missing `HF_TOKEN` skip diarization cleanly so the rest of the pipeline can continue.
- Updated stale vision defaults to `Qwen3VL-32B-Instruct-Q4_K_M.gguf`.
- Switched the NVIDIA worker to Celery `--pool=solo` while local CUDA vision/summarization stability work continues.
- Routed NVIDIA vision through official vLLM on the large Blackwell card and summarization through CUDA `llama.cpp` on a 3090-class card.
- Made CUDA audio-event classification read extracted WAV files directly and fail soft so optional CLAP issues do not block summarization.

## 2026-03-12

- Validated duplicate handling on Strix Halo: repeated uploads were classified as exact duplicates, default search returned one representative, and explicit video-targeted search/ask remained possible.
- Fixed video status reporting so active reruns show as queued or processing instead of being hidden by older completed jobs.
- Validated NVIDIA planner behavior on a mixed multi-GPU host: current free memory and topology now guide placement instead of static total VRAM alone.
- Added the CUDA backend image with CUDA PyTorch wheels, CUDA `llama.cpp`, NeMo support, and vendored `llama.cpp` as the default build path.
- Updated the NVIDIA Compose override to use `gpus: all`, keep PostgreSQL and Redis internal-only, and preserve unset `CUDA_VISIBLE_DEVICES` by default.
- Recorded sizing guidance for the shipped model set: roughly `24 GB` free as the floor, `32 GB` as a practical single-accelerator target, `50.5 GB` for warm worker models plus one endpoint, and about `100 GB` free for the whole stack at the default memory fraction.

## 2026-03-10

- Validated the Strix Halo ROCm `llama_server` path with managed idle teardown; post-run idle checks returned the GPU to `0%` use at roughly `607-609 MHz`.
- Kept ROCm vLLM as experimental after finding slow startup, memory contention, FP8 model incompatibility on `gfx1151`, and teardown hazards.
- Added CLAP audio-event context as structured supporting input to summarization.
- Benchmarked audio context on a narrated demo clip: audio context added narration evidence and one extra key point, with higher latency than the no-audio comparison.
- Added support-scoring calibration so the narrated benchmark emitted both produced narration and noticeable music-bed hints.
- Added a checked-in CLAP fixture pack and packaged baseline profile at `backend/tests/fixtures/audio_calibration/` and `backend/app/assets/audio_event_calibration.json`.
- Validated real weather-radio and applause fixtures, kept meeting-room and software-demo clips exploratory, fixed primary-prompt calibration, and added exploratory clips as negative contrast.
