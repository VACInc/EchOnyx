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
- Ask mode keeps a local follow-up chat thread and reuses prior turns for grounded follow-up questions
- User labels (tags) and label-scoped search/ask
- Retry (resume) and Reset (reprocess) pipeline controls
- Duplicate detection with configurable suppression thresholds
- Settings model management with verify/add flow for built-in entries and Hugging Face model ids

## Quick Start (Docker)

### Prereqs

- Docker + Docker Compose
- 32GB RAM minimum (128GB recommended for Strix Halo)
- Hugging Face account + accepted pyannote terms if you want pyannote diarization

### 1) Configure

```bash
git clone <repository-url>
cd EchOnyx
cp .env.example .env
```

Update `.env` if you want diarization:

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
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up -d
```
If you are building on a host without a visible GPU during `docker build`, set `CUDA_ARCHITECTURES` for your target cards. On the live `ai-server`, `86;120` was validated for `RTX 3090 + RTX PRO 6000 Blackwell`.
The NVIDIA override now uses `gpus: all`, so normal Docker Compose exposes every visible NVIDIA GPU to backend and worker containers.
The NVIDIA worker currently runs Celery with `--pool=solo` for stability while local CUDA `llama.cpp` vision and summarization loads are being hardened.
On NVIDIA, the audio-event path now reads extracted WAV audio directly instead of depending on `torchaudio` file I/O, so a bad `torchcodec` runtime no longer blocks summarization.
The CUDA image now builds `llama-cpp-python` against its bundled vendored `llama.cpp` by default; only opt into an external `llama.cpp` checkout if you are intentionally testing an upstream override.
The NVIDIA endpoint services now self-place from live `nvidia-smi` free-memory data when explicit device pins are unset. On a single smaller GPU, they automatically switch to stage-by-stage endpoint loading instead of trying to keep both vision and summarization hot.
On the live `ai-server`, the current mixed NVIDIA path is: summarization on a pinned `3090` via bundled-vendor CUDA `llama.cpp`, and vision on the `RTX PRO 6000` via official `vLLM 0.11.2`. That split is now live-accepted end to end for upload, batch, summary, search, ask, and similar.
Summaries and `ask` answers now strip `<think>...</think>` reasoning blocks before they are stored or returned.

**Apple Silicon / Metal:**
Run backend and worker on the host, not Docker. The initial Metal bring-up now auto-selects smaller defaults on unified-memory Macs:
- `WHISPER_MODEL=small`
- `EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5`
- `VISION_MODEL=Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf`
- `SUMMARIZATION_MODEL=Qwen2.5-3B-Instruct.Q4_K_M.gguf`

On Apple host runs, the worker should use Celery `--pool=solo`, and the default Apple path now uses local project `data/` directories instead of `/data/...`. Follow `backend/README.md` for the host-run commands. This path is meant to prove functionality on a `16 GB` Apple Silicon machine, not the full high-capacity default stack yet.

### 3) Access

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- FastAPI docs: `http://localhost:8000/docs`
- PostgreSQL and Redis stay internal to the Compose network by default; they are no longer published on host ports unless you add your own override.

## Configuration (Key Env Vars)

Set these in `.env` as needed:

- `GPU_BACKEND`: `cuda` | `metal` | `vulkan` | `rocm` | `cpu`
- `MODEL_LOADING`: `sequential` (low memory) or `parallel`
- `HF_TOKEN`: optional overall, but required if you want pyannote diarization
- `VISION_ENDPOINT_URL`, `VISION_ENDPOINT_MODEL`: use an external VL server
- `SUMMARIZATION_ENDPOINT_URL`, `SUMMARIZATION_ENDPOINT_MODEL`: use an external LLM server
- `AUDIO_EVENT_CALIBRATION_PATH`: optional JSON profile that overrides CLAP prompts and support thresholds
- `ROCM_LLM_RUNTIME`: `llama_server` (managed idle teardown) or `vllm`
- `ROCM_LLM_IDLE_TIMEOUT_S`: idle shutdown for ROCm `llama_server` endpoints
- `INSTALL_VLLM=1`: opt-in build flag for the heavier ROCm `vLLM` image path
- `VLLM_INSTALL_METHOD`: `wheel` (official ROCm wheel index) or `source`
- `CUDA_WHL_URL`, `CUDA_TORCH_VERSION`, `CUDA_TORCHAUDIO_VERSION`, `CUDA_TORCHVISION_VERSION`: CUDA PyTorch image build controls
- `CUDA_ARCHITECTURES`: optional CUDA arch list for `llama.cpp` image builds; use target SMs such as `86;120` for `3090 + RTX PRO 6000 Blackwell`
- `CUDA_VISIBLE_DEVICES`: leave it unset by default; setting it to an empty string hides all CUDA devices. When it is unset, the planner now narrows local `llama.cpp` loads to the selected CUDA devices automatically before the first import.
- `NVIDIA_VISION_VISIBLE_DEVICES`, `NVIDIA_SUMMARIZATION_VISIBLE_DEVICES`: optional role-specific NVIDIA endpoint-service pinning overrides; when they are unset, the managed NVIDIA endpoints auto-pick GPUs from current free memory
- `NVIDIA_ENDPOINT_IDLE_TIMEOUT_SECONDS`: idle teardown for managed NVIDIA endpoint runtimes
- `LLAMA_BUILD_CUDA=1`: enable CUDA `llama.cpp` builds in the NVIDIA backend image
- `INSTALL_NEMO=1`: include NeMo so Canary ASR works in the NVIDIA image
- Audio-event classification is treated as supporting context only; if that stage fails, summarization continues without audio hints
- `VISION_VLLM_MODEL_ID`, `SUMMARIZATION_VLLM_MODEL_ID`: Hugging Face model ids for the `vLLM` runtime
- `EMBEDDING_MODEL`: embedding model id (HF)
- `UPLOAD_DIR`, `MODEL_CACHE_DIR`: storage locations

## Models

Defaults are configured in `.env.example`. All models are swappable:

- **Transcription**: `WHISPER_MODEL` (explicit ASR selector; no silent fallback)
- **Diarization**: `DIARIZATION_MODEL` (pyannote, optional if `HF_TOKEN` is unset)
- **Vision**: `VISION_MODEL` (GGUF) or `VISION_ENDPOINT_*`
- **Summarization**: `SUMMARIZATION_MODEL` (GGUF) or `SUMMARIZATION_ENDPOINT_*`
- **Embeddings**: `EMBEDDING_MODEL` (HF)
- **Audio Hints**: `AUDIO_EVENT_MODEL` (defaults to CLAP for raw-audio source cues)

On Apple Silicon bring-up, the repo now defaults to smaller local models so a `16 GB` Mac can process videos sequentially on Metal.

GGUF models can be downloaded automatically via the built-in model downloader when needed.

## Audio Calibration

Use the fixture-driven calibration command to generate an `AUDIO_EVENT_CALIBRATION_PATH` profile from labeled media fixtures:

```bash
python -m app.core.audio_calibration \
  --manifest /path/to/audio-calibration-manifest.json \
  --output /data/models/audio_event_calibration.json
```

Manifest shape:

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

Relative `media_path` values are resolved from the manifest directory. The command accepts audio or video files and will extract temporary audio automatically when needed.
Set `use_for_calibration` to `false` for exploratory fixtures you want to keep in the pack without letting them tune the default profile yet.

The repo now ships:

- a checked-in fixture pack in `backend/tests/fixtures/audio_calibration/` with both validated and exploratory clips
- a conservative packaged baseline profile at `backend/app/assets/audio_event_calibration.json`

That packaged baseline loads automatically when `/data/models/audio_event_calibration.json` is absent, and a custom `AUDIO_EVENT_CALIBRATION_PATH` still overrides it.

Current note: the default packaged profile intentionally remains conservative. Live Strix Halo validation on March 10, 2026 confirmed that the real weather-radio and applause fixtures are audio-separable enough to keep in the active calibration path, while the current real meeting and software-demo fixtures remain exploratory because raw CLAP audio-only classification still collapses them toward produced narration.
Primary prompt calibration now scores the real primary prompt variants instead of reusing one score per class, and the packaged baseline has been regenerated from the four validated fixtures.
Exploratory fixtures are now also used as negative contrast during calibration, so they can help reject over-broad prompt choices without being promoted into the validated calibration set.

Regenerate the packaged baseline with:

```bash
python -m app.core.audio_calibration \
  --manifest backend/tests/fixtures/audio_calibration/manifest.json \
  --output backend/app/assets/audio_event_calibration.json
```

## Operations Guide (Operator Focus)

### Upload + Processing
- Upload a video in the UI and processing starts immediately.
- Processing steps: audio extraction → transcription → diarization → transcript merge → frame extraction → vision analysis → summarization → embedding.
- If `HF_TOKEN` is not configured, diarization is skipped and the rest of the pipeline continues with transcript-only speaker data.

### Retry vs Reset
- **Retry**: resumes from the last successful step (idempotent).
- **Reset**: restarts the entire pipeline from scratch.
- Completed videos do not rerun by accident. A full rerun now requires an explicit forced reset or reprocess action.

### Duplicate Handling
- Duplicate policy is configurable in Settings.
- Default behavior collapses exact duplicates out of default search results while keeping one representative indexed.
- Suppressed duplicates can still be targeted directly with explicit `video_id` / `video_ids` search and ask requests.

### Labels (Tags)
- Add labels on a video’s detail page.
- Use labels as filters in Search/Ask to target only those videos.

### Model Management
- Settings now exposes selectors for ASR, diarization, vision, summarization, embeddings, and audio events.
- You can verify a built-in registry name or Hugging Face model id, then add it into the selector before saving.

### Delete Videos
- Use the Delete button on a video detail page to remove the video and all associated data (artifacts + embeddings).

## Search & Q/A

Use the Search page to:

- Search transcripts and summaries
- Ask natural-language questions with follow-up chat in the same thread
- Apply label filters to narrow the scope
- Similar-video ranking now favors transcript and key-point overlap more heavily than generic narration style

## Hardware Support

| Profile | Description | Model Loading |
|---------|-------------|---------------|
| Strix Halo | AMD APU with 128GB unified memory | Sequential (current default) |
| RTX 5090 | Single high-VRAM NVIDIA GPU (32GB+) | Parallel |
| Multi-GPU | Multiple NVIDIA GPUs | Parallel |
| CPU Only | Fallback for systems without GPU | Sequential |

Current accelerator sizing guidance for the shipped model set:
- Plan against free accelerator memory, not only installed VRAM or unified memory.
- The Settings runtime panel now shows installed accelerator memory separately from the active free-memory budget.
- Rough floor is about `24 GB` free to run the largest current stage sequentially.
- Practical single-accelerator target is about `32 GB` free.
- Keeping the worker-side models warm needs about `26.5 GB` of budget.
- Keeping worker-side models warm plus one local endpoint at a time needs about `50.5 GB` of budget.
- Keeping the whole current stack resident on one accelerator needs about `74.5 GB` of budget, which is about `100 GB` free at the default `GPU_MEMORY_FRACTION=0.75`.
- On multi-GPU systems, the planner now prefers the largest currently free accelerator first, then falls back to topology-aware spread.
- CUDA worker-side models now honor the planner's preferred device selection, and the NVIDIA Compose override now defaults vision/summarization to dedicated CUDA `llama_cpp.server` containers instead of in-process worker loads.
- The CUDA backend image now smoke-builds successfully on the live `ai-server`, and the mixed `3090 + RTX PRO 6000` runtime has now passed a live end-to-end acceptance run.
- Embedding indexing now sanitizes Chroma metadata to scalar-safe values before insert so malformed slide/topic payloads do not fail the whole job near the end.

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
  - use current free memory, not only total VRAM, when deciding placement
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

- **Diarization missing**: set `HF_TOKEN` and accept pyannote terms if you want speaker labels. Without it, uploads still process but diarization is skipped.
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
