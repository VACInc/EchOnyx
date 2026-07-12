"""Automatic model downloader for GGUF and other models."""

import logging
import os
import threading
from urllib.parse import unquote, urlparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

logger = logging.getLogger(__name__)


@dataclass
class DownloadProgress:
    """Track download progress for a model."""
    model_name: str
    status: str = "pending"  # pending, downloading, completed, failed
    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_bytes_per_sec: float = 0
    eta_seconds: float | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def progress_percent(self) -> float:
        if self.total_bytes == 0:
            return 0
        return (self.downloaded_bytes / self.total_bytes) * 100

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "status": self.status,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "progress_percent": round(self.progress_percent, 1),
            "speed_mbps": round(self.speed_bytes_per_sec / (1024 * 1024), 2),
            "eta_seconds": round(self.eta_seconds) if self.eta_seconds else None,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# Global download progress tracker
_download_progress: dict[str, DownloadProgress] = {}
_progress_lock = threading.Lock()

PYANNOTE_LICENSE_URLS = (
    "https://huggingface.co/pyannote/speaker-diarization-community-1",
    "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "https://huggingface.co/pyannote/segmentation-3.0",
)

WHISPER_ALIAS_REPOS = {
    "tiny": ("Systran/faster-whisper-tiny", "openai/whisper-tiny"),
    "base": ("Systran/faster-whisper-base", "openai/whisper-base"),
    "small": ("Systran/faster-whisper-small", "openai/whisper-small"),
    "medium": ("Systran/faster-whisper-medium", "openai/whisper-medium"),
    "large": ("Systran/faster-whisper-large-v3", "openai/whisper-large"),
    "large-v3": ("Systran/faster-whisper-large-v3", "openai/whisper-large-v3"),
    "large-v3-turbo": ("Systran/faster-whisper-large-v3-turbo", "openai/whisper-large-v3-turbo"),
}


def get_all_download_progress() -> dict[str, dict]:
    """Get progress for all model downloads."""
    with _progress_lock:
        return {name: prog.to_dict() for name, prog in _download_progress.items()}


def get_download_progress(model_name: str) -> dict | None:
    """Get progress for a specific model download."""
    with _progress_lock:
        if model_name in _download_progress:
            return _download_progress[model_name].to_dict()
    return None


def reserve_download_progress(model_name: str, total_bytes: int = 0) -> bool:
    """Reserve a progress slot before an async download task starts."""
    with _progress_lock:
        existing = _download_progress.get(model_name)
        if existing and existing.status == "downloading":
            return False
        _download_progress[model_name] = DownloadProgress(
            model_name=model_name,
            status="downloading",
            total_bytes=total_bytes,
            started_at=datetime.now(),
        )
        return True


def _update_progress(model_name: str, **kwargs):
    """Update progress for a model download."""
    with _progress_lock:
        if model_name not in _download_progress:
            _download_progress[model_name] = DownloadProgress(model_name=model_name)
        for key, value in kwargs.items():
            setattr(_download_progress[model_name], key, value)

# Model registry mapping model names to HuggingFace repo info
MODEL_REGISTRY = {
    # Vision models - skip for now (Qwen3-Omni GGUF not widely available)
    # User would need to manually download or use a different vision model
    "Qwen3-Omni-30B-A3B-Q4_K_M.gguf": {
        "repo_id": "Qwen/Qwen3-30B-A3B-GGUF",
        "filename": "Qwen3-30B-A3B-Q4_K_M.gguf",
        "fallback_repo": "bartowski/Qwen_Qwen3-30B-A3B-GGUF",
        "fallback_filename": "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
        "size_gb": 24.0,
    },
    "Qwen3VL-32B-Instruct-Q4_K_M.gguf": {
        "repo_id": "Qwen/Qwen3-VL-32B-Instruct-GGUF",
        "filename": "Qwen3VL-32B-Instruct-Q4_K_M.gguf",
        "size_gb": 24.0,
        # llama_server refuses to start the vision endpoint without the
        # projector; keep in sync with the config.py mmproj auto-attach.
        "companions": ["mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"],
    },
    "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf": {
        "repo_id": "Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF",
        "filename": "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf",
        "size_gb": 19.0,
        "companions": ["mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf"],
    },
    "mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf": {
        "repo_id": "Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF",
        "filename": "mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf",
        "size_gb": 0.8,
    },
    "Qwen3VL-8B-Instruct-Q4_K_M.gguf": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "filename": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 5.1,
        "companions": ["mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"],
    },
    "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf": {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "filename": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
        "size_gb": 0.8,
    },
    "mmproj-Qwen3VL-32B-Instruct-F16.gguf": {
        "repo_id": "Qwen/Qwen3-VL-32B-Instruct-GGUF",
        "filename": "mmproj-Qwen3VL-32B-Instruct-F16.gguf",
        "size_gb": 1.5,
    },
    "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf": {
        "repo_id": "Qwen/Qwen3-VL-32B-Instruct-GGUF",
        "filename": "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf",
        "size_gb": 1.0,
    },
    "Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf": {
        "repo_id": "mradermacher/Qwen2.5-VL-3B-Instruct-GGUF",
        "filename": "Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf",
        "size_gb": 4.0,
        "companions": ["Qwen2.5-VL-3B-Instruct.mmproj-fp16.gguf"],
    },
    "Qwen2.5-VL-3B-Instruct.mmproj-fp16.gguf": {
        "repo_id": "mradermacher/Qwen2.5-VL-3B-Instruct-GGUF",
        "filename": "Qwen2.5-VL-3B-Instruct.mmproj-fp16.gguf",
        "size_gb": 1.0,
    },
    # Summarization models - Qwen3 30B A3B (MoE)
    "Qwen3-30B-A3B-Q4_K_M.gguf": {
        "repo_id": "Qwen/Qwen3-30B-A3B-GGUF",
        "filename": "Qwen3-30B-A3B-Q4_K_M.gguf",
        "fallback_repo": "bartowski/Qwen_Qwen3-30B-A3B-GGUF",
        "fallback_filename": "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
        "size_gb": 24.0,
    },
    "Qwen2.5-3B-Instruct.Q4_K_M.gguf": {
        "repo_id": "mradermacher/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct.Q4_K_M.gguf",
        "size_gb": 3.0,
    },
    # Alternative smaller models for testing/lower memory
    "qwen2.5-7b-instruct-q4_k_m.gguf": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_gb": 5.0,
    },
    "qwen2.5-14b-instruct-q4_k_m.gguf": {
        "repo_id": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "size_gb": 9.0,
    },
}


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def pyannote_token_guidance() -> str:
    urls = ", ".join(PYANNOTE_LICENSE_URLS)
    return (
        "HF_TOKEN is required to download pyannote models. "
        f"Set HF_TOKEN and accept the model licenses at: {urls}"
    )


def missing_model_download_message(model_name: str) -> str:
    return (
        f"Model {model_name} is not downloaded. "
        "Download it in Settings or set MODEL_AUTO_DOWNLOAD=true."
    )


def read_only_model_dir_message(model_name: str, target_dir: Path, *, env_var: str = "MODEL_CACHE_DIR") -> str:
    expected_file = target_dir / Path(model_name).name
    return (
        f"Cannot download model {Path(model_name).name}; target directory {target_dir} is not writable. "
        f"Expected file: {expected_file}. Set {env_var} to a writable model path or pre-download the file."
    )


def _dedupe_paths(paths: list[Path]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return tuple(deduped)


def model_search_dirs(cache_dir: Path | None = None) -> tuple[Path, ...]:
    """Return model directories to probe for local GGUF files."""
    dirs: list[Path] = []
    if cache_dir is not None:
        dirs.append(Path(cache_dir))
    env_cache = os.environ.get("MODEL_CACHE_DIR", "").strip()
    if env_cache:
        dirs.append(Path(env_cache))
    dirs.extend([Path("/models"), Path("/data/models")])
    return _dedupe_paths(dirs)


def _is_bare_filename(path: Path) -> bool:
    return not path.is_absolute() and path.parent == Path(".") and path.name == str(path)


def _case_insensitive_file_match(path: Path) -> Path | None:
    directory = path.parent
    try:
        entries = tuple(directory.iterdir())
    except OSError:
        return None
    target = path.name.lower()
    for entry in entries:
        try:
            is_file = entry.is_file()
        except OSError:
            continue
        if is_file and entry.name.lower() == target:
            return entry
    return None


def resolve_local_model_path(model_name_or_path: str | Path, cache_dir: Path | None = None) -> Path | None:
    """Resolve an existing local model file with case-insensitive filename fallback."""
    raw_text = str(model_name_or_path).strip()
    if not raw_text or raw_text.startswith(("http://", "https://")):
        return None

    raw_path = Path(raw_text)
    if _is_bare_filename(raw_path):
        candidates = [directory / raw_path.name for directory in model_search_dirs(cache_dir)]
    else:
        candidates = [raw_path]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    for candidate in candidates:
        match = _case_insensitive_file_match(candidate)
        if match is not None:
            return match

    return None


def model_download_target(model_name_or_path: str | Path, cache_dir: Path) -> tuple[str, Path]:
    """Return the registry filename and directory to use if a local file must be downloaded."""
    raw_path = Path(str(model_name_or_path).strip())
    if _is_bare_filename(raw_path):
        return raw_path.name, Path(cache_dir)
    return raw_path.name, raw_path.parent


def _path_has_write_bit(path: Path) -> bool:
    try:
        return bool(path.stat().st_mode & 0o222)
    except OSError:
        return False


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return None
        current = parent
    return current


def _download_dir_writable(path: Path) -> bool:
    if path.exists():
        return path.is_dir() and os.access(path, os.W_OK) and _path_has_write_bit(path)
    parent = _nearest_existing_parent(path.parent)
    return bool(parent and parent.is_dir() and os.access(parent, os.W_OK) and _path_has_write_bit(parent))


def ensure_download_dir_writable(model_name: str, target_dir: Path, *, env_var: str = "MODEL_CACHE_DIR") -> None:
    if not _download_dir_writable(target_dir):
        raise RuntimeError(read_only_model_dir_message(model_name, target_dir, env_var=env_var))


def _enum_value(value) -> str:
    return getattr(value, "value", value)


def _registry_lookup(model_name: str) -> tuple[str, dict] | tuple[None, None]:
    model_info = MODEL_REGISTRY.get(model_name)
    if model_info is not None:
        return model_name, model_info
    model_name_lower = model_name.lower()
    for registry_name, registry_info in MODEL_REGISTRY.items():
        if registry_name.lower() == model_name_lower:
            return registry_name, registry_info
    return None, None


def get_model_expected_size_gb(model_name: str) -> float | None:
    """Return expected download size for registry-backed models when known."""
    _, model_info = _registry_lookup(model_name)
    if model_info is None:
        return None
    size_gb = model_info.get("size_gb")
    return float(size_gb) if size_gb is not None else None


def companion_model_names(model_name: str) -> list[str]:
    """Registry-declared files that must ship alongside a model.

    A vision GGUF without its mmproj projector downloads fine but leaves the
    llama_server endpoint unable to start, so downloads must treat the pair as
    one unit. Only registry-resolvable companions are returned, under their
    canonical registry names.
    """
    _, model_info = _registry_lookup(model_name)
    if model_info is None:
        return []
    resolved: list[str] = []
    for companion in model_info.get("companions", ()):
        canonical, info = _registry_lookup(companion)
        if canonical is not None and info is not None and canonical != model_name:
            resolved.append(canonical)
    return resolved


def _hf_cache_dir() -> Path:
    cache = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
    )
    if cache:
        return Path(cache)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_cache_roots(cache_dir: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for root in (cache_dir, _hf_cache_dir()):
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _repo_cache_name(repo_id: str) -> str:
    return f"models--{repo_id.replace('/', '--')}"


def _cached_snapshot_path(repo_id: str, cache_dir: Path) -> Path | None:
    for root in _hf_repo_cache_roots(cache_dir):
        snapshots = root / _repo_cache_name(repo_id) / "snapshots"
        if not snapshots.exists():
            continue
        try:
            snapshot_dirs = [path for path in snapshots.iterdir() if path.is_dir()]
        except OSError:
            continue
        if snapshot_dirs:
            return max(snapshot_dirs, key=lambda path: path.stat().st_mtime)
    return None


def _hf_repo_cached(repo_id: str, cache_dir: Path) -> bool:
    return _cached_snapshot_path(repo_id, cache_dir) is not None


def _whisper_alias_repo_ids(model_name: str, backend: str | None = None) -> tuple[str, ...]:
    normalized = model_name.strip().lower()
    if normalized.startswith("whisper-"):
        normalized = normalized.removeprefix("whisper-")
    if normalized.startswith("openai/whisper-"):
        normalized = normalized.removeprefix("openai/whisper-")
    repos = WHISPER_ALIAS_REPOS.get(normalized)
    if not repos:
        return ()
    backend_value = _enum_value(backend)
    if backend_value in {"cuda", "cpu", None, ""}:
        return repos
    return (repos[1], repos[0])


def _resolve_snapshot_repo_id(
    model_name: str,
    *,
    component: str | None = None,
    backend: str | None = None,
) -> str | None:
    if model_name.endswith(".gguf") or model_name.startswith(("http://", "https://")):
        return None
    if "/" in model_name:
        return model_name
    if component == "asr" or _whisper_alias_repo_ids(model_name, backend):
        repos = _whisper_alias_repo_ids(model_name, backend)
        return repos[0] if repos else None
    return None


def _snapshot_repo_ids_for_cache_check(
    model_name: str,
    *,
    component: str | None = None,
    backend: str | None = None,
) -> tuple[str, ...]:
    if model_name.endswith(".gguf") or model_name.startswith(("http://", "https://")):
        return ()
    if "/" in model_name:
        return (model_name,)
    if component == "asr" or _whisper_alias_repo_ids(model_name, backend):
        return _whisper_alias_repo_ids(model_name, backend)
    return ()


def is_model_cached(
    model_name: str,
    cache_dir: Path,
    *,
    component: str | None = None,
    backend: str | None = None,
) -> bool:
    """Best-effort local cache check for GGUF files and Hugging Face snapshots."""
    if model_name.startswith(("http://", "https://")):
        try:
            return (cache_dir / _filename_from_url(model_name)).exists()
        except ValueError:
            return False
    if model_name.endswith(".gguf"):
        if resolve_local_model_path(model_name, cache_dir) is not None:
            return True
        registry_name, _ = _registry_lookup(model_name)
        return bool(registry_name and resolve_local_model_path(registry_name, cache_dir) is not None)
    if (cache_dir / model_name).exists():
        return True
    return any(
        _hf_repo_cached(repo_id, cache_dir)
        for repo_id in _snapshot_repo_ids_for_cache_check(
            model_name,
            component=component,
            backend=backend,
        )
    )


def _repo_total_size_bytes(repo_id: str, token: str | None) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        repo_info = api.repo_info(repo_id, token=token)
    except Exception:
        return 0
    return int(sum((getattr(file, "size", 0) or 0) for file in repo_info.siblings))


def _download_hf_snapshot(
    model_name: str,
    repo_id: str,
    cache_dir: Path,
    token: str | None,
) -> Path:
    if repo_id.startswith("pyannote/") and not token:
        _update_progress(model_name, status="failed", error=pyannote_token_guidance())
        raise ValueError(pyannote_token_guidance())

    cached_path = _cached_snapshot_path(repo_id, cache_dir)
    if cached_path is not None:
        _update_progress(model_name, status="completed", completed_at=datetime.now())
        logger.info("Model already cached: %s", repo_id)
        return cached_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    total_size = _repo_total_size_bytes(repo_id, token)
    _update_progress(
        model_name,
        status="downloading",
        total_bytes=total_size,
        started_at=datetime.now(),
    )
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        token=token,
    )
    _update_progress(
        model_name,
        status="completed",
        downloaded_bytes=total_size,
        completed_at=datetime.now(),
    )
    logger.info("Successfully downloaded Hugging Face snapshot: %s", repo_id)
    return Path(downloaded_path)


def _normalize_hf_url(url: str) -> str:
    """Convert Hugging Face blob URLs to resolve URLs for direct download."""
    if "huggingface.co" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    return url


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    filename = Path(unquote(parsed.path)).name
    if not filename:
        raise ValueError(f"Could not infer filename from URL: {url}")
    return filename


def download_model_from_url(url: str, cache_dir: Path) -> Path:
    """
    Download a model file from a direct URL.

    Supports Hugging Face blob URLs by converting them to resolve URLs.
    """
    url = _normalize_hf_url(url)
    filename = _filename_from_url(url)
    model_path = cache_dir / filename

    # Probe for size to determine if we already have the file
    total_size = 0
    try:
        head = httpx.head(url, follow_redirects=True, timeout=30)
        if head.status_code < 400:
            total_size = int(head.headers.get("content-length") or 0)
    except Exception:
        total_size = 0

    if model_path.exists():
        if total_size == 0 or model_path.stat().st_size == total_size:
            _update_progress(url, status="completed", completed_at=datetime.now())
            logger.info(f"Model already exists: {model_path}")
            return model_path

    ensure_download_dir_writable(filename, cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _update_progress(url, status="downloading", total_bytes=total_size, started_at=datetime.now())

    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    downloaded = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                _update_progress(
                    url,
                    downloaded_bytes=downloaded,
                    total_bytes=total_size,
                )

    tmp_path.replace(model_path)
    _update_progress(url, status="completed", downloaded_bytes=downloaded, completed_at=datetime.now())
    logger.info(f"Successfully downloaded: {model_path}")
    return model_path


def download_model(
    model_name: str,
    cache_dir: Path,
    *,
    component: str | None = None,
    backend: str | None = None,
    env_var: str = "MODEL_CACHE_DIR",
) -> Path:
    """
    Download a model from HuggingFace Hub if not already present.

    Args:
        model_name: Name of the model file (e.g., "qwen3-30b-a3b-q4_k_m.gguf")
        cache_dir: Directory to store downloaded models

    Returns:
        Path to the downloaded model file
    """
    import time
    from huggingface_hub import HfApi

    if model_name.startswith(("http://", "https://")):
        return download_model_from_url(model_name, cache_dir)

    hf_token = get_hf_token()
    snapshot_repo_id = _resolve_snapshot_repo_id(
        model_name,
        component=component,
        backend=backend,
    )
    if snapshot_repo_id is not None:
        ensure_download_dir_writable(model_name, cache_dir, env_var=env_var)
        return _download_hf_snapshot(model_name, snapshot_repo_id, cache_dir, hf_token)

    # Check if already downloaded
    existing_path = resolve_local_model_path(model_name, cache_dir)
    if existing_path is not None:
        logger.info(f"Model already exists: {existing_path}")
        _update_progress(model_name, status="completed", completed_at=datetime.now())
        return existing_path

    download_name, target_dir = model_download_target(model_name, cache_dir)
    model_path = target_dir / download_name

    # Get model info from registry (case-insensitive fallback)
    registry_name, model_info = _registry_lookup(download_name)
    if registry_name and registry_name != download_name:
        model_path = target_dir / registry_name

    if model_info is None:
        if download_name.endswith(".gguf"):
            logger.warning(f"Model {download_name} not in registry. Please download manually.")
            _update_progress(model_name, status="failed", error="Model not in registry")
            raise FileNotFoundError(
                f"Model {download_name} not found in registry. "
                f"Please download it manually to {target_dir}"
            )
        raise ValueError(f"Unknown model: {model_name}")

    existing_path = resolve_local_model_path(model_path, cache_dir)
    if existing_path is not None:
        logger.info(f"Model already exists: {existing_path}")
        _update_progress(model_name, status="completed", completed_at=datetime.now())
        return existing_path

    ensure_download_dir_writable(model_path.name, target_dir, env_var=env_var)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Get file size first for progress tracking
    api = HfApi()
    try:
        repo_info = api.repo_info(model_info["repo_id"], token=hf_token)
        file_info = next(
            (f for f in repo_info.siblings if f.rfilename == model_info["filename"]),
            None,
        )
        total_size = (file_info.size if file_info and file_info.size else 0)
    except Exception:
        total_size = 0

    _update_progress(
        model_name,
        status="downloading",
        total_bytes=total_size,
        started_at=datetime.now()
    )

    # Try primary repo first
    try:
        if total_size:
            logger.info(
                f"Downloading {model_name} from {model_info['repo_id']} ({total_size / 1e9:.1f} GB)..."
            )
        else:
            logger.info(f"Downloading {model_name} from {model_info['repo_id']}...")

        # Download with progress monitoring
        start_time = time.time()
        downloaded_path = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["filename"],
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
        )

        # Update progress on completion
        elapsed = time.time() - start_time
        speed = total_size / elapsed if elapsed > 0 else 0

        # Rename to expected name if different
        downloaded = Path(downloaded_path)
        expected_name = model_path.name
        if downloaded.name != expected_name:
            downloaded.rename(model_path)

        _update_progress(
            model_name,
            status="completed",
            downloaded_bytes=total_size,
            speed_bytes_per_sec=speed,
            completed_at=datetime.now()
        )

        logger.info(f"Successfully downloaded: {model_path}")
        return model_path

    except Exception as e:
        error_detail = str(e)
        logger.warning(f"Primary download failed: {error_detail}")

        # Try fallback repo if available
        if "fallback_repo" in model_info:
            try:
                logger.info(f"Trying fallback: {model_info['fallback_repo']}...")

                # Get fallback file size
                try:
                    repo_info = api.repo_info(model_info["fallback_repo"], token=hf_token)
                    file_info = next(
                        (f for f in repo_info.siblings if f.rfilename == model_info["fallback_filename"]),
                        None
                    )
                    total_size = file_info.size if file_info else 0
                    _update_progress(model_name, total_bytes=total_size)
                except Exception:
                    pass

                start_time = time.time()
                downloaded_path = hf_hub_download(
                    repo_id=model_info["fallback_repo"],
                    filename=model_info["fallback_filename"],
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                    token=hf_token,
                )

                elapsed = time.time() - start_time
                speed = total_size / elapsed if elapsed > 0 else 0

                downloaded = Path(downloaded_path)
                expected_name = model_path.name
                if downloaded.name != expected_name:
                    downloaded.rename(model_path)

                _update_progress(
                    model_name,
                    status="completed",
                    downloaded_bytes=total_size,
                    speed_bytes_per_sec=speed,
                    completed_at=datetime.now()
                )

                logger.info(f"Successfully downloaded from fallback: {model_path}")
                return model_path

            except Exception as e2:
                error_detail = str(e2)
                logger.error(f"Fallback download also failed: {error_detail}")

        token_hint = ""
        if not hf_token:
            token_hint = " Set HF_TOKEN if the repo is gated."

        _update_progress(model_name, status="failed", error=error_detail)
        raise FileNotFoundError(
            f"Failed to download model {model_name} ({error_detail}). "
            f"Please download manually from HuggingFace to {target_dir}.{token_hint}"
        )


def ensure_model_available(
    model_name: str,
    cache_dir: Path,
    *,
    component: str | None = None,
    backend: str | None = None,
) -> Path:
    """
    Ensure a model is available, downloading if necessary.

    This is a synchronous wrapper for use in model loading.
    """
    return download_model(model_name, cache_dir, component=component, backend=backend)


async def download_model_async(
    model_name: str,
    cache_dir: Path,
    *,
    component: str | None = None,
    backend: str | None = None,
    env_var: str = "MODEL_CACHE_DIR",
) -> Path:
    """Async version of model download."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: download_model(
            model_name,
            cache_dir,
            component=component,
            backend=backend,
            env_var=env_var,
        ),
    )


def list_available_models() -> dict[str, dict]:
    """List all models available in the registry."""
    return MODEL_REGISTRY.copy()


def search_huggingface_gguf(query: str, limit: int = 10) -> list[dict]:
    """
    Search HuggingFace for GGUF models.

    Args:
        query: Search query (e.g., "qwen 7b gguf")
        limit: Maximum results to return

    Returns:
        List of matching repos with their GGUF files
    """
    from huggingface_hub import HfApi

    api = HfApi()
    results = []

    try:
        # Search for repos
        repos = api.list_models(
            search=query,
            filter="gguf",
            limit=limit,
        )

        for repo in repos:
            try:
                files = list_repo_files(repo.id)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                if gguf_files:
                    results.append({
                        "repo_id": repo.id,
                        "gguf_files": gguf_files[:5],  # Limit files shown
                    })
            except Exception:
                pass

    except Exception as e:
        logger.error(f"HuggingFace search failed: {e}")

    return results
