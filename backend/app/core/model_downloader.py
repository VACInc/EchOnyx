"""Automatic model downloader for GGUF and other models."""

import logging
import os
import threading
from urllib.parse import unquote, urlparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import httpx
from huggingface_hub import hf_hub_download, list_repo_files

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
    },
    "Qwen3VL-32B-Instruct-Q4_K_M.gguf": {
        "repo_id": "Qwen/Qwen3-VL-32B-Instruct-GGUF",
        "filename": "Qwen3VL-32B-Instruct-Q4_K_M.gguf",
    },
    "mmproj-Qwen3VL-32B-Instruct-F16.gguf": {
        "repo_id": "Qwen/Qwen3-VL-32B-Instruct-GGUF",
        "filename": "mmproj-Qwen3VL-32B-Instruct-F16.gguf",
    },
    "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf": {
        "repo_id": "Qwen/Qwen3-VL-32B-Instruct-GGUF",
        "filename": "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf",
    },
    # Summarization models - Qwen3 30B A3B (MoE)
    "Qwen3-30B-A3B-Q4_K_M.gguf": {
        "repo_id": "Qwen/Qwen3-30B-A3B-GGUF",
        "filename": "Qwen3-30B-A3B-Q4_K_M.gguf",
        "fallback_repo": "bartowski/Qwen_Qwen3-30B-A3B-GGUF",
        "fallback_filename": "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
    },
    # Alternative smaller models for testing/lower memory
    "qwen2.5-7b-instruct-q4_k_m.gguf": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
    },
    "qwen2.5-14b-instruct-q4_k_m.gguf": {
        "repo_id": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
    },
}


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


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

    cache_dir.mkdir(parents=True, exist_ok=True)

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


def download_model(model_name: str, cache_dir: Path) -> Path:
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

    model_path = cache_dir / model_name

    # Check if already downloaded
    if model_path.exists():
        logger.info(f"Model already exists: {model_path}")
        _update_progress(model_name, status="completed", completed_at=datetime.now())
        return model_path

    # Get model info from registry (case-insensitive fallback)
    model_info = MODEL_REGISTRY.get(model_name)
    if model_info is None:
        model_name_lower = model_name.lower()
        for registry_name, registry_info in MODEL_REGISTRY.items():
            if registry_name.lower() == model_name_lower:
                model_info = registry_info
                break

    if model_info is None:
        if model_name.endswith(".gguf"):
            logger.warning(f"Model {model_name} not in registry. Please download manually.")
            _update_progress(model_name, status="failed", error="Model not in registry")
            raise FileNotFoundError(
                f"Model {model_name} not found in registry. "
                f"Please download it manually to {cache_dir}"
            )
        raise ValueError(f"Unknown model: {model_name}")

    hf_token = get_hf_token()
    cache_dir.mkdir(parents=True, exist_ok=True)

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
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
        )

        # Update progress on completion
        elapsed = time.time() - start_time
        speed = total_size / elapsed if elapsed > 0 else 0

        # Rename to expected name if different
        downloaded = Path(downloaded_path)
        if downloaded.name != model_name:
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
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False,
                    token=hf_token,
                )

                elapsed = time.time() - start_time
                speed = total_size / elapsed if elapsed > 0 else 0

                downloaded = Path(downloaded_path)
                if downloaded.name != model_name:
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
            f"Please download manually from HuggingFace to {cache_dir}.{token_hint}"
        )


def ensure_model_available(model_name: str, cache_dir: Path) -> Path:
    """
    Ensure a model is available, downloading if necessary.

    This is a synchronous wrapper for use in model loading.
    """
    return download_model(model_name, cache_dir)


async def download_model_async(model_name: str, cache_dir: Path) -> Path:
    """Async version of model download."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, download_model, model_name, cache_dir)


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
