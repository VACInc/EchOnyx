"""Bootstrap a local llama_cpp OpenAI server with optional model downloads."""

from __future__ import annotations

import importlib.util
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.model_downloader import (
    download_model,
    missing_model_download_message,
    model_download_target,
    resolve_local_model_path,
)


logger = logging.getLogger(__name__)

_PIN_ENV_KEYS = (
    "MODEL_VISIBLE_DEVICES",
    "NVIDIA_VISION_VISIBLE_DEVICES",
    "NVIDIA_SUMMARIZATION_VISIBLE_DEVICES",
)


@dataclass(frozen=True)
class LlamaCppServerConfig:
    model_path: Path
    model_alias: str
    host: str
    port: int
    context_size: int
    gpu_layers: int
    main_gpu: int | None
    split_mode: int | None
    chat_format: str
    clip_model_path: Path | None
    extra_args: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "LlamaCppServerConfig":
        raw_model_path = os.environ.get("MODEL_PATH", "").strip()
        if not raw_model_path:
            raise RuntimeError("MODEL_PATH is required.")

        raw_clip_model_path = os.environ.get("MODEL_MMPROJ", "").strip()
        raw_main_gpu = os.environ.get("MODEL_MAIN_GPU", "").strip()
        raw_split_mode = os.environ.get("MODEL_SPLIT_MODE", "").strip()
        return cls(
            model_path=Path(raw_model_path),
            model_alias=os.environ.get("MODEL_NAME", "").strip() or Path(raw_model_path).name,
            host=os.environ.get("LISTEN_HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=int(os.environ.get("PORT", "8000")),
            context_size=int(os.environ.get("MODEL_CONTEXT_SIZE", "8192")),
            gpu_layers=int(os.environ.get("MODEL_GPU_LAYERS", "-1")),
            main_gpu=_parse_int(raw_main_gpu),
            split_mode=_parse_int(raw_split_mode),
            chat_format=os.environ.get("MODEL_CHAT_FORMAT", "").strip(),
            clip_model_path=Path(raw_clip_model_path) if raw_clip_model_path else None,
            extra_args=tuple(shlex.split(os.environ.get("LLAMA_SERVER_EXTRA_ARGS", ""))),
        )


def _model_cache_dir() -> Path:
    return Path(os.environ.get("MODEL_CACHE_DIR", "/data/models").strip() or "/data/models")


def _ensure_local_file(path: Path, *, env_var: str = "MODEL_PATH") -> Path:
    resolved = resolve_local_model_path(path, _model_cache_dir())
    if resolved is not None:
        return resolved
    if not _auto_download_enabled():
        raise RuntimeError(missing_model_download_message(path.name))
    model_name, target_dir = model_download_target(path, _model_cache_dir())
    return download_model(model_name, target_dir, env_var=env_var)


def _auto_download_enabled() -> bool:
    value = os.environ.get("MODEL_AUTO_DOWNLOAD", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _parse_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _split_device_tokens(value: str) -> tuple[str, ...]:
    tokens = tuple(part.strip() for part in value.split(",") if part.strip())
    return tokens


def _parse_device_indices(value: str) -> tuple[int, ...]:
    raw_parts = _split_device_tokens(value)
    if not raw_parts:
        return ()
    try:
        return tuple(int(part) for part in raw_parts)
    except ValueError:
        return ()


def _resolve_pinned_visible_devices(pin: str, parent_visible: str) -> tuple[str, tuple[int, ...]]:
    """Resolve host-index GPU pins into this process's ``CUDA_VISIBLE_DEVICES``.

    ``CUDA_VISIBLE_DEVICES`` does not compose: exporting a new value makes CUDA
    re-enumerate the physical/nvidia-smi device namespace, so pins are exported
    VERBATIM as physical ids (never parent-relative ordinals). A parent
    ``CUDA_VISIBLE_DEVICES`` acts as a deployment allowlist: pins outside it are
    dropped with a warning, and a pin set resolving to nothing fails fast instead
    of letting llama.cpp fall back to device 0. Returns
    ``(cuda_visible_devices, host_device_ids)``; ``main_gpu`` is translated to an
    ordinal within that list afterwards.
    """
    pin_tokens = _split_device_tokens(pin)
    parent_tokens = _split_device_tokens(parent_visible)

    child_tokens: list[str] = []
    for token in pin_tokens:
        if parent_tokens and token not in parent_tokens:
            logger.warning(
                "Ignoring GPU pin %r: it is not in the deployment's visible CUDA devices "
                "allowlist %s (CUDA_VISIBLE_DEVICES).",
                token,
                list(parent_tokens),
            )
            continue
        child_tokens.append(token)

    if not child_tokens:
        raise RuntimeError(
            f"None of the pinned GPUs {list(pin_tokens)} are usable within the visible "
            f"CUDA devices {list(parent_tokens)}. Set the pin to a physical GPU index "
            f"(as reported by nvidia-smi) that is included in CUDA_VISIBLE_DEVICES."
        )
    child_visible = ",".join(child_tokens)
    return child_visible, _parse_device_indices(child_visible)


def _resolve_visible_devices(env: dict[str, str]) -> tuple[int, ...]:
    """Narrow ``CUDA_VISIBLE_DEVICES`` from role pins and report host device ids.

    The returned tuple is the ordered list of HOST GPU ids this process will see,
    used to translate ``main_gpu`` to a child-local ordinal.
    """
    parent_visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()

    pin = ""
    for key in _PIN_ENV_KEYS:
        value = env.get(key, "").strip()
        if value:
            pin = value
            break

    if pin:
        child_visible, host_ids = _resolve_pinned_visible_devices(pin, parent_visible)
        env["CUDA_VISIBLE_DEVICES"] = child_visible
        # Physical-id pins are only meaningful under PCI bus ordering; CUDA's
        # default fastest-first ordering diverges from nvidia-smi indices on
        # mixed-GPU hosts (confirmed live on a 3090 + RTX PRO 6000 mix).
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        return host_ids

    if parent_visible:
        return _parse_device_indices(parent_visible)
    return ()


def _translate_visible_main_gpu(main_gpu: int, host_device_ids: tuple[int, ...]) -> int:
    if not host_device_ids:
        return main_gpu
    if main_gpu in host_device_ids:
        return host_device_ids.index(main_gpu)
    return main_gpu


def _infer_single_gpu_target(env: dict[str, str]) -> tuple[int | None, int | None]:
    cuda_visible = _split_device_tokens(env.get("CUDA_VISIBLE_DEVICES", ""))
    if len(cuda_visible) == 1:
        return 0, 0

    return None, None


def build_server_env(config: LlamaCppServerConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["MODEL"] = str(_ensure_local_file(config.model_path, env_var="MODEL_PATH"))
    env["MODEL_ALIAS"] = config.model_alias
    env["HOST"] = config.host
    env["PORT"] = str(config.port)
    env["N_CTX"] = str(config.context_size)
    env["N_GPU_LAYERS"] = str(config.gpu_layers)

    host_device_ids = _resolve_visible_devices(env)
    inferred_main_gpu, inferred_split_mode = _infer_single_gpu_target(env)
    main_gpu = (
        _translate_visible_main_gpu(config.main_gpu, host_device_ids)
        if config.main_gpu is not None
        else inferred_main_gpu
    )
    split_mode = config.split_mode
    if split_mode is None and main_gpu is not None:
        split_mode = 0
    if split_mode is None:
        split_mode = inferred_split_mode
    if main_gpu is not None:
        env["MAIN_GPU"] = str(main_gpu)
    if split_mode is not None:
        env["SPLIT_MODE"] = str(split_mode)

    if config.chat_format:
        env["CHAT_FORMAT"] = config.chat_format

    if config.clip_model_path is not None:
        env["CLIP_MODEL_PATH"] = str(_ensure_local_file(config.clip_model_path, env_var="MODEL_MMPROJ"))

    return env


def build_server_command(config: LlamaCppServerConfig) -> list[str]:
    return [sys.executable, "-m", "llama_cpp.server", *config.extra_args]


def ensure_server_dependencies() -> None:
    requirements = {
        "sse_starlette": "sse-starlette>=2.1.3",
        "starlette_context": "starlette-context>=0.3.6",
    }
    missing = [package for module_name, package in requirements.items() if importlib.util.find_spec(module_name) is None]
    if not missing:
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def main() -> None:
    config = LlamaCppServerConfig.from_env()
    ensure_server_dependencies()
    env = build_server_env(config)
    command = build_server_command(config)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
