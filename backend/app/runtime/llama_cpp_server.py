"""Bootstrap a local llama_cpp OpenAI server with optional model downloads."""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.model_downloader import download_model


@dataclass(frozen=True)
class LlamaCppServerConfig:
    model_path: Path
    model_alias: str
    host: str
    port: int
    context_size: int
    gpu_layers: int
    chat_format: str
    clip_model_path: Path | None
    extra_args: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "LlamaCppServerConfig":
        raw_model_path = os.environ.get("MODEL_PATH", "").strip()
        if not raw_model_path:
            raise RuntimeError("MODEL_PATH is required.")

        raw_clip_model_path = os.environ.get("MODEL_MMPROJ", "").strip()
        return cls(
            model_path=Path(raw_model_path),
            model_alias=os.environ.get("MODEL_NAME", "").strip() or Path(raw_model_path).name,
            host=os.environ.get("LISTEN_HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=int(os.environ.get("PORT", "8000")),
            context_size=int(os.environ.get("MODEL_CONTEXT_SIZE", "8192")),
            gpu_layers=int(os.environ.get("MODEL_GPU_LAYERS", "-1")),
            chat_format=os.environ.get("MODEL_CHAT_FORMAT", "").strip(),
            clip_model_path=Path(raw_clip_model_path) if raw_clip_model_path else None,
            extra_args=tuple(shlex.split(os.environ.get("LLAMA_SERVER_EXTRA_ARGS", ""))),
        )


def _ensure_local_file(path: Path) -> Path:
    if path.exists():
        return path
    return download_model(path.name, path.parent)


def build_server_env(config: LlamaCppServerConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["MODEL"] = str(_ensure_local_file(config.model_path))
    env["MODEL_ALIAS"] = config.model_alias
    env["HOST"] = config.host
    env["PORT"] = str(config.port)
    env["N_CTX"] = str(config.context_size)
    env["N_GPU_LAYERS"] = str(config.gpu_layers)

    if config.chat_format:
        env["CHAT_FORMAT"] = config.chat_format

    if config.clip_model_path is not None:
        env["CLIP_MODEL_PATH"] = str(_ensure_local_file(config.clip_model_path))

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
