"""Shared helpers for persisting runtime configuration to the env file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings


def stringify_env_value(value: Any) -> str:
    """Serialize supported env values consistently."""
    enum_value = getattr(value, "value", value)
    if enum_value is None:
        return ""
    if isinstance(enum_value, bool):
        return "true" if enum_value else "false"
    return str(enum_value)


def resolve_env_file_path() -> Path:
    """Resolve the configured env file path."""
    env_file = Settings.model_config.get("env_file", ".env")
    if isinstance(env_file, (list, tuple)):
        env_file = env_file[0]
    return Path(env_file or ".env")


def write_env_updates(path: Path, updates: dict[str, Any]) -> None:
    """Apply env key updates to the configured env file."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    key_to_index: dict[str, int] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        key_to_index[key] = index

    for key, value in updates.items():
        if value is None:
            if key in key_to_index:
                lines.pop(key_to_index[key])
                key_to_index = {}
                for index, line in enumerate(lines):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in line:
                        continue
                    current_key = line.split("=", 1)[0].strip()
                    key_to_index[current_key] = index
            continue

        updated_line = f"{key}={stringify_env_value(value)}"
        if key in key_to_index:
            lines[key_to_index[key]] = updated_line
        else:
            lines.append(updated_line)

    payload = "\n".join(lines).rstrip()
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")
