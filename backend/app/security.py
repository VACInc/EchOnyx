"""Security helpers for browser-origin handling."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.config import Settings

DEFAULT_LOCAL_ORIGIN_REGEX = (
    r"^https?://("
    r"localhost"
    r"|127(?:\.\d{1,3}){3}"
    r"|0\.0\.0\.0"
    r"|host\.docker\.internal"
    r"|(?:10(?:\.\d{1,3}){3})"
    r"|(?:192\.168(?:\.\d{1,3}){2})"
    r"|(?:172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    r"|(?:[A-Za-z0-9-]+(?:\.local)?)"
    r")(?::\d{1,5})?$"
)


def parse_allowed_origins(raw_value: str | Sequence[str] | None) -> list[str]:
    """Normalize a comma-separated or sequence origin allowlist."""
    if not raw_value:
        return []
    if isinstance(raw_value, str):
        parts = raw_value.split(",")
    else:
        parts = list(raw_value)
    normalized: list[str] = []
    seen: set[str] = set()
    for origin in parts:
        clean = str(origin).strip().rstrip("/")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return normalized


def is_origin_allowed(
    origin: str | None,
    *,
    allowed_origins: Sequence[str],
    allow_origin_regex: str | None,
) -> bool:
    """Check whether a browser origin should be trusted."""
    if not origin:
        return True
    normalized = origin.strip().rstrip("/")
    if not normalized:
        return True
    if normalized in parse_allowed_origins(allowed_origins):
        return True
    if allow_origin_regex and re.match(allow_origin_regex, normalized, flags=re.IGNORECASE):
        return True
    return False


def cors_configuration(settings: Settings) -> tuple[list[str], str]:
    """Return the effective CORS allowlist and regex."""
    origins = parse_allowed_origins(settings.cors_allowed_origins)
    regex = settings.cors_allow_origin_regex.strip() or DEFAULT_LOCAL_ORIGIN_REGEX
    return origins, regex
