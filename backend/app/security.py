"""Security helpers for browser-origin handling."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Sequence
from urllib.parse import urlparse

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
HF_REPO_ID_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
GGUF_FILENAME_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$")
GENERIC_MODEL_NAME_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}$")


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


def is_private_network_host(host: str | None) -> bool:
    """Return whether a hostname points at localhost or a private network."""
    normalized = (host or "").strip().lower()
    if not normalized:
        return False
    if normalized in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or normalized.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback


def validate_endpoint_url(url: str | None) -> str:
    """Allow only blank, HTTPS, or explicit local/private-network HTTP endpoints."""
    clean = (url or "").strip()
    if not clean:
        return ""
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Endpoint URL must use http or https.")
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("Endpoint URL must include a hostname.")
    if parsed.username or parsed.password:
        raise ValueError("Endpoint URL must not embed credentials.")
    if parsed.scheme == "http" and not is_private_network_host(parsed.hostname):
        raise ValueError("Plain HTTP endpoints must stay on localhost or a private network.")
    return clean.rstrip("/")


def validate_model_name(model_name: str | None, *, allow_gguf: bool = True) -> str:
    """Validate configurable model identifiers and reject path-like values."""
    clean = (model_name or "").strip()
    if not clean:
        raise ValueError("Model name is required.")
    if any(char in clean for char in ("\r", "\n", "\t")):
        raise ValueError("Model name contains unsupported control characters.")
    if clean.startswith(("/", ".", "~")) or "\\" in clean or clean.startswith(("http://", "https://")):
        raise ValueError("Filesystem paths and URLs are not valid model names.")
    if allow_gguf and clean.endswith(".gguf"):
        if not GGUF_FILENAME_REGEX.match(clean) or "/" in clean:
            raise ValueError("GGUF model names must be plain filenames, not paths.")
        return clean
    if "/" in clean:
        if not HF_REPO_ID_REGEX.match(clean):
            raise ValueError("Hugging Face model ids must look like namespace/repo.")
        return clean
    if not GENERIC_MODEL_NAME_REGEX.match(clean):
        raise ValueError("Model name contains unsupported characters.")
    return clean


def apply_security_headers(response) -> None:
    """Apply low-friction API security headers."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
