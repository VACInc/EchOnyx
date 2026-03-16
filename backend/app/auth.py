"""Single-admin auth, session, rate-limit, and audit helpers."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Request, WebSocket
from redis import asyncio as redis_async
from sqlalchemy import delete, select

from app.config import Settings, get_settings
from app.database import async_session_maker
from app.models.auth import AuditLog, AuthSession, OidcLoginState

PBKDF2_ITERATIONS = 390_000
PASSWORD_HASH_SCHEME = "pbkdf2_sha256"
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
PUBLIC_PATHS = {
    "/health",
    "/api/auth/session",
    "/api/auth/login",
    "/api/auth/setup",
    "/api/auth/oidc/login",
    "/api/auth/oidc/callback",
}
IN_MEMORY_RATE_LIMITS: dict[str, tuple[int, float]] = {}
IN_MEMORY_RATE_LIMIT_LOCK = asyncio.Lock()


@dataclass(slots=True)
class AuthState:
    """Resolved request auth state."""

    configured: bool
    authenticated: bool
    session_id: str | None = None
    actor_label: str | None = None
    csrf_token_hash: str | None = None
    setup_required: bool = False


class RateLimiter:
    """Best-effort Redis-backed fixed-window rate limiter with in-memory fallback."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis = None
        self._redis_disabled = False

    async def check(self, key: str, *, limit: int, window_seconds: int) -> tuple[bool, int]:
        if limit <= 0:
            return True, 0
        if self.redis_url and not self._redis_disabled:
            allowed, retry_after = await self._check_redis(key, limit=limit, window_seconds=window_seconds)
            if allowed is not None:
                return allowed, retry_after
        return await self._check_memory(key, limit=limit, window_seconds=window_seconds)

    async def _check_redis(self, key: str, *, limit: int, window_seconds: int) -> tuple[bool | None, int]:
        try:
            if self._redis is None:
                self._redis = redis_async.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
            bucket = f"rate:{key}"
            count = await self._redis.incr(bucket)
            if count == 1:
                await self._redis.expire(bucket, window_seconds)
            ttl = await self._redis.ttl(bucket)
            return count <= limit, max(ttl, 0)
        except Exception:
            self._redis_disabled = True
            self._redis = None
            return None, 0

    async def _check_memory(self, key: str, *, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.time()
        async with IN_MEMORY_RATE_LIMIT_LOCK:
            count, reset_at = IN_MEMORY_RATE_LIMITS.get(key, (0, now + window_seconds))
            if now >= reset_at:
                count = 0
                reset_at = now + window_seconds
            count += 1
            IN_MEMORY_RATE_LIMITS[key] = (count, reset_at)
        return count <= limit, max(int(reset_at - now), 0)


_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(get_settings().redis_url)
    return _rate_limiter


def auth_is_configured(settings: Settings) -> bool:
    """Return whether a password hash has been configured."""
    return (not settings.auth_required) or local_password_auth_is_configured(settings) or oidc_is_enabled(settings)


def local_password_auth_is_configured(settings: Settings) -> bool:
    """Return whether local password auth is configured."""
    return bool(settings.auth_password_hash.strip())


def oidc_is_enabled(settings: Settings) -> bool:
    """Return whether OIDC is configured enough to be offered."""
    return (
        settings.auth_required
        and settings.oidc_enabled
        and bool(settings.oidc_issuer_url.strip())
        and bool(settings.oidc_client_id.strip())
    )


def auth_setup_is_required(settings: Settings) -> bool:
    """Return whether there is no usable auth method yet."""
    return settings.auth_required and not auth_is_configured(settings)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_password(password: str, *, salt: str | None = None, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    digest = base64.urlsafe_b64encode(derived).decode("ascii")
    return f"{PASSWORD_HASH_SCHEME}${iterations}${salt}${digest}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations_raw, salt, expected = encoded.split("$", 3)
    except ValueError:
        return False
    if scheme != PASSWORD_HASH_SCHEME:
        return False
    try:
        iterations = int(iterations_raw)
    except ValueError:
        return False
    candidate = hash_password(password, salt=salt, iterations=iterations)
    return hmac.compare_digest(candidate, encoded)


def build_secure_cookie_flag(request: Request) -> bool:
    return request.url.scheme == "https"


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded:
        return forwarded
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _is_private_or_loopback_host(host: str) -> bool:
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


def setup_request_is_allowed(request: Request) -> bool:
    return _is_private_or_loopback_host(get_client_ip(request))


def is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS


async def create_session(*, user_label: str, client_ip: str, user_agent: str | None) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(24)
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.auth_session_ttl_hours)
    session = AuthSession(
        user_label=user_label,
        token_hash=hash_token(token),
        csrf_token_hash=hash_token(csrf_token),
        expires_at=expires_at,
        last_seen_at=datetime.now(timezone.utc),
        client_ip=client_ip,
        user_agent=(user_agent or "")[:512] or None,
    )
    async with async_session_maker() as db:
        db.add(session)
        await db.commit()
    return token, csrf_token


async def resolve_session(token: str | None) -> AuthState:
    settings = get_settings()
    if not settings.auth_required:
        return AuthState(configured=True, authenticated=True, actor_label="local")
    configured = auth_is_configured(settings)
    if not configured or not token:
        return AuthState(
            configured=configured,
            authenticated=False,
            setup_required=not configured,
        )

    async with async_session_maker() as db:
        result = await db.execute(select(AuthSession).where(AuthSession.token_hash == hash_token(token)))
        session = result.scalar_one_or_none()
        if not session:
            return AuthState(configured=True, authenticated=False)
        now = datetime.now(timezone.utc)
        expires_at = session.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            await db.delete(session)
            await db.commit()
            return AuthState(configured=True, authenticated=False)
        session.last_seen_at = now
        await db.commit()
        return AuthState(
            configured=True,
            authenticated=True,
            session_id=str(session.id),
            actor_label=session.user_label,
            csrf_token_hash=session.csrf_token_hash,
        )


async def revoke_session(token: str | None) -> None:
    if not token:
        return
    async with async_session_maker() as db:
        result = await db.execute(select(AuthSession).where(AuthSession.token_hash == hash_token(token)))
        session = result.scalar_one_or_none()
        if session:
            await db.delete(session)
            await db.commit()


async def rotate_password(password_hash: str) -> None:
    settings = get_settings()
    async with async_session_maker() as db:
        await db.execute(delete(AuthSession))
        await db.commit()
    settings.auth_password_hash = password_hash


async def cleanup_security_state() -> None:
    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.audit_log_retention_days)
    async with async_session_maker() as db:
        await db.execute(delete(AuthSession).where(AuthSession.expires_at <= datetime.now(timezone.utc)))
        await db.execute(delete(OidcLoginState).where(OidcLoginState.expires_at <= datetime.now(timezone.utc)))
        await db.execute(delete(AuditLog).where(AuditLog.created_at < cutoff))
        await db.commit()


async def write_audit_log(
    *,
    event: str,
    method: str,
    path: str,
    status_code: int,
    session_id: str | None,
    actor_label: str | None,
    client_ip: str | None,
    user_agent: str | None,
    details: dict[str, Any] | None = None,
) -> None:
    try:
        audit = AuditLog(
            session_id=uuid.UUID(session_id) if session_id else None,
            actor_label=actor_label,
            event=event,
            method=method,
            path=path[:512],
            status_code=status_code,
            client_ip=(client_ip or "")[:128] or None,
            user_agent=(user_agent or "")[:512] or None,
            details=details,
        )
        async with async_session_maker() as db:
            db.add(audit)
            await db.commit()
    except Exception:
        return


def apply_auth_cookies(response, *, token: str, csrf_token: str, request: Request) -> None:
    settings = get_settings()
    max_age = settings.auth_session_ttl_hours * 3600
    secure = build_secure_cookie_flag(request)
    response.set_cookie(
        settings.auth_session_cookie_name,
        token,
        max_age=max_age,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        settings.auth_csrf_cookie_name,
        csrf_token,
        max_age=max_age,
        httponly=False,
        secure=secure,
        samesite="lax",
        path="/",
    )


def clear_auth_cookies(response, request: Request) -> None:
    settings = get_settings()
    secure = build_secure_cookie_flag(request)
    response.delete_cookie(settings.auth_session_cookie_name, path="/", secure=secure, samesite="lax")
    response.delete_cookie(settings.auth_csrf_cookie_name, path="/", secure=secure, samesite="lax")


async def resolve_request_auth_state(request: Request) -> AuthState:
    state = getattr(request.state, "auth_state", None)
    if isinstance(state, AuthState):
        return state
    settings = get_settings()
    token = request.cookies.get(settings.auth_session_cookie_name)
    state = await resolve_session(token)
    request.state.auth_state = state
    return state


async def authenticate_websocket(websocket: WebSocket) -> AuthState:
    settings = get_settings()
    token = websocket.cookies.get(settings.auth_session_cookie_name)
    return await resolve_session(token)
