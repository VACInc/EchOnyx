"""HTTP middleware for auth, CSRF, rate limits, audit logs, and request caps."""

from __future__ import annotations

import hmac
from typing import Callable

from fastapi import Request
from starlette.responses import JSONResponse, Response

from app.auth import (
    MUTATING_METHODS,
    PUBLIC_PATHS,
    get_client_ip,
    get_rate_limiter,
    hash_token,
    resolve_request_auth_state,
    write_audit_log,
)
from app.config import get_settings
from app.security import apply_security_headers


def _protected_api_path(path: str) -> bool:
    return path.startswith("/api") and path not in PUBLIC_PATHS


def _is_upload_like_request(path: str, method: str) -> bool:
    return method == "POST" and path in {"/api/videos/upload", "/api/batch"}


def _audit_event(path: str, method: str) -> str:
    if path.startswith("/api/auth/"):
        return f"auth.{path.rsplit('/', 1)[-1]}"
    if path.startswith("/api/settings"):
        return "settings.write"
    if path.startswith("/api/action-items"):
        return "action_items.write"
    if path.startswith("/api/videos"):
        return "videos.write"
    if path.startswith("/api/jobs"):
        return "jobs.write"
    if path.startswith("/api/batch"):
        return "batch.write"
    if path.startswith("/api/search"):
        return "search.write"
    return f"api.{method.lower()}"


async def _enforce_json_request_limit(request: Request) -> Response | None:
    settings = get_settings()
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        return None
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.max_json_request_bytes:
                return JSONResponse({"detail": "JSON request body too large."}, status_code=413)
        except ValueError:
            pass
    body = await request.body()
    if len(body) > settings.max_json_request_bytes:
        return JSONResponse({"detail": "JSON request body too large."}, status_code=413)
    return None


async def _enforce_rate_limit(request: Request, authenticated_key: str | None) -> Response | None:
    settings = get_settings()
    limiter = get_rate_limiter()
    path = request.url.path
    method = request.method.upper()
    client_ip = get_client_ip(request)

    if path in {"/api/auth/login", "/api/auth/setup"} and method == "POST":
        allowed, retry_after = await limiter.check(
            f"auth:{path}:{client_ip}",
            limit=settings.login_rate_limit_attempts,
            window_seconds=settings.login_rate_limit_window_seconds,
        )
        if not allowed:
            return JSONResponse(
                {"detail": "Too many authentication attempts. Try again later."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

    if method not in MUTATING_METHODS:
        return None

    identity = authenticated_key or client_ip
    if _is_upload_like_request(path, method):
        allowed, retry_after = await limiter.check(
            f"upload:{identity}",
            limit=settings.upload_rate_limit_requests,
            window_seconds=settings.upload_rate_limit_window_seconds,
        )
        if not allowed:
            return JSONResponse(
                {"detail": "Upload rate limit exceeded. Try again later."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

    allowed, retry_after = await limiter.check(
        f"write:{identity}",
        limit=settings.write_rate_limit_requests,
        window_seconds=settings.write_rate_limit_window_seconds,
    )
    if not allowed:
        return JSONResponse(
            {"detail": "Write rate limit exceeded. Try again later."},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    return None


def _csrf_valid(request: Request, csrf_token_hash: str | None) -> bool:
    if request.method.upper() not in MUTATING_METHODS or request.url.path in PUBLIC_PATHS:
        return True
    if not csrf_token_hash:
        return False
    settings = get_settings()
    cookie_token = request.cookies.get(settings.auth_csrf_cookie_name)
    header_token = request.headers.get("x-csrf-token", "").strip()
    if not cookie_token or not header_token or not hmac.compare_digest(cookie_token, header_token):
        return False
    return hmac.compare_digest(hash_token(header_token), csrf_token_hash)


def _should_audit(request: Request) -> bool:
    return request.url.path.startswith("/api/") and request.method.upper() in MUTATING_METHODS


async def security_http_middleware(request: Request, call_next: Callable) -> Response:
    path = request.url.path
    if not (path in {"/health", "/ready"} or path.startswith("/api")):
        response = await call_next(request)
        apply_security_headers(response)
        return response

    auth_state = await resolve_request_auth_state(request)
    client_ip = get_client_ip(request)
    authenticated_key = auth_state.session_id or client_ip

    size_response = await _enforce_json_request_limit(request)
    if size_response is not None:
        apply_security_headers(size_response)
        return size_response

    rate_response = await _enforce_rate_limit(request, authenticated_key)
    if rate_response is not None:
        apply_security_headers(rate_response)
        return rate_response

    settings = get_settings()
    if settings.auth_required and _protected_api_path(path):
        if auth_state.setup_required:
            response = JSONResponse(
                {"detail": "Authentication setup is required.", "setup_required": True},
                status_code=401,
            )
            apply_security_headers(response)
            return response
        if not auth_state.authenticated:
            response = JSONResponse({"detail": "Authentication required."}, status_code=401)
            apply_security_headers(response)
            return response
        if not _csrf_valid(request, auth_state.csrf_token_hash):
            response = JSONResponse({"detail": "Missing or invalid CSRF token."}, status_code=403)
            apply_security_headers(response)
            return response

    try:
        response = await call_next(request)
    except Exception:
        if _should_audit(request):
            await write_audit_log(
                event=_audit_event(path, request.method.upper()),
                method=request.method.upper(),
                path=path,
                status_code=500,
                session_id=auth_state.session_id,
                actor_label=auth_state.actor_label,
                client_ip=client_ip,
                user_agent=request.headers.get("user-agent"),
                details=None,
            )
        raise

    apply_security_headers(response)

    if _should_audit(request):
        await write_audit_log(
            event=_audit_event(path, request.method.upper()),
            method=request.method.upper(),
            path=path,
            status_code=response.status_code,
            session_id=auth_state.session_id,
            actor_label=auth_state.actor_label,
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent"),
            details=None,
        )

    return response
