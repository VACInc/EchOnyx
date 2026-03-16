"""OIDC helpers for external identity providers like Authentik."""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import httpx
from fastapi import HTTPException, Request
from sqlalchemy import delete, select

from app.auth import get_client_ip, hash_token
from app.config import Settings, get_settings
from app.database import async_session_maker
from app.models.auth import OidcLoginState

OIDC_STATE_TTL_MINUTES = 10


def make_oidc_client(**kwargs):
    return httpx.AsyncClient(**kwargs)


@dataclass(slots=True)
class OidcProviderMetadata:
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    issuer: str


@dataclass(slots=True)
class OidcCallbackIdentity:
    actor_label: str
    email: str | None
    subject: str
    groups: list[str]


def _split_csv(raw: str) -> set[str]:
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _normalize_groups(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _base64url_sha256(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _well_known_url(issuer_url: str) -> str:
    issuer = issuer_url.strip().rstrip("/")
    if issuer.endswith("/.well-known/openid-configuration"):
        return issuer
    return f"{issuer}/.well-known/openid-configuration"


def _request_origin(request: Request) -> str:
    scheme = request.url.scheme
    host = request.url.hostname or "localhost"
    if request.url.port:
        return f"{scheme}://{host}:{request.url.port}"
    return f"{scheme}://{host}"


def default_frontend_redirect_url(request: Request) -> str:
    settings = get_settings()
    if settings.oidc_frontend_redirect_url.strip():
        return settings.oidc_frontend_redirect_url.strip()
    scheme = request.url.scheme
    host = request.url.hostname or "localhost"
    return f"{scheme}://{host}:3000/"


def resolve_oidc_redirect_uri(request: Request) -> str:
    settings = get_settings()
    if settings.oidc_redirect_uri.strip():
        return settings.oidc_redirect_uri.strip()
    return str(request.url_for("oidc_callback"))


def sanitize_post_login_redirect(request: Request, next_url: str | None) -> str:
    fallback = default_frontend_redirect_url(request)
    if not next_url:
        return fallback
    parsed = urlparse(next_url)
    if not parsed.scheme:
        if next_url.startswith("/"):
            fallback_origin = urlparse(fallback)
            return urlunparse(
                (
                    fallback_origin.scheme,
                    fallback_origin.netloc,
                    next_url,
                    "",
                    "",
                    "",
                )
            )
        return fallback
    if parsed.scheme not in {"http", "https"}:
        return fallback
    if parsed.hostname != request.url.hostname:
        return fallback
    return next_url


async def fetch_oidc_metadata(settings: Settings) -> OidcProviderMetadata:
    if not settings.oidc_issuer_url.strip():
        raise HTTPException(status_code=409, detail="OIDC issuer URL is not configured.")
    async with make_oidc_client(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(_well_known_url(settings.oidc_issuer_url))
        response.raise_for_status()
    payload = response.json()
    try:
        return OidcProviderMetadata(
            authorization_endpoint=payload["authorization_endpoint"],
            token_endpoint=payload["token_endpoint"],
            userinfo_endpoint=payload["userinfo_endpoint"],
            issuer=payload.get("issuer", settings.oidc_issuer_url.strip()),
        )
    except KeyError as exc:
        raise HTTPException(status_code=502, detail="OIDC provider metadata is incomplete.") from exc


async def create_login_redirect(request: Request, next_url: str | None) -> str:
    settings = get_settings()
    metadata = await fetch_oidc_metadata(settings)
    state = secrets.token_urlsafe(24)
    code_verifier = secrets.token_urlsafe(48)
    redirect_url = sanitize_post_login_redirect(request, next_url)
    login_state = OidcLoginState(
        state_hash=hash_token(state),
        code_verifier=code_verifier,
        redirect_url=redirect_url,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=OIDC_STATE_TTL_MINUTES),
        client_ip=get_client_ip(request),
        user_agent=(request.headers.get("user-agent") or "")[:512] or None,
    )
    async with async_session_maker() as db:
        db.add(login_state)
        await db.commit()

    query = urlencode(
        {
            "response_type": "code",
            "client_id": settings.oidc_client_id,
            "redirect_uri": resolve_oidc_redirect_uri(request),
            "scope": settings.oidc_scopes.strip(),
            "state": state,
            "code_challenge": _base64url_sha256(code_verifier),
            "code_challenge_method": "S256",
        }
    )
    return f"{metadata.authorization_endpoint}?{query}"


async def consume_login_state(request: Request, state: str) -> OidcLoginState:
    async with async_session_maker() as db:
        result = await db.execute(select(OidcLoginState).where(OidcLoginState.state_hash == hash_token(state)))
        login_state = result.scalar_one_or_none()
        if not login_state:
            raise HTTPException(status_code=400, detail="Invalid or expired OIDC state.")
        if login_state.expires_at.replace(tzinfo=timezone.utc) <= datetime.now(timezone.utc):
            await db.delete(login_state)
            await db.commit()
            raise HTTPException(status_code=400, detail="Expired OIDC state.")
        stored = login_state
        await db.delete(login_state)
        await db.commit()
    if stored.user_agent and request.headers.get("user-agent") and stored.user_agent != request.headers.get("user-agent"):
        raise HTTPException(status_code=400, detail="OIDC state validation failed.")
    return stored


def _identity_from_userinfo(settings: Settings, userinfo: dict[str, Any]) -> OidcCallbackIdentity:
    subject = str(userinfo.get("sub") or "").strip()
    if not subject:
        raise HTTPException(status_code=502, detail="OIDC userinfo is missing 'sub'.")
    email = str(userinfo.get("email") or "").strip() or None
    groups = _normalize_groups(userinfo.get("groups"))
    allowed_emails = _split_csv(settings.oidc_allowed_emails)
    allowed_groups = _split_csv(settings.oidc_allowed_groups)
    if allowed_emails and (not email or email.lower() not in allowed_emails):
        raise HTTPException(status_code=403, detail="OIDC account is not allowed.")
    if allowed_groups and not ({group.lower() for group in groups} & allowed_groups):
        raise HTTPException(status_code=403, detail="OIDC account is not allowed.")
    actor_label = (
        email
        or str(userinfo.get("preferred_username") or "").strip()
        or str(userinfo.get("name") or "").strip()
        or subject
    )
    return OidcCallbackIdentity(
        actor_label=actor_label[:64],
        email=email,
        subject=subject,
        groups=groups,
    )


async def exchange_code_for_identity(request: Request, *, code: str, state: str) -> tuple[OidcCallbackIdentity, str]:
    settings = get_settings()
    metadata = await fetch_oidc_metadata(settings)
    login_state = await consume_login_state(request, state)
    token_payload = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": settings.oidc_client_id,
        "redirect_uri": resolve_oidc_redirect_uri(request),
        "code_verifier": login_state.code_verifier,
    }
    if settings.oidc_client_secret.strip():
        token_payload["client_secret"] = settings.oidc_client_secret

    async with make_oidc_client(timeout=30.0, follow_redirects=True) as client:
        token_response = await client.post(
            metadata.token_endpoint,
            data=token_payload,
            headers={"Accept": "application/json"},
        )
        if token_response.status_code >= 400:
            raise HTTPException(status_code=502, detail="OIDC token exchange failed.")
        token_data = token_response.json()
        access_token = str(token_data.get("access_token") or "").strip()
        if not access_token:
            raise HTTPException(status_code=502, detail="OIDC token exchange returned no access token.")
        userinfo_response = await client.get(
            metadata.userinfo_endpoint,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        if userinfo_response.status_code >= 400:
            raise HTTPException(status_code=502, detail="OIDC userinfo lookup failed.")
        identity = _identity_from_userinfo(settings, userinfo_response.json())
    return identity, (login_state.redirect_url or sanitize_post_login_redirect(request, None))


async def cleanup_expired_oidc_states() -> None:
    async with async_session_maker() as db:
        await db.execute(delete(OidcLoginState).where(OidcLoginState.expires_at <= datetime.now(timezone.utc)))
        await db.commit()
