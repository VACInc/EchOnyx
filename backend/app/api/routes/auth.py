"""Session auth endpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.auth import (
    apply_auth_cookies,
    auth_origin_is_allowed,
    auth_setup_is_required,
    auth_transport_is_secure,
    clear_auth_cookies,
    create_session,
    get_client_ip,
    hash_password,
    local_password_auth_is_configured,
    oidc_is_enabled,
    resolve_request_auth_state,
    revoke_session,
    rotate_password,
    setup_request_is_allowed,
    verify_password,
    write_audit_log,
)
from app.config import get_settings
from app.env_utils import resolve_env_file_path, stringify_env_value, write_env_updates
from app.oidc import create_login_redirect, default_frontend_redirect_url, exchange_code_for_identity

router = APIRouter()


class AuthOidcStatus(BaseModel):
    enabled: bool
    provider_name: str | None = None
    login_path: str | None = None


class AuthSessionResponse(BaseModel):
    authenticated: bool
    setup_required: bool
    actor_label: str | None = None
    password_enabled: bool
    oidc: AuthOidcStatus


class AuthPasswordPayload(BaseModel):
    password: str = Field(min_length=12, max_length=256)


class ChangePasswordPayload(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)


def _auth_methods_payload() -> tuple[bool, AuthOidcStatus]:
    settings = get_settings()
    oidc_enabled = oidc_is_enabled(settings)
    return (
        local_password_auth_is_configured(settings),
        AuthOidcStatus(
            enabled=oidc_enabled,
            provider_name=settings.oidc_provider_name if oidc_enabled else None,
            login_path="/api/auth/oidc/login" if oidc_enabled else None,
        ),
    )


@router.get("/session", response_model=AuthSessionResponse)
async def get_auth_session(request: Request) -> AuthSessionResponse:
    state = await resolve_request_auth_state(request)
    password_enabled, oidc_status = _auth_methods_payload()
    return AuthSessionResponse(
        authenticated=state.authenticated,
        setup_required=state.setup_required,
        actor_label=state.actor_label,
        password_enabled=password_enabled,
        oidc=oidc_status,
    )


@router.post("/setup", response_model=AuthSessionResponse)
async def setup_auth(payload: AuthPasswordPayload, request: Request):
    settings = get_settings()
    if local_password_auth_is_configured(settings):
        raise HTTPException(status_code=409, detail="Authentication is already configured.")
    if not auth_origin_is_allowed(request):
        raise HTTPException(status_code=403, detail="Cross-origin setup is not allowed.")
    if not setup_request_is_allowed(request):
        raise HTTPException(status_code=403, detail="Initial setup is only allowed from localhost.")
    if not auth_transport_is_secure(request):
        raise HTTPException(status_code=400, detail="HTTPS is required for remote authentication.")

    password_hash = hash_password(payload.password)
    write_env_updates(resolve_env_file_path(), {"AUTH_PASSWORD_HASH": password_hash})
    os.environ["AUTH_PASSWORD_HASH"] = stringify_env_value(password_hash)
    get_settings.cache_clear()

    token, csrf_token = await create_session(
        user_label="admin",
        client_ip=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )
    password_enabled, oidc_status = _auth_methods_payload()

    response = JSONResponse(
        AuthSessionResponse(
            authenticated=True,
            setup_required=False,
            actor_label="admin",
            password_enabled=password_enabled,
            oidc=oidc_status,
        ).model_dump()
    )
    apply_auth_cookies(response, token=token, csrf_token=csrf_token, request=request)
    return response


@router.post("/login", response_model=AuthSessionResponse)
async def login(payload: AuthPasswordPayload, request: Request):
    settings = get_settings()
    if not local_password_auth_is_configured(settings):
        raise HTTPException(status_code=409, detail="Authentication has not been configured yet.")
    if not auth_origin_is_allowed(request):
        raise HTTPException(status_code=403, detail="Cross-origin login is not allowed.")
    if not auth_transport_is_secure(request):
        raise HTTPException(status_code=400, detail="HTTPS is required for remote authentication.")
    if not verify_password(payload.password, settings.auth_password_hash):
        raise HTTPException(status_code=401, detail="Invalid password.")

    token, csrf_token = await create_session(
        user_label="admin",
        client_ip=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )
    password_enabled, oidc_status = _auth_methods_payload()

    response = JSONResponse(
        AuthSessionResponse(
            authenticated=True,
            setup_required=False,
            actor_label="admin",
            password_enabled=password_enabled,
            oidc=oidc_status,
        ).model_dump()
    )
    apply_auth_cookies(response, token=token, csrf_token=csrf_token, request=request)
    return response


@router.post("/logout", response_model=AuthSessionResponse)
async def logout(request: Request):
    settings = get_settings()
    await revoke_session(request.cookies.get(settings.auth_session_cookie_name))
    password_enabled, oidc_status = _auth_methods_payload()

    response = JSONResponse(
        AuthSessionResponse(
            authenticated=False,
            setup_required=auth_setup_is_required(get_settings()),
            actor_label=None,
            password_enabled=password_enabled,
            oidc=oidc_status,
        ).model_dump()
    )
    clear_auth_cookies(response, request)
    return response


@router.post("/password", response_model=AuthSessionResponse)
async def change_password(payload: ChangePasswordPayload, request: Request):
    settings = get_settings()
    if not local_password_auth_is_configured(settings):
        raise HTTPException(status_code=409, detail="Authentication has not been configured yet.")
    if not auth_transport_is_secure(request):
        raise HTTPException(status_code=400, detail="HTTPS is required for remote authentication.")
    if not verify_password(payload.current_password, settings.auth_password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")

    password_hash = hash_password(payload.new_password)
    write_env_updates(resolve_env_file_path(), {"AUTH_PASSWORD_HASH": password_hash})
    os.environ["AUTH_PASSWORD_HASH"] = stringify_env_value(password_hash)
    await rotate_password(password_hash)
    get_settings.cache_clear()

    token, csrf_token = await create_session(
        user_label="admin",
        client_ip=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )
    password_enabled, oidc_status = _auth_methods_payload()

    response = JSONResponse(
        AuthSessionResponse(
            authenticated=True,
            setup_required=False,
            actor_label="admin",
            password_enabled=password_enabled,
            oidc=oidc_status,
        ).model_dump()
    )
    apply_auth_cookies(response, token=token, csrf_token=csrf_token, request=request)
    return response


@router.get("/oidc/login", name="oidc_login")
async def oidc_login(
    request: Request,
    next_url: str | None = Query(default=None, max_length=1024),
):
    settings = get_settings()
    if not oidc_is_enabled(settings):
        raise HTTPException(status_code=404, detail="OIDC is not configured.")
    if not auth_origin_is_allowed(request):
        raise HTTPException(status_code=403, detail="Cross-origin login is not allowed.")
    if not auth_transport_is_secure(request):
        raise HTTPException(status_code=400, detail="HTTPS is required for remote authentication.")
    return RedirectResponse(await create_login_redirect(request, next_url), status_code=307)


@router.get("/oidc/callback", name="oidc_callback")
async def oidc_callback(
    request: Request,
    code: str = Query(min_length=1, max_length=4096),
    state: str = Query(min_length=1, max_length=1024),
):
    settings = get_settings()
    if not oidc_is_enabled(settings):
        raise HTTPException(status_code=404, detail="OIDC is not configured.")
    if not auth_transport_is_secure(request):
        raise HTTPException(status_code=400, detail="HTTPS is required for remote authentication.")

    try:
        identity, redirect_url = await exchange_code_for_identity(request, code=code, state=state)
    except HTTPException as exc:
        await write_audit_log(
            event="auth.oidc_callback",
            method="GET",
            path="/api/auth/oidc/callback",
            status_code=exc.status_code,
            session_id=None,
            actor_label=None,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            details={"result": "failed"},
        )
        fallback = f"{default_frontend_redirect_url(request).rstrip('/')}?auth_error=oidc"
        return RedirectResponse(fallback, status_code=307)

    token, csrf_token = await create_session(
        user_label=identity.actor_label,
        client_ip=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )
    await write_audit_log(
        event="auth.oidc_callback",
        method="GET",
        path="/api/auth/oidc/callback",
        status_code=307,
        session_id=None,
        actor_label=identity.actor_label,
        client_ip=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
        details={"result": "success"},
    )
    response = RedirectResponse(redirect_url, status_code=307)
    apply_auth_cookies(response, token=token, csrf_token=csrf_token, request=request)
    return response
