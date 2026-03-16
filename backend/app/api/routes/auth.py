"""Session auth endpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth import (
    apply_auth_cookies,
    auth_is_configured,
    clear_auth_cookies,
    create_session,
    get_client_ip,
    hash_password,
    resolve_request_auth_state,
    revoke_session,
    rotate_password,
    setup_request_is_allowed,
    verify_password,
)
from app.config import get_settings
from app.env_utils import resolve_env_file_path, stringify_env_value, write_env_updates

router = APIRouter()


class AuthSessionResponse(BaseModel):
    authenticated: bool
    setup_required: bool
    actor_label: str | None = None


class AuthPasswordPayload(BaseModel):
    password: str = Field(min_length=12, max_length=256)


class ChangePasswordPayload(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)


@router.get("/session", response_model=AuthSessionResponse)
async def get_auth_session(request: Request) -> AuthSessionResponse:
    state = await resolve_request_auth_state(request)
    return AuthSessionResponse(
        authenticated=state.authenticated,
        setup_required=state.setup_required,
        actor_label=state.actor_label,
    )


@router.post("/setup", response_model=AuthSessionResponse)
async def setup_auth(payload: AuthPasswordPayload, request: Request):
    settings = get_settings()
    if auth_is_configured(settings):
        raise HTTPException(status_code=409, detail="Authentication is already configured.")
    if not setup_request_is_allowed(request):
        raise HTTPException(status_code=403, detail="Initial setup is only allowed from localhost or a private network.")

    password_hash = hash_password(payload.password)
    write_env_updates(resolve_env_file_path(), {"AUTH_PASSWORD_HASH": password_hash})
    os.environ["AUTH_PASSWORD_HASH"] = stringify_env_value(password_hash)
    get_settings.cache_clear()

    token, csrf_token = await create_session(
        user_label="admin",
        client_ip=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )

    from fastapi.responses import JSONResponse

    response = JSONResponse(
        AuthSessionResponse(authenticated=True, setup_required=False, actor_label="admin").model_dump()
    )
    apply_auth_cookies(response, token=token, csrf_token=csrf_token, request=request)
    return response


@router.post("/login", response_model=AuthSessionResponse)
async def login(payload: AuthPasswordPayload, request: Request):
    settings = get_settings()
    if not auth_is_configured(settings):
        raise HTTPException(status_code=409, detail="Authentication has not been configured yet.")
    if not verify_password(payload.password, settings.auth_password_hash):
        raise HTTPException(status_code=401, detail="Invalid password.")

    token, csrf_token = await create_session(
        user_label="admin",
        client_ip=get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )

    from fastapi.responses import JSONResponse

    response = JSONResponse(
        AuthSessionResponse(authenticated=True, setup_required=False, actor_label="admin").model_dump()
    )
    apply_auth_cookies(response, token=token, csrf_token=csrf_token, request=request)
    return response


@router.post("/logout", response_model=AuthSessionResponse)
async def logout(request: Request):
    settings = get_settings()
    await revoke_session(request.cookies.get(settings.auth_session_cookie_name))

    from fastapi.responses import JSONResponse

    response = JSONResponse(
        AuthSessionResponse(
            authenticated=False,
            setup_required=not auth_is_configured(get_settings()),
            actor_label=None,
        ).model_dump()
    )
    clear_auth_cookies(response, request)
    return response


@router.post("/password", response_model=AuthSessionResponse)
async def change_password(payload: ChangePasswordPayload, request: Request):
    settings = get_settings()
    if not auth_is_configured(settings):
        raise HTTPException(status_code=409, detail="Authentication has not been configured yet.")
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

    from fastapi.responses import JSONResponse

    response = JSONResponse(
        AuthSessionResponse(authenticated=True, setup_required=False, actor_label="admin").model_dump()
    )
    apply_auth_cookies(response, token=token, csrf_token=csrf_token, request=request)
    return response
