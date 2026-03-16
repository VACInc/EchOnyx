from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi import FastAPI, Request
from sqlalchemy import select


async def _build_oidc_stack(monkeypatch, tmp_path, **env_overrides):
    defaults = {
        "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'oidc.sqlite3'}",
        "AUTH_REQUIRED": "true",
        "AUTH_PASSWORD_HASH": "",
        "OIDC_ENABLED": "true",
        "OIDC_PROVIDER_NAME": "Authentik",
        "OIDC_ISSUER_URL": "https://issuer.example",
        "OIDC_CLIENT_ID": "echonyx-client",
        "OIDC_CLIENT_SECRET": "super-secret",
        "OIDC_SCOPES": "openid profile email",
        "REDIS_URL": "",
        "AUDIO_EVENT_CALIBRATION_PATH": str(tmp_path / "audio-event-calibration.json"),
    }
    defaults.update(env_overrides)
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)

    loaded = {}
    loaded["app.config"] = (
        importlib.reload(sys.modules["app.config"])
        if "app.config" in sys.modules
        else importlib.import_module("app.config")
    )
    loaded["app.models"] = (
        sys.modules["app.models"] if "app.models" in sys.modules else importlib.import_module("app.models")
    )
    for name in ["app.database", "app.auth", "app.oidc", "app.http_security", "app.api.routes.auth"]:
        if name in sys.modules:
            loaded[name] = importlib.reload(sys.modules[name])
        else:
            loaded[name] = importlib.import_module(name)

    database = loaded["app.database"]
    auth_module = loaded["app.auth"]
    oidc_module = loaded["app.oidc"]
    http_security = loaded["app.http_security"]
    auth_route = loaded["app.api.routes.auth"]
    models = loaded["app.models"]

    auth_module.IN_MEMORY_RATE_LIMITS.clear()
    auth_module._rate_limiter = None

    await database.init_db()
    await auth_module.cleanup_security_state()

    app = FastAPI()
    app.middleware("http")(http_security.security_http_middleware)
    app.include_router(auth_route.router, prefix="/api/auth")

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/api/protected")
    async def protected():
        return {"ok": True}

    return SimpleNamespace(
        app=app,
        auth=auth_module,
        oidc=oidc_module,
        database=database,
        models=models,
    )


def _patch_oidc_provider(monkeypatch, oidc_module, *, email: str, groups: list[str] | None = None):
    provider = FastAPI()
    observed: dict[str, list[dict[str, str]]] = {"tokens": []}

    @provider.get("/.well-known/openid-configuration")
    async def openid_configuration():
        return {
            "issuer": "https://issuer.example",
            "authorization_endpoint": "https://issuer.example/authorize",
            "token_endpoint": "https://issuer.example/token",
            "userinfo_endpoint": "https://issuer.example/userinfo",
        }

    @provider.post("/token")
    async def token(request: Request):
        form = dict(await request.form())
        observed["tokens"].append({key: str(value) for key, value in form.items()})
        return {
            "access_token": "access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

    @provider.get("/userinfo")
    async def userinfo():
        return {
            "sub": "oidc-user-1",
            "email": email,
            "preferred_username": email.split("@", 1)[0],
            "groups": groups or ["echonyx-admins"],
        }

    transport = httpx.ASGITransport(app=provider)

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(oidc_module, "make_oidc_client", MockAsyncClient)
    return observed


@pytest.mark.asyncio
async def test_oidc_session_and_callback_flow(monkeypatch, tmp_path):
    stack = await _build_oidc_stack(monkeypatch, tmp_path)
    observed = _patch_oidc_provider(
        monkeypatch,
        stack.oidc,
        email="oidc-admin@example.com",
    )
    transport = httpx.ASGITransport(app=stack.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:8000",
        follow_redirects=False,
    ) as client:
        session = await client.get("/api/auth/session")
        assert session.status_code == 200
        assert session.json() == {
            "authenticated": False,
            "setup_required": False,
            "actor_label": None,
            "password_enabled": False,
            "oidc": {
                "enabled": True,
                "provider_name": "Authentik",
                "login_path": "/api/auth/oidc/login",
            },
        }

        login = await client.get(
            "/api/auth/oidc/login",
            params={"next_url": "http://127.0.0.1:3000/search"},
        )
        assert login.status_code == 307
        redirect = urlparse(login.headers["location"])
        assert redirect.netloc == "issuer.example"
        params = parse_qs(redirect.query)
        assert params["client_id"] == ["echonyx-client"]
        assert params["code_challenge_method"] == ["S256"]
        assert params["redirect_uri"] == ["http://127.0.0.1:8000/api/auth/oidc/callback"]

        callback = await client.get(
            "/api/auth/oidc/callback",
            params={"code": "auth-code", "state": params["state"][0]},
        )
        assert callback.status_code == 307
        assert callback.headers["location"] == "http://127.0.0.1:3000/search"
        assert client.cookies.get("echonyx_session")
        assert client.cookies.get("echonyx_csrf")
        assert observed["tokens"][0]["code"] == "auth-code"
        assert observed["tokens"][0]["client_secret"] == "super-secret"
        assert observed["tokens"][0]["code_verifier"]

        session_after = await client.get("/api/auth/session")
        assert session_after.status_code == 200
        assert session_after.json()["authenticated"] is True
        assert session_after.json()["actor_label"] == "oidc-admin@example.com"

    async with stack.database.async_session_maker() as db:
        result = await db.execute(select(stack.models.OidcLoginState))
        assert result.scalars().all() == []


@pytest.mark.asyncio
async def test_oidc_allowlist_rejects_unapproved_user(monkeypatch, tmp_path):
    stack = await _build_oidc_stack(
        monkeypatch,
        tmp_path,
        OIDC_ALLOWED_EMAILS="allowed@example.com",
    )
    _patch_oidc_provider(
        monkeypatch,
        stack.oidc,
        email="blocked@example.com",
    )
    transport = httpx.ASGITransport(app=stack.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:8000",
        follow_redirects=False,
    ) as client:
        login = await client.get("/api/auth/oidc/login")
        state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
        callback = await client.get(
            "/api/auth/oidc/callback",
            params={"code": "auth-code", "state": state},
        )

        assert callback.status_code == 307
        assert callback.headers["location"] == "http://127.0.0.1:3000?auth_error=oidc"

        session = await client.get("/api/auth/session")
        assert session.status_code == 200
        assert session.json()["authenticated"] is False
