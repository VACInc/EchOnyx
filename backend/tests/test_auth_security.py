from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, File, UploadFile
from sqlalchemy import select


async def _build_auth_stack(monkeypatch, tmp_path, **env_overrides):
    defaults = {
        "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'auth.sqlite3'}",
        "AUTH_REQUIRED": "true",
        "AUTH_PASSWORD_HASH": "",
        "REDIS_URL": "",
        "LOGIN_RATE_LIMIT_ATTEMPTS": "2",
        "LOGIN_RATE_LIMIT_WINDOW_SECONDS": "60",
        "WRITE_RATE_LIMIT_REQUESTS": "20",
        "WRITE_RATE_LIMIT_WINDOW_SECONDS": "60",
        "UPLOAD_RATE_LIMIT_REQUESTS": "2",
        "UPLOAD_RATE_LIMIT_WINDOW_SECONDS": "60",
        "MAX_JSON_REQUEST_BYTES": "128",
        "AUDIO_EVENT_CALIBRATION_PATH": str(tmp_path / "audio-event-calibration.json"),
    }
    defaults.update(env_overrides)
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)

    loaded = {}
    loaded["app.config"] = importlib.reload(sys.modules["app.config"]) if "app.config" in sys.modules else importlib.import_module("app.config")
    loaded["app.models"] = sys.modules["app.models"] if "app.models" in sys.modules else importlib.import_module("app.models")
    for name in ["app.database", "app.auth", "app.oidc", "app.http_security", "app.api.routes.auth"]:
        if name in sys.modules:
            loaded[name] = importlib.reload(sys.modules[name])
        else:
            loaded[name] = importlib.import_module(name)

    database = loaded["app.database"]
    auth_module = loaded["app.auth"]
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
    async def protected_get():
        return {"ok": True}

    @app.post("/api/protected")
    async def protected_post(payload: dict):
        return payload

    @app.post("/api/videos/upload")
    async def protected_upload(file: UploadFile = File(...)):
        return {"filename": file.filename}

    return SimpleNamespace(
        app=app,
        auth=auth_module,
        database=database,
        models=models,
    )


def _client(stack, *, client_host: str = "127.0.0.1", base_url: str = "http://127.0.0.1:8000"):
    transport = httpx.ASGITransport(app=stack.app, client=(client_host, 12345))
    return httpx.AsyncClient(transport=transport, base_url=base_url)


@pytest.mark.asyncio
async def test_setup_session_and_csrf_protect_writes(monkeypatch, tmp_path):
    stack = await _build_auth_stack(monkeypatch, tmp_path)
    async with _client(stack) as client:
        session = await client.get("/api/auth/session")
        assert session.status_code == 200
        assert session.json() == {
            "authenticated": False,
            "setup_required": True,
            "actor_label": None,
            "password_enabled": False,
            "oidc": {
                "enabled": False,
                "provider_name": None,
                "login_path": None,
            },
        }

        blocked = await client.get("/api/protected")
        assert blocked.status_code == 401
        assert blocked.json()["setup_required"] is True

        setup = await client.post("/api/auth/setup", json={"password": "very-secure-password"})
        assert setup.status_code == 200
        assert client.cookies.get("echonyx_session")
        assert client.cookies.get("echonyx_csrf")
        assert setup.json()["password_enabled"] is True

        allowed = await client.get("/api/protected")
        assert allowed.status_code == 200

        missing_csrf = await client.post("/api/protected", json={"ok": True})
        assert missing_csrf.status_code == 403

        with_csrf = await client.post(
            "/api/protected",
            json={"ok": True},
            headers={"X-CSRF-Token": client.cookies["echonyx_csrf"]},
        )
        assert with_csrf.status_code == 200
        assert with_csrf.json() == {"ok": True}


@pytest.mark.asyncio
async def test_login_rate_limit(monkeypatch, tmp_path):
    stack = await _build_auth_stack(
        monkeypatch,
        tmp_path,
        AUTH_PASSWORD_HASH="pbkdf2_sha256$390000$testsalt$5JzY7TZxF3_C1Cae5TEqyiwC-UVlA_sTzYTw4L11d5w=",
    )
    async with _client(stack) as client:
        first = await client.post("/api/auth/login", json={"password": "wrong-password"})
        second = await client.post("/api/auth/login", json={"password": "wrong-password"})
        third = await client.post("/api/auth/login", json={"password": "wrong-password"})

        assert first.status_code == 401
        assert second.status_code == 401
        assert third.status_code == 429


@pytest.mark.asyncio
async def test_json_request_limit(monkeypatch, tmp_path):
    stack = await _build_auth_stack(monkeypatch, tmp_path, MAX_JSON_REQUEST_BYTES="48")
    async with _client(stack) as client:
        response = await client.post(
            "/api/auth/setup",
            json={"password": "x" * 200},
        )

        assert response.status_code == 413


@pytest.mark.asyncio
async def test_upload_rate_limit(monkeypatch, tmp_path):
    stack = await _build_auth_stack(monkeypatch, tmp_path, UPLOAD_RATE_LIMIT_REQUESTS="2")
    async with _client(stack) as client:
        setup = await client.post("/api/auth/setup", json={"password": "very-secure-password"})
        assert setup.status_code == 200
        headers = {"X-CSRF-Token": client.cookies["echonyx_csrf"]}

        files = {"file": ("sample.mp4", b"video-bytes", "video/mp4")}
        first = await client.post("/api/videos/upload", files=files, headers=headers)
        second = await client.post("/api/videos/upload", files=files, headers=headers)
        third = await client.post("/api/videos/upload", files=files, headers=headers)

        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 429


@pytest.mark.asyncio
async def test_mutations_write_audit_logs(monkeypatch, tmp_path):
    stack = await _build_auth_stack(monkeypatch, tmp_path)
    async with _client(stack) as client:
        await client.post("/api/auth/setup", json={"password": "very-secure-password"})
        await client.post(
            "/api/protected",
            json={"ok": True},
            headers={"X-CSRF-Token": client.cookies["echonyx_csrf"]},
        )

    async with stack.database.async_session_maker() as db:
        result = await db.execute(select(stack.models.AuditLog).order_by(stack.models.AuditLog.created_at.asc()))
        logs = result.scalars().all()

    assert len(logs) >= 2
    assert logs[0].event == "auth.setup"
    assert any(log.path == "/api/protected" and log.status_code == 200 for log in logs)


@pytest.mark.asyncio
async def test_remote_setup_rejected_even_with_forwarded_loopback(monkeypatch, tmp_path):
    stack = await _build_auth_stack(monkeypatch, tmp_path)

    async with _client(stack, client_host="192.168.0.10", base_url="http://192.168.0.20:8000") as client:
        response = await client.post(
            "/api/auth/setup",
            json={"password": "very-secure-password"},
            headers={"X-Forwarded-For": "127.0.0.1"},
        )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert "scripts/bootstrap-admin.sh" in detail
    assert "AUTH_PASSWORD_HASH" in detail
    assert "OIDC" in detail


@pytest.mark.asyncio
async def test_setup_allows_explicit_cidr_override(monkeypatch, tmp_path):
    stack = await _build_auth_stack(
        monkeypatch,
        tmp_path,
        AUTH_SETUP_ALLOWED_CIDRS="127.0.0.1/32,::1/128,192.168.0.0/24",
    )

    async with _client(stack, client_host="192.168.0.10", base_url="https://192.168.0.20:8000") as client:
        response = await client.post("/api/auth/setup", json={"password": "very-secure-password"})

    assert response.status_code == 200
    assert response.json()["authenticated"] is True


@pytest.mark.asyncio
async def test_cross_origin_setup_rejected(monkeypatch, tmp_path):
    stack = await _build_auth_stack(monkeypatch, tmp_path)

    async with _client(stack) as client:
        response = await client.post(
            "/api/auth/setup",
            json={"password": "very-secure-password"},
            headers={"Origin": "http://192.168.0.10:3000"},
        )

    assert response.status_code == 403
    assert "cross-origin" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_remote_http_login_requires_https(monkeypatch, tmp_path):
    stack = await _build_auth_stack(
        monkeypatch,
        tmp_path,
        AUTH_PASSWORD_HASH="pbkdf2_sha256$390000$testsalt$5JzY7TZxF3_C1Cae5TEqyiwC-UVlA_sTzYTw4L11d5w=",
    )

    async with _client(stack, client_host="192.168.0.10", base_url="http://192.168.0.20:8000") as client:
        response = await client.post("/api/auth/login", json={"password": "wrong-password"})

    assert response.status_code == 400
    assert "https" in response.json()["detail"].lower()
