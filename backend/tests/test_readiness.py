from types import SimpleNamespace

import httpx
import pytest

from app import main


@pytest.mark.asyncio
async def test_get_readiness_status_reports_ready(monkeypatch, tmp_path):
    async def database_ok():
        return {"status": "ok"}

    async def redis_ok(_settings):
        return {"status": "ok"}

    monkeypatch.setattr(main, "_check_database_ready", database_ok)
    monkeypatch.setattr(main, "_check_redis_ready", redis_ok)
    monkeypatch.setattr(main, "_check_chroma_ready", lambda _settings: {"status": "ok"})

    payload, status_code = await main.get_readiness_status(
        SimpleNamespace(chroma_persist_dir=tmp_path, redis_url="redis://localhost:6379/0")
    )

    assert status_code == 200
    assert payload == {
        "status": "ready",
        "checks": {
            "database": {"status": "ok"},
            "redis": {"status": "ok"},
            "chroma": {"status": "ok"},
        },
    }


@pytest.mark.asyncio
async def test_get_readiness_status_reports_failed_components(monkeypatch, tmp_path):
    async def database_ok():
        return {"status": "ok"}

    async def redis_error(_settings):
        return {"status": "error", "detail": "ConnectionError"}

    monkeypatch.setattr(main, "_check_database_ready", database_ok)
    monkeypatch.setattr(main, "_check_redis_ready", redis_error)
    monkeypatch.setattr(main, "_check_chroma_ready", lambda _settings: {"status": "ok"})

    payload, status_code = await main.get_readiness_status(
        SimpleNamespace(chroma_persist_dir=tmp_path, redis_url="redis://localhost:6379/0")
    )

    assert status_code == 503
    assert payload["status"] == "not_ready"
    assert payload["failed"] == ["redis"]
    assert payload["checks"]["redis"] == {"status": "error", "detail": "ConnectionError"}


def test_chroma_ready_check_writes_to_persist_dir(tmp_path):
    response = main._check_chroma_ready(SimpleNamespace(chroma_persist_dir=tmp_path / "chroma"))

    assert response == {"status": "ok"}
    assert (tmp_path / "chroma").is_dir()


@pytest.mark.asyncio
async def test_ready_endpoint_uses_readiness_status(monkeypatch):
    async def ready_status(_settings):
        return {
            "status": "not_ready",
            "failed": ["database"],
            "checks": {
                "database": {"status": "error", "detail": "TimeoutError"},
                "redis": {"status": "ok"},
                "chroma": {"status": "ok"},
            },
        }, 503

    monkeypatch.setattr(main, "get_readiness_status", ready_status)
    app = main.create_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json()["failed"] == ["database"]
