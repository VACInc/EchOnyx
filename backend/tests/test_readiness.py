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

    async def worker_ok(_settings):
        return {"status": "ok"}

    monkeypatch.setattr(main, "_check_database_ready", database_ok)
    monkeypatch.setattr(main, "_check_redis_ready", redis_ok)
    monkeypatch.setattr(main, "_check_worker_ready", worker_ok)
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
            "worker": {"status": "ok"},
        },
    }


@pytest.mark.asyncio
async def test_get_readiness_status_reports_failed_components(monkeypatch, tmp_path):
    async def database_ok():
        return {"status": "ok"}

    async def redis_error(_settings):
        return {"status": "error", "detail": "ConnectionError"}

    async def worker_ok(_settings):
        return {"status": "ok"}

    monkeypatch.setattr(main, "_check_database_ready", database_ok)
    monkeypatch.setattr(main, "_check_redis_ready", redis_error)
    monkeypatch.setattr(main, "_check_worker_ready", worker_ok)
    monkeypatch.setattr(main, "_check_chroma_ready", lambda _settings: {"status": "ok"})

    payload, status_code = await main.get_readiness_status(
        SimpleNamespace(chroma_persist_dir=tmp_path, redis_url="redis://localhost:6379/0")
    )

    assert status_code == 503
    assert payload["status"] == "not_ready"
    assert payload["failed"] == ["redis"]
    assert payload["checks"]["redis"] == {"status": "error", "detail": "ConnectionError"}


@pytest.mark.asyncio
async def test_get_readiness_status_degrades_when_only_worker_is_missing(monkeypatch, tmp_path):
    async def database_ok():
        return {"status": "ok"}

    async def redis_ok(_settings):
        return {"status": "ok"}

    async def worker_missing(_settings):
        return {"status": "error", "detail": "no recent worker heartbeat"}

    monkeypatch.setattr(main, "_check_database_ready", database_ok)
    monkeypatch.setattr(main, "_check_redis_ready", redis_ok)
    monkeypatch.setattr(main, "_check_worker_ready", worker_missing)
    monkeypatch.setattr(main, "_check_chroma_ready", lambda _settings: {"status": "ok"})

    payload, status_code = await main.get_readiness_status(
        SimpleNamespace(chroma_persist_dir=tmp_path, redis_url="redis://localhost:6379/0")
    )

    assert status_code == 200
    assert payload["status"] == "degraded"
    assert payload["degraded"] == ["worker"]
    assert "failed" not in payload
    assert payload["checks"]["worker"] == {
        "status": "error",
        "detail": "no recent worker heartbeat",
    }

    strict_payload, strict_status_code = await main.get_readiness_status(
        SimpleNamespace(chroma_persist_dir=tmp_path, redis_url="redis://localhost:6379/0"),
        strict=True,
    )

    assert strict_status_code == 503
    assert strict_payload["status"] == "not_ready"
    assert strict_payload["failed"] == ["worker"]
    assert "degraded" not in strict_payload


def test_chroma_ready_check_writes_to_persist_dir(tmp_path):
    response = main._check_chroma_ready(SimpleNamespace(chroma_persist_dir=tmp_path / "chroma"))

    assert response == {"status": "ok"}
    assert (tmp_path / "chroma").is_dir()


@pytest.mark.asyncio
async def test_worker_ready_check_reads_heartbeat_key(monkeypatch):
    class FakeRedis:
        def __init__(self):
            self.closed = False
            self.key = ""

        async def exists(self, key):
            self.key = key
            return 1

        async def aclose(self):
            self.closed = True

    client = FakeRedis()
    monkeypatch.setattr(main.redis_async, "from_url", lambda *_args, **_kwargs: client)

    response = await main._check_worker_ready(SimpleNamespace(redis_url="redis://localhost:6379/0"))

    assert response == {"status": "ok"}
    assert client.key == "echonyx:worker:heartbeat"
    assert client.closed is True


@pytest.mark.asyncio
async def test_worker_ready_check_reports_missing_heartbeat(monkeypatch):
    class FakeRedis:
        async def exists(self, _key):
            return 0

        async def aclose(self):
            pass

    monkeypatch.setattr(main.redis_async, "from_url", lambda *_args, **_kwargs: FakeRedis())

    response = await main._check_worker_ready(SimpleNamespace(redis_url="redis://localhost:6379/0"))

    assert response == {"status": "error", "detail": "no recent worker heartbeat"}


@pytest.mark.asyncio
async def test_ready_endpoint_uses_readiness_status(monkeypatch):
    async def ready_status(_settings, *, strict=False):
        assert strict is False
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


@pytest.mark.asyncio
async def test_ready_endpoint_passes_strict_query(monkeypatch):
    seen = {}

    async def ready_status(_settings, *, strict=False):
        seen["strict"] = strict
        return {"status": "not_ready", "checks": {}, "failed": ["worker"]}, 503

    monkeypatch.setattr(main, "get_readiness_status", ready_status)
    app = main.create_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ready?strict=1")

    assert response.status_code == 503
    assert seen["strict"] is True
