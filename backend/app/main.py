"""FastAPI application entry point."""

import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis import asyncio as redis_async
from sqlalchemy import text

from app.api.routes import action_items, auth, batch, jobs, search, settings as settings_routes, summaries, videos
from app.api.websocket import router as ws_router
from app.auth import cleanup_security_state
from app.config import get_settings
from app.database import async_session_maker
from app.http_security import security_http_middleware
from app.security import cors_configuration

READINESS_TIMEOUT_SECONDS = 2.0


def _ready_ok() -> dict[str, str]:
    return {"status": "ok"}


def _ready_error(exc: Exception | str) -> dict[str, str]:
    detail = exc if isinstance(exc, str) else exc.__class__.__name__
    return {"status": "error", "detail": str(detail)}


async def _check_database_ready() -> dict[str, str]:
    try:
        async with async_session_maker() as session:
            await asyncio.wait_for(
                session.execute(text("SELECT 1")),
                timeout=READINESS_TIMEOUT_SECONDS,
            )
        return _ready_ok()
    except Exception as exc:
        return _ready_error(exc)


async def _check_redis_ready(settings) -> dict[str, str]:
    if not settings.redis_url:
        return _ready_error("Redis URL is not configured")

    client = redis_async.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=READINESS_TIMEOUT_SECONDS,
        socket_timeout=READINESS_TIMEOUT_SECONDS,
    )
    try:
        await asyncio.wait_for(client.ping(), timeout=READINESS_TIMEOUT_SECONDS)
        return _ready_ok()
    except Exception as exc:
        return _ready_error(exc)
    finally:
        close = getattr(client, "aclose", None)
        if close is not None:
            await close()
        else:  # pragma: no cover - compatibility with older redis-py
            await client.close()


def _check_chroma_ready(settings) -> dict[str, str]:
    try:
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        marker = settings.chroma_persist_dir / f".ready-{uuid.uuid4().hex}.tmp"
        marker.write_text("ok", encoding="utf-8")
        marker.unlink(missing_ok=True)
        return _ready_ok()
    except Exception as exc:
        return _ready_error(exc)


async def collect_readiness_checks(settings) -> dict[str, dict[str, str]]:
    database, redis = await asyncio.gather(
        _check_database_ready(),
        _check_redis_ready(settings),
    )
    return {
        "database": database,
        "redis": redis,
        "chroma": _check_chroma_ready(settings),
    }


async def get_readiness_status(settings) -> tuple[dict, int]:
    checks = await collect_readiness_checks(settings)
    failed = [name for name, check in checks.items() if check.get("status") != "ok"]
    payload = {
        "status": "not_ready" if failed else "ready",
        "checks": checks,
    }
    if failed:
        payload["failed"] = failed
    return payload, 503 if failed else 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    import asyncio
    import logging
    from app.database import init_db

    logger = logging.getLogger(__name__)
    settings = get_settings()

    # Ensure directories exist
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    # Initialize database tables
    logger.info("Initializing database tables...")
    await init_db()
    logger.info("Database tables initialized")
    await cleanup_security_state()

    # Requeue orphaned jobs that never received a Celery task id
    recovery_task = None
    try:
        from app.database import async_session_maker
        from app.workers.enqueue import requeue_orphaned_jobs, recover_stale_processing_jobs

        async with async_session_maker() as session:
            requeued = await requeue_orphaned_jobs(session)
            if requeued:
                logger.info("Requeued %d orphaned queued job(s)", requeued)
            recovered = await recover_stale_processing_jobs(session)
            if recovered:
                logger.info("Recovered %d stale processing job(s)", recovered)

        async def recovery_loop():
            while True:
                try:
                    async with async_session_maker() as session:
                        await requeue_orphaned_jobs(session)
                        await recover_stale_processing_jobs(session)
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning("Job recovery loop failed: %s", exc)
                await asyncio.sleep(300)

        recovery_task = asyncio.create_task(recovery_loop())
    except Exception as exc:  # pragma: no cover - startup best effort
        logger.warning("Failed to requeue orphaned jobs: %s", exc)

    yield

    # Cleanup
    logger.info("Shutting down...")
    if recovery_task:
        recovery_task.cancel()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Local video and presentation summarization system",
        version="0.1.0",
        lifespan=lifespan,
    )

    allowed_origins, allow_origin_regex = cors_configuration(settings)

    # CORS middleware - scoped to explicit origins plus local/private-network browsers by default
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=allow_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(security_http_middleware)

    # Include routers
    app.include_router(auth.router, prefix=f"{settings.api_prefix}/auth", tags=["auth"])
    app.include_router(videos.router, prefix=f"{settings.api_prefix}/videos", tags=["videos"])
    app.include_router(action_items.router, prefix=f"{settings.api_prefix}/action-items", tags=["action-items"])
    app.include_router(jobs.router, prefix=f"{settings.api_prefix}/jobs", tags=["jobs"])
    app.include_router(batch.router, prefix=f"{settings.api_prefix}/batch", tags=["batch"])
    app.include_router(summaries.router, prefix=f"{settings.api_prefix}/summaries", tags=["summaries"])
    app.include_router(search.router, prefix=f"{settings.api_prefix}/search", tags=["search"])
    app.include_router(settings_routes.router, prefix=f"{settings.api_prefix}/settings", tags=["settings"])
    app.include_router(ws_router, prefix=f"{settings.api_prefix}/ws", tags=["websocket"])

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/ready")
    async def readiness_check():
        """Readiness check endpoint."""
        payload, status_code = await get_readiness_status(settings)
        return JSONResponse(payload, status_code=status_code)

    return app


app = create_app()
