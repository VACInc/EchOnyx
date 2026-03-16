"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import action_items, auth, batch, jobs, search, settings as settings_routes, summaries, videos
from app.api.websocket import router as ws_router
from app.auth import cleanup_security_state
from app.config import get_settings
from app.http_security import security_http_middleware
from app.security import cors_configuration


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

    return app


app = create_app()
