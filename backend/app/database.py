"""Database connection and session management."""

from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings

settings = get_settings()


def _build_engine(*, use_null_pool: bool = False):
    kwargs = {
        "echo": settings.debug,
        "pool_pre_ping": True,
    }
    if use_null_pool:
        kwargs["poolclass"] = NullPool
    else:
        kwargs["pool_size"] = 10
        kwargs["max_overflow"] = 20
    return create_async_engine(settings.database_url, **kwargs)


def _build_session_maker(*, use_null_pool: bool = False):
    engine = _build_engine(use_null_pool=use_null_pool)
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


engine = _build_engine()
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@lru_cache(maxsize=1)
def get_worker_async_session_maker():
    """Create a loop-safe sessionmaker for Celery workers."""
    return _build_session_maker(use_null_pool=True)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database sessions."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    from app.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_schema_updates(conn)


async def _ensure_schema_updates(conn) -> None:
    video_columns = {
        column["name"]
        for column in await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_columns("videos"))
    }

    if "duplicate_info" not in video_columns:
        await conn.execute(text("ALTER TABLE videos ADD COLUMN duplicate_info JSON"))
