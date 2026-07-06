"""Celery worker liveness heartbeat."""

import logging
import threading
import time

from redis import Redis

WORKER_HEARTBEAT_KEY = "echonyx:worker:heartbeat"
WORKER_HEARTBEAT_INTERVAL_SECONDS = 30
WORKER_HEARTBEAT_TTL_SECONDS = 90

logger = logging.getLogger(__name__)

_heartbeat_lock = threading.Lock()
_heartbeat_stop = threading.Event()
_heartbeat_thread: threading.Thread | None = None


def _write_worker_heartbeat(client: Redis) -> None:
    client.set(
        WORKER_HEARTBEAT_KEY,
        str(time.time()),
        ex=WORKER_HEARTBEAT_TTL_SECONDS,
    )


def _heartbeat_loop(redis_url: str) -> None:
    client: Redis | None = None
    while not _heartbeat_stop.is_set():
        try:
            if client is None:
                client = Redis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )
            _write_worker_heartbeat(client)
        except Exception as exc:  # pragma: no cover - runtime best effort
            logger.warning("Worker heartbeat write failed: %s", exc)
            if client is not None:
                client.close()
                client = None
        _heartbeat_stop.wait(WORKER_HEARTBEAT_INTERVAL_SECONDS)

    if client is not None:
        client.close()


def start_worker_heartbeat(redis_url: str) -> None:
    """Start the best-effort worker liveness heartbeat thread."""
    global _heartbeat_thread

    if not redis_url:
        logger.warning("Worker heartbeat disabled because Redis URL is not configured")
        return

    with _heartbeat_lock:
        if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
            return
        _heartbeat_stop.clear()
        _heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(redis_url,),
            name="echonyx-worker-heartbeat",
            daemon=True,
        )
        _heartbeat_thread.start()


def stop_worker_heartbeat() -> None:
    """Stop the heartbeat thread during worker shutdown."""
    _heartbeat_stop.set()
