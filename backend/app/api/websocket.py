"""WebSocket endpoints for real-time progress updates."""

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.database import async_session_maker
from app.models.job import Job

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections for job progress updates."""

    def __init__(self):
        # job_id -> list of websocket connections
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept a new WebSocket connection for a job."""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str):
        """Remove a WebSocket connection."""
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast_to_job(self, job_id: str, message: dict[str, Any]):
        """Send a message to all connections watching a job."""
        if job_id not in self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections[job_id]:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn, job_id)


manager = ConnectionManager()


@router.websocket("/jobs/{job_id}")
async def job_progress_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress updates.

    Messages sent:
    - {"type": "status", "status": "processing", "progress": 45.5, "current_step": "transcription"}
    - {"type": "step_update", "step": "transcription", "progress": 80, "eta_seconds": 30}
    - {"type": "completed", "status": "completed"}
    - {"type": "error", "message": "...", "step": "transcription"}
    """
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        await websocket.close(code=4000, reason="Invalid job ID")
        return

    await manager.connect(websocket, job_id)

    try:
        # Send initial status
        async with async_session_maker() as session:
            result = await session.execute(select(Job).where(Job.id == jid))
            job = result.scalar_one_or_none()

            if not job:
                await websocket.send_json({
                    "type": "error",
                    "message": "Job not found",
                })
                return

            await websocket.send_json({
                "type": "status",
                "status": job.status,
                "progress": job.progress,
                "current_step": job.current_step,
                "step_progress": job.step_progress,
            })

        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for client messages (ping/pong or close)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )

                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, job_id)


async def notify_job_progress(
    job_id: str,
    status: str,
    progress: float,
    current_step: str | None = None,
    step_progress: dict | None = None,
):
    """Notify all watchers of a job about progress update."""
    await manager.broadcast_to_job(
        job_id,
        {
            "type": "status",
            "status": status,
            "progress": progress,
            "current_step": current_step,
            "step_progress": step_progress,
        },
    )


async def notify_job_step(
    job_id: str,
    step: str,
    progress: float,
    eta_seconds: int | None = None,
):
    """Notify all watchers about a step update."""
    await manager.broadcast_to_job(
        job_id,
        {
            "type": "step_update",
            "step": step,
            "progress": progress,
            "eta_seconds": eta_seconds,
        },
    )


async def notify_job_completed(job_id: str):
    """Notify all watchers that job is completed."""
    await manager.broadcast_to_job(
        job_id,
        {
            "type": "completed",
            "status": "completed",
        },
    )


async def notify_job_error(job_id: str, message: str, step: str | None = None):
    """Notify all watchers about an error."""
    await manager.broadcast_to_job(
        job_id,
        {
            "type": "error",
            "message": message,
            "step": step,
        },
    )
