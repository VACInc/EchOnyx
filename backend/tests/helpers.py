import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path


def create_sample_video(
    path: Path,
    *,
    color: str,
    frequency: int,
    duration: float = 1.2,
) -> Path:
    """Generate a tiny MP4 fixture with ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=320x240:d={duration}:r=24",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={frequency}:duration={duration}:sample_rate=16000",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


class StreamingUpload:
    """Upload-like object that fails if code tries to read the whole file at once."""

    def __init__(
        self,
        filename: str,
        data: bytes,
        *,
        content_type: str = "video/mp4",
        chunk_limit: int = 512 * 1024,
    ):
        self.filename = filename
        self.content_type = content_type
        self._data = data
        self._offset = 0
        self._chunk_limit = chunk_limit
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            raise AssertionError("Upload must be read in chunks, not all at once.")
        if size > self._chunk_limit:
            size = self._chunk_limit
        if self._offset >= len(self._data):
            return b""
        end = min(self._offset + size, len(self._data))
        chunk = self._data[self._offset:end]
        self._offset = end
        return chunk


class SequenceResult:
    def __init__(self, scalar=None, items: Sequence | None = None):
        self._scalar = scalar
        self._items = list(items or [])

    def scalar_one_or_none(self):
        return self._scalar

    def scalar_one(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return list(self._items)


def ensure_timestamp_defaults(obj) -> None:
    now = datetime.now(timezone.utc)
    if getattr(obj, "created_at", None) is None:
        obj.created_at = now
    if getattr(obj, "updated_at", None) is None:
        obj.updated_at = now
