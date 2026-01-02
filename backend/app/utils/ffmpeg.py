"""FFmpeg utilities for audio/video processing."""

import asyncio
import logging
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)


async def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Extract audio from video file.

    Args:
        video_path: Path to the video file
        output_path: Path for the output audio file
        sample_rate: Audio sample rate (default 16kHz for Whisper)
        channels: Number of audio channels (default mono)

    Returns:
        Path to the extracted audio file
    """
    loop = asyncio.get_event_loop()

    def do_extract():
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(output_path),
                    acodec="pcm_s16le",
                    ar=sample_rate,
                    ac=channels,
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Audio extracted: {output_path}")
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise

    return await loop.run_in_executor(None, do_extract)


async def get_video_info(video_path: Path) -> dict:
    """
    Get video metadata using ffprobe.

    Returns:
        Dictionary with video info:
        {
            "duration": float,
            "width": int,
            "height": int,
            "fps": float,
            "codec": str,
            "audio_codec": str,
        }
    """
    loop = asyncio.get_event_loop()

    def do_probe():
        try:
            probe = ffmpeg.probe(str(video_path))

            # Find video stream
            video_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "video"),
                None
            )

            # Find audio stream
            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"),
                None
            )

            result = {
                "duration": float(probe["format"].get("duration", 0)),
            }

            if video_stream:
                result.update({
                    "width": video_stream.get("width"),
                    "height": video_stream.get("height"),
                    "codec": video_stream.get("codec_name"),
                })

                # Calculate FPS
                fps_str = video_stream.get("r_frame_rate", "0/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    result["fps"] = float(num) / float(den) if float(den) > 0 else 0
                else:
                    result["fps"] = float(fps_str)

            if audio_stream:
                result["audio_codec"] = audio_stream.get("codec_name")

            return result

        except ffmpeg.Error as e:
            logger.error(f"FFprobe error: {e.stderr.decode() if e.stderr else str(e)}")
            raise

    return await loop.run_in_executor(None, do_probe)


async def extract_frame(
    video_path: Path,
    output_path: Path,
    timestamp: float,
) -> Path:
    """
    Extract a single frame from video at a specific timestamp.

    Args:
        video_path: Path to the video file
        output_path: Path for the output image
        timestamp: Time in seconds to extract frame

    Returns:
        Path to the extracted frame
    """
    loop = asyncio.get_event_loop()

    def do_extract():
        try:
            (
                ffmpeg
                .input(str(video_path), ss=timestamp)
                .output(str(output_path), vframes=1, format="image2")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"Frame extraction error: {e.stderr.decode() if e.stderr else str(e)}")
            raise

    return await loop.run_in_executor(None, do_extract)


async def extract_frames_interval(
    video_path: Path,
    output_dir: Path,
    interval: float = 5.0,
) -> list[dict]:
    """
    Extract frames at regular intervals.

    Args:
        video_path: Path to the video file
        output_dir: Directory for output frames
        interval: Interval between frames in seconds

    Returns:
        List of extracted frame info:
        [{"timestamp": float, "path": str}, ...]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video duration
    info = await get_video_info(video_path)
    duration = info.get("duration", 0)

    if duration == 0:
        return []

    frames = []
    timestamp = 0.0

    while timestamp < duration:
        frame_path = output_dir / f"frame_{timestamp:.2f}.jpg"
        await extract_frame(video_path, frame_path, timestamp)

        frames.append({
            "timestamp": timestamp,
            "path": str(frame_path),
        })

        timestamp += interval

    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames
