"""Scene detection and keyframe extraction."""

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


async def extract_keyframes(
    video_path: Path,
    output_dir: Path,
) -> list[dict]:
    """
    Extract keyframes from video using change detection with persistence.

    Args:
        video_path: Path to the video file
        output_dir: Directory for output frames

    Returns:
        List of extracted frame info:
        [{"timestamp": float, "path": str, "scene_idx": int}, ...]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    loop = asyncio.get_event_loop()

    def do_extract():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Failed to open video for keyframe extraction: %s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = frame_count / fps if fps > 0 else 0.0
        if duration <= 0:
            logger.warning("Could not determine duration for %s", video_path)
            cap.release()
            return []

        sample_interval = max(settings.frame_persistence_seconds, 0.5)
        max_keyframes = settings.max_keyframes
        resize_width = max(settings.frame_resize_width, 32)

        def frame_signature(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            if width != resize_width:
                new_height = max(1, int(height * (resize_width / width)))
                gray = cv2.resize(gray, (resize_width, new_height), interpolation=cv2.INTER_AREA)
            return gray

        def mean_abs_diff(a, b):
            return float(np.mean(cv2.absdiff(a, b)))

        def dhash(gray):
            resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            diff = resized[:, 1:] > resized[:, :-1]
            return diff.flatten().astype(np.uint8)

        def hash_distance(a, b):
            if a is None or b is None:
                return 999
            return int(np.count_nonzero(a != b))

        frames = []
        last_hash = None
        baseline_sig = None
        candidate = None
        candidate_frame = None
        candidate_time = 0.0

        idx = 0
        t = 0.0
        while t < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                t += sample_interval
                continue

            sig = frame_signature(frame)

            if baseline_sig is None:
                frame_path = output_dir / f"frame_{idx:04d}_{t:.2f}.jpg"
                cv2.imwrite(str(frame_path), frame)
                baseline_sig = sig
                last_hash = dhash(sig)
                frames.append({
                    "timestamp": t,
                    "path": str(frame_path),
                    "scene_idx": idx,
                    "reason": "initial",
                })
                idx += 1
                t += sample_interval
                continue

            diff_from_baseline = mean_abs_diff(sig, baseline_sig)

            if candidate is None:
                if diff_from_baseline >= settings.frame_change_threshold:
                    candidate = sig
                    candidate_frame = frame
                    candidate_time = t
            else:
                diff_from_candidate = mean_abs_diff(sig, candidate)
                if diff_from_candidate <= settings.frame_stability_threshold:
                    frame_hash = dhash(sig)
                    if hash_distance(frame_hash, last_hash) >= settings.frame_dedupe_threshold:
                        frame_path = output_dir / f"frame_{idx:04d}_{candidate_time:.2f}.jpg"
                        cv2.imwrite(str(frame_path), candidate_frame)
                        frames.append({
                            "timestamp": candidate_time,
                            "path": str(frame_path),
                            "scene_idx": idx,
                            "reason": "change",
                        })
                        idx += 1
                        last_hash = frame_hash
                        baseline_sig = candidate
                        if max_keyframes > 0 and len(frames) >= max_keyframes:
                            break
                    candidate = None
                    candidate_frame = None
                elif diff_from_baseline < settings.frame_change_threshold:
                    candidate = None
                    candidate_frame = None
                else:
                    candidate = sig
                    candidate_frame = frame
                    candidate_time = t

            t += sample_interval

        cap.release()

        if not frames or (len(frames) <= 1 and duration > settings.keyframe_extraction_interval):
            logger.info(
                "Dynamic keyframe extraction produced %d frame(s); falling back to interval sampling.",
                len(frames),
            )
            fallback_max = settings.max_keyframes
            if fallback_max <= 0:
                fallback_max = int(duration / settings.keyframe_extraction_interval) + 1
            return extract_interval_frames(
                video_path,
                output_dir,
                max_frames=fallback_max,
                interval=settings.keyframe_extraction_interval,
            )

        return frames

    return await loop.run_in_executor(None, do_extract)


def extract_interval_frames(
    video_path: Path,
    output_dir: Path,
    max_frames: int,
    interval: float = 10.0,
) -> list[dict]:
    """
    Extract frames at regular intervals (fallback when scene detection fails).

    Args:
        video_path: Path to the video file
        output_dir: Directory for output frames
        max_frames: Maximum number of frames
        interval: Interval between frames in seconds

    Returns:
        List of extracted frame info
    """
    import ffmpeg

    # Get video duration
    try:
        probe = ffmpeg.probe(str(video_path))
        duration = float(probe["format"]["duration"])
    except Exception:
        duration = 0

    if duration == 0:
        return []

    # Calculate actual interval
    if max_frames <= 0:
        max_frames = int(duration / interval) + 1
    num_frames = min(max_frames, int(duration / interval) + 1)
    actual_interval = duration / num_frames if num_frames > 0 else interval

    frames = []
    timestamp = 0.0

    for idx in range(num_frames):
        frame_path = output_dir / f"frame_{idx:04d}_{timestamp:.2f}.jpg"

        try:
            (
                ffmpeg
                .input(str(video_path), ss=timestamp)
                .output(str(frame_path), vframes=1, format="image2", qscale=2)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            frames.append({
                "timestamp": timestamp,
                "path": str(frame_path),
                "scene_idx": idx,
            })
        except Exception as e:
            logger.warning(f"Failed to extract frame at {timestamp}: {e}")

        timestamp += actual_interval

    return frames


async def detect_slide_changes(
    video_path: Path,
    threshold: float = 15.0,
) -> list[float]:
    """
    Detect potential slide changes in a presentation video.

    Uses a lower threshold to catch subtle slide transitions.

    Args:
        video_path: Path to the video file
        threshold: Detection threshold (lower = more sensitive)

    Returns:
        List of timestamps where slides changed
    """
    loop = asyncio.get_event_loop()

    def do_detect():
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=30)  # ~1 second at 30fps
        )

        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        return [start.get_seconds() for start, _ in scene_list]

    return await loop.run_in_executor(None, do_detect)
