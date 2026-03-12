import types
import uuid
from datetime import UTC, datetime, timedelta

from app.config import DuplicateHandlingPolicy
from app.core.duplicates import (
    best_duplicate_match,
    build_duplicate_fingerprint,
    classify_duplicate_score,
    evaluate_duplicate_match,
    is_duplicate_suppressed,
)
from app.models.video import Video


def _video(*, title: str, transcript_text: str, summary_text: str | None = None, duplicate_info=None) -> Video:
    summary = None
    if summary_text:
        summary = {
            "executive_summary": summary_text,
            "key_points": [summary_text],
            "topics": [],
        }
    return Video(
        id=uuid.uuid4(),
        filename=f"{title}.mp4",
        original_filename=f"{title}.mp4",
        file_path=f"/tmp/{title}.mp4",
        file_size=1,
        mime_type="video/mp4",
        title=title,
        transcript={
            "segments": [
                {"start": 0.0, "end": 4.0, "text": transcript_text, "speaker": "Speaker 1"},
            ]
        },
        summary=summary,
        duplicate_info=duplicate_info,
        created_at=datetime.now(UTC) - timedelta(minutes=1),
    )


def test_duplicate_fingerprint_collapses_spaced_letter_tokens():
    fingerprint = build_duplicate_fingerprint(
        transcript={"segments": [{"text": "R O C M is enabled on Strix Halo."}]},
        summary=None,
        title="Probe",
    )

    assert "rocm" in fingerprint.tokens


def test_evaluate_duplicate_match_scores_near_identical_content_highly():
    source = build_duplicate_fingerprint(
        transcript={"segments": [{"text": "The budget review is due Friday. ROCM is enabled."}]},
        summary=None,
        title="Probe One",
    )
    candidate = build_duplicate_fingerprint(
        transcript={"segments": [{"text": "The budget review is due Friday. R O C M is enabled."}]},
        summary=None,
        title="Probe Two",
    )

    assert evaluate_duplicate_match(source=source, candidate=candidate) >= 0.95


def test_best_duplicate_match_respects_policy_and_skips_suppressed_candidates():
    representative = _video(
        title="Representative",
        transcript_text="The budget review is due Friday. ROCM is enabled.",
        summary_text="Budget review due Friday.",
    )
    suppressed = _video(
        title="Suppressed",
        transcript_text="The budget review is due Friday. ROCM is enabled.",
        summary_text="Budget review due Friday.",
        duplicate_info={"suppressed": True},
    )
    source = _video(
        title="New Upload",
        transcript_text="The budget review is due Friday. R O C M is enabled.",
        summary_text="Budget review due Friday.",
    )
    settings = types.SimpleNamespace(
        duplicate_detection_policy=DuplicateHandlingPolicy.COLLAPSE_EXACT,
        duplicate_exact_threshold=0.95,
        duplicate_probable_threshold=0.85,
    )

    match = best_duplicate_match(
        source_video=source,
        candidate_videos=[suppressed, representative],
        settings=settings,
    )

    assert match is not None
    assert match["representative_video_id"] == str(representative.id)
    assert match["classification"] == "exact_duplicate"
    assert match["suppressed"] is True
    assert is_duplicate_suppressed(suppressed) is True
    assert classify_duplicate_score(0.9, exact_threshold=0.95, probable_threshold=0.85) == "probable_duplicate"
