"""Duplicate video detection and suppression heuristics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from app.config import DuplicateHandlingPolicy, Settings
from app.models.video import Video

TOKEN_REGEX = re.compile(r"[a-z0-9]{3,}")
SINGLE_CHAR_REGEX = re.compile(r"^[A-Za-z0-9]$")
STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "been",
    "from",
    "have",
    "into",
    "just",
    "more",
    "only",
    "should",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "were",
    "with",
    "would",
}


@dataclass(frozen=True)
class DuplicateFingerprint:
    combined_text: str
    transcript_text: str
    key_point_text: str
    tokens: frozenset[str]
    bigrams: frozenset[str]


def build_duplicate_fingerprint(
    *,
    transcript: dict | None,
    summary: dict | None,
    title: str | None = None,
) -> DuplicateFingerprint:
    transcript_segments = []
    if transcript:
        for segment in (transcript.get("segments") or [])[:40]:
            text = _normalize_text(str(segment.get("text") or ""))
            if text:
                transcript_segments.append(text)

    key_points = []
    if summary:
        for point in (summary.get("key_points") or [])[:20]:
            text = _normalize_text(str(point or ""))
            if text:
                key_points.append(text)
        for topic in (summary.get("topics") or [])[:10]:
            topic_text = " ".join(
                part
                for part in (
                    _normalize_text(str(topic.get("topic") or "")),
                    _normalize_text(str(topic.get("summary") or "")),
                )
                if part
            )
            if topic_text:
                key_points.append(topic_text)
        executive_summary = _normalize_text(str(summary.get("executive_summary") or ""))
        if executive_summary:
            key_points.append(executive_summary)

    title_text = _normalize_text(title or "")
    combined_parts = [part for part in [title_text, *transcript_segments, *key_points] if part]
    combined_text = "\n".join(combined_parts)
    fingerprint_text = " ".join(part for part in [*transcript_segments, *key_points] if part) or combined_text
    combined_tokens = _tokens(fingerprint_text)

    return DuplicateFingerprint(
        combined_text=combined_text,
        transcript_text=" ".join(transcript_segments),
        key_point_text=" ".join(key_points),
        tokens=frozenset(combined_tokens),
        bigrams=frozenset(_bigrams(combined_tokens)),
    )


def evaluate_duplicate_match(
    *,
    source: DuplicateFingerprint,
    candidate: DuplicateFingerprint,
) -> float:
    if not source.tokens or not candidate.tokens:
        return 0.0

    overlap = source.tokens & candidate.tokens
    if not overlap:
        return 0.0

    union = source.tokens | candidate.tokens
    token_jaccard = len(overlap) / max(len(union), 1)
    containment = len(overlap) / max(min(len(source.tokens), len(candidate.tokens)), 1)

    bigram_overlap = source.bigrams & candidate.bigrams
    bigram_union = source.bigrams | candidate.bigrams
    bigram_jaccard = len(bigram_overlap) / max(len(bigram_union), 1) if bigram_union else 0.0

    source_text = source.transcript_text or source.key_point_text or source.combined_text
    candidate_text = candidate.transcript_text or candidate.key_point_text or candidate.combined_text
    sequence_ratio = SequenceMatcher(None, source_text[:4000], candidate_text[:4000]).ratio()

    score = (
        (containment * 0.4)
        + (sequence_ratio * 0.3)
        + (bigram_jaccard * 0.2)
        + (token_jaccard * 0.1)
    )
    return max(0.0, min(score, 1.0))


def classify_duplicate_score(
    score: float,
    *,
    exact_threshold: float,
    probable_threshold: float,
) -> str:
    if score >= exact_threshold:
        return "exact_duplicate"
    if score >= probable_threshold:
        return "probable_duplicate"
    return "distinct"


def duplicate_suppressed(
    classification: str,
    *,
    policy: DuplicateHandlingPolicy,
) -> bool:
    if policy == DuplicateHandlingPolicy.COLLAPSE_PROBABLE:
        return classification in {"exact_duplicate", "probable_duplicate"}
    if policy == DuplicateHandlingPolicy.COLLAPSE_EXACT:
        return classification == "exact_duplicate"
    return False


def best_duplicate_match(
    *,
    source_video: Video,
    candidate_videos: Sequence[Video],
    settings: Settings,
) -> dict | None:
    if settings.duplicate_detection_policy == DuplicateHandlingPolicy.OFF:
        return None

    source_fingerprint = build_duplicate_fingerprint(
        transcript=source_video.transcript,
        summary=source_video.summary,
        title=source_video.title or source_video.original_filename,
    )
    if not source_fingerprint.tokens:
        return None

    best_video = None
    best_score = 0.0

    for candidate in candidate_videos:
        if is_duplicate_suppressed(candidate):
            continue
        candidate_fingerprint = build_duplicate_fingerprint(
            transcript=candidate.transcript,
            summary=candidate.summary,
            title=candidate.title or candidate.original_filename,
        )
        score = evaluate_duplicate_match(source=source_fingerprint, candidate=candidate_fingerprint)
        if score > best_score:
            best_score = score
            best_video = candidate

    if best_video is None:
        return None

    classification = classify_duplicate_score(
        best_score,
        exact_threshold=settings.duplicate_exact_threshold,
        probable_threshold=settings.duplicate_probable_threshold,
    )

    return {
        "score": round(best_score, 4),
        "classification": classification,
        "suppressed": duplicate_suppressed(
            classification,
            policy=settings.duplicate_detection_policy,
        ),
        "policy": settings.duplicate_detection_policy.value,
        "thresholds": {
            "exact": settings.duplicate_exact_threshold,
            "probable": settings.duplicate_probable_threshold,
        },
        "representative_video_id": str(best_video.id),
        "representative_title": best_video.title or best_video.original_filename,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint_version": 1,
    }


def is_duplicate_suppressed(video: Video) -> bool:
    duplicate_info = video.duplicate_info or {}
    return bool(duplicate_info.get("suppressed"))


def _normalize_text(text: str) -> str:
    words = " ".join(text.split()).strip()
    if not words:
        return ""

    parts = words.split()
    collapsed: list[str] = []
    index = 0
    while index < len(parts):
        token = re.sub(r"[^A-Za-z0-9]", "", parts[index])
        if not SINGLE_CHAR_REGEX.match(token):
            collapsed.append(parts[index])
            index += 1
            continue

        run = [token]
        cursor = index + 1
        while cursor < len(parts):
            next_token = re.sub(r"[^A-Za-z0-9]", "", parts[cursor])
            if not SINGLE_CHAR_REGEX.match(next_token):
                break
            run.append(next_token)
            cursor += 1

        if len(run) >= 3:
            collapsed.append("".join(run))
            index = cursor
            continue

        collapsed.append(parts[index])
        index += 1

    return " ".join(collapsed)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_REGEX.findall(text.lower())
        if token not in STOPWORDS
    ]


def _bigrams(tokens: Iterable[str]) -> list[str]:
    token_list = list(tokens)
    return [
        f"{token_list[index]}::{token_list[index + 1]}"
        for index in range(len(token_list) - 1)
    ]
