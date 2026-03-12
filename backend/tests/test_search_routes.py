import uuid
import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.api.routes import search as search_module
from app.api.routes.search import (
    RAGQuestion,
    SearchWarmRequest,
    ask_question,
    find_similar,
    search,
    warm_search_runtime,
)
from app.models.video import Video
from tests.helpers import SequenceResult


class DummySession:
    def __init__(self, results):
        self._results = list(results)

    async def execute(self, _query):
        return self._results.pop(0)


class DummyManager:
    def __init__(self, loaded_models=None):
        self._loaded_models = list(loaded_models or [])

    def get_loaded_models(self):
        return list(self._loaded_models)


@pytest.mark.asyncio
async def test_search_uses_semantic_results_with_tag_filter(monkeypatch):
    kept_video = Video(
        id=uuid.uuid4(),
        filename="keep.mp4",
        original_filename="keep.mp4",
        file_path="/tmp/keep.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Quarterly Review",
        tags=["Finance"],
        summary={"executive_summary": "Budget review"},
    )
    skipped_video = Video(
        id=uuid.uuid4(),
        filename="skip.mp4",
        original_filename="skip.mp4",
        file_path="/tmp/skip.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Engineering Notes",
        tags=["Engineering"],
        summary={"executive_summary": "Build review"},
    )

    captured: dict[str, object] = {}

    async def fake_search_content(query: str, video_ids=None, n_results: int = 0, **_kwargs):
        captured["query"] = query
        captured["video_ids"] = video_ids
        captured["n_results"] = n_results
        return [
            {
                "text": "Budget risks were reduced",
                "metadata": {
                    "video_id": str(kept_video.id),
                    "type": "summary",
                    "section": "executive_summary",
                },
                "score": 0.88,
            },
        ]

    monkeypatch.setattr(search_module, "search_content", fake_search_content)

    db = DummySession([SequenceResult(items=[kept_video, skipped_video])])
    response = await search(q="budget", tags=["finance"], limit=5, db=db)

    assert response.total >= 1
    assert response.results[0].video_id == str(kept_video.id)
    assert response.results[0].context == "summary section: executive_summary"
    assert captured["video_ids"] == [str(kept_video.id)]


@pytest.mark.asyncio
async def test_search_skips_suppressed_duplicate_videos_by_default(monkeypatch):
    kept_video = Video(
        id=uuid.uuid4(),
        filename="keep.mp4",
        original_filename="keep.mp4",
        file_path="/tmp/keep.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Representative",
        summary={"executive_summary": "Budget review"},
    )
    duplicate_video = Video(
        id=uuid.uuid4(),
        filename="dup.mp4",
        original_filename="dup.mp4",
        file_path="/tmp/dup.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Duplicate",
        summary={"executive_summary": "Budget review"},
        duplicate_info={"suppressed": True, "representative_video_id": str(kept_video.id)},
    )

    async def fake_search_content(query: str, video_ids=None, n_results: int = 0, **_kwargs):
        return [
            {
                "text": "Budget review",
                "metadata": {
                    "video_id": str(kept_video.id),
                    "type": "summary",
                    "section": "executive_summary",
                },
                "score": 0.9,
            },
        ]

    monkeypatch.setattr(search_module, "search_content", fake_search_content)

    db = DummySession([SequenceResult(items=[kept_video, duplicate_video])])
    response = await search(q="budget", tags=None, limit=5, db=db)

    assert response.total == 1
    assert response.results[0].video_id == str(kept_video.id)


@pytest.mark.asyncio
async def test_ask_question_keeps_explicit_duplicate_video_selection(monkeypatch):
    duplicate_video = Video(
        id=uuid.uuid4(),
        filename="dup.mp4",
        original_filename="dup.mp4",
        file_path="/tmp/dup.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Duplicate",
        transcript={
            "segments": [
                {
                    "start": 12.0,
                    "end": 16.0,
                    "text": "The budget review is due Friday.",
                    "speaker": "Speaker 1",
                },
            ]
        },
        duplicate_info={"suppressed": True},
    )
    captured: dict[str, object] = {}

    async def fake_search_content(query: str, video_ids=None, n_results: int = 0, **_kwargs):
        captured["video_ids"] = video_ids
        return []

    async def fake_complete(messages, max_tokens: int = 0, temperature: float = 0.0):
        return "The budget review is due Friday."

    monkeypatch.setattr(search_module, "search_content", fake_search_content)
    monkeypatch.setattr(search_module, "complete_with_summarization_model", fake_complete)

    db = DummySession([SequenceResult(items=[duplicate_video])])
    response = await ask_question(
        RAGQuestion(question="When is the budget review due?", video_ids=[str(duplicate_video.id)]),
        db=db,
    )

    assert response.sources[0].video_id == str(duplicate_video.id)
    assert captured["video_ids"] == [str(duplicate_video.id)]


@pytest.mark.asyncio
async def test_ask_question_uses_all_requested_video_ids(monkeypatch):
    video_one = Video(
        id=uuid.uuid4(),
        filename="one.mp4",
        original_filename="one.mp4",
        file_path="/tmp/one.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Board Meeting",
        summary={"executive_summary": "Board budget discussion"},
    )
    video_two = Video(
        id=uuid.uuid4(),
        filename="two.mp4",
        original_filename="two.mp4",
        file_path="/tmp/two.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Ops Review",
        summary={"executive_summary": "Ops budget discussion"},
    )

    captured: dict[str, object] = {}

    async def fake_search_content(query: str, video_ids=None, n_results: int = 0, **_kwargs):
        captured["query"] = query
        captured["video_ids"] = video_ids
        captured["n_results"] = n_results
        return [
            {
                "text": "The budget was approved.",
                "metadata": {
                    "video_id": str(video_one.id),
                    "type": "transcript",
                    "timestamp": 12.0,
                    "speaker": "Speaker 1",
                },
                "score": 0.91,
            },
            {
                "text": "Operations confirmed the budget.",
                "metadata": {
                    "video_id": str(video_two.id),
                    "type": "summary",
                    "section": "key_points",
                },
                "score": 0.87,
            },
        ]

    async def fake_complete(messages, max_tokens: int = 0, temperature: float = 0.0):
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        return "The budget was approved and operations confirmed the plan."

    monkeypatch.setattr(search_module, "search_content", fake_search_content)
    monkeypatch.setattr(search_module, "complete_with_summarization_model", fake_complete)

    db = DummySession([SequenceResult(items=[video_one, video_two])])
    response = await ask_question(
        RAGQuestion(
            question="Was the budget approved?",
            video_ids=[str(video_one.id), str(video_two.id)],
        ),
        db=db,
    )

    assert response.answer.startswith("The budget was approved")
    assert len(response.sources) == 2
    assert captured["video_ids"] == [str(video_one.id), str(video_two.id)]
    assert captured["max_tokens"] == 768


@pytest.mark.asyncio
async def test_find_similar_uses_embedding_matches(monkeypatch):
    source_video = Video(
        id=uuid.uuid4(),
        filename="source.mp4",
        original_filename="source.mp4",
        file_path="/tmp/source.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Roadmap",
        summary={"executive_summary": "Roadmap review"},
    )
    similar_video = Video(
        id=uuid.uuid4(),
        filename="similar.mp4",
        original_filename="similar.mp4",
        file_path="/tmp/similar.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Planning",
        summary={"executive_summary": "Planning session overlap"},
    )

    async def fake_find_similar_content(video_id: str, n_results: int = 0):
        assert video_id == str(source_video.id)
        assert n_results >= 3
        return [{"video_id": str(similar_video.id), "score": 0.79}]

    monkeypatch.setattr(search_module, "find_similar_content", fake_find_similar_content)

    db = DummySession([
        SequenceResult(scalar=source_video),
        SequenceResult(items=[similar_video]),
    ])
    response = await find_similar(str(source_video.id), limit=3, db=db)

    assert response.total == 1
    assert response.results[0].video_id == str(similar_video.id)
    assert response.results[0].text == "Planning session overlap"
    assert response.results[0].relevance_score == pytest.approx(0.79 * 0.75)


@pytest.mark.asyncio
async def test_search_falls_back_when_semantic_search_times_out(monkeypatch):
    video = Video(
        id=uuid.uuid4(),
        filename="meeting.mp4",
        original_filename="meeting.mp4",
        file_path="/tmp/meeting.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Meeting",
        transcript={
            "segments": [
                {"start": 12.0, "end": 16.0, "text": "The budget was approved.", "speaker": "Speaker 1"},
            ]
        },
    )

    async def slow_search_content(**_kwargs):
        await asyncio.sleep(0.02)
        return []

    monkeypatch.setattr(search_module, "SEMANTIC_SEARCH_TIMEOUT_S", 0.001)
    monkeypatch.setattr(search_module, "search_content", slow_search_content)

    db = DummySession([SequenceResult(items=[video])])
    response = await search(q="budget", tags=None, speaker=None, limit=5, db=db)

    assert response.total == 1
    assert response.results[0].video_id == str(video.id)
    assert response.results[0].text == "The budget was approved."


@pytest.mark.asyncio
async def test_search_skips_semantic_when_embedding_not_warm_on_sequential_rocm(monkeypatch):
    video = Video(
        id=uuid.uuid4(),
        filename="meeting.mp4",
        original_filename="meeting.mp4",
        file_path="/tmp/meeting.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Roadmap Review",
        transcript={
            "segments": [
                {
                    "start": 18.0,
                    "end": 22.0,
                    "text": "The budget roadmap review is due Friday.",
                    "speaker": "Speaker 1",
                },
            ]
        },
    )

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("semantic search should have been skipped")

    settings = type(
        "Settings",
        (),
        {
            "model_loading": search_module.ModelLoadingStrategy.SEQUENTIAL,
            "gpu_backend": search_module.GPUBackend.ROCM,
        },
    )()

    monkeypatch.setattr(search_module, "get_settings", lambda: settings)
    monkeypatch.setattr(search_module, "get_model_manager", lambda: DummyManager())
    monkeypatch.setattr(search_module, "search_content", should_not_run)

    db = DummySession([SequenceResult(items=[video])])
    response = await search(q="budget roadmap due", tags=None, speaker=None, limit=5, db=db)

    assert response.total == 1
    assert response.results[0].text == "The budget roadmap review is due Friday."
    assert response.results[0].timestamp == pytest.approx(18.0)


@pytest.mark.asyncio
async def test_search_hybrid_ranking_prefers_transcript_over_low_signal_slide(monkeypatch):
    video = Video(
        id=uuid.uuid4(),
        filename="probe.mp4",
        original_filename="probe.mp4",
        file_path="/tmp/probe.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Friday Probe",
        transcript={
            "segments": [
                {
                    "start": 12.0,
                    "end": 16.0,
                    "text": "The budget review is due Friday.",
                    "speaker": "Speaker 1",
                },
            ]
        },
    )

    async def fake_search_content(query: str, video_ids=None, n_results: int = 0, **_kwargs):
        assert query == "Friday"
        return [
            {
                "text": ": ",
                "metadata": {
                    "video_id": str(video.id),
                    "type": "slide",
                    "timestamp": 0.0,
                    "slide_title": "",
                },
                "score": 0.99,
            },
            {
                "text": "The budget review is due Friday.",
                "metadata": {
                    "video_id": str(video.id),
                    "type": "transcript",
                    "timestamp": 12.0,
                    "speaker": "Speaker 1",
                },
                "score": 0.74,
            },
        ]

    monkeypatch.setattr(search_module, "search_content", fake_search_content)

    db = DummySession([SequenceResult(items=[video])])
    response = await search(q="Friday", tags=None, limit=5, db=db)

    assert response.results[0].text == "The budget review is due Friday."
    assert response.results[0].context != "slide"
    assert all(result.text.strip() != ":" for result in response.results)


@pytest.mark.asyncio
async def test_search_collapses_cross_video_duplicates_preferring_newer_video(monkeypatch):
    older_video = Video(
        id=uuid.uuid4(),
        filename="older.mp4",
        original_filename="older.mp4",
        file_path="/tmp/older.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Older Probe",
        transcript={
            "segments": [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "text": "The budget review is due Friday R O C M is enabled on Strix Halo.",
                    "speaker": "Speaker 1",
                },
            ]
        },
        created_at=datetime.now(UTC) - timedelta(days=1),
    )
    newer_video = Video(
        id=uuid.uuid4(),
        filename="newer.mp4",
        original_filename="newer.mp4",
        file_path="/tmp/newer.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Newer Probe",
        transcript={
            "segments": [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "text": "The budget review is due Friday ROCM is enabled on Strix Halo.",
                    "speaker": "Speaker 1",
                },
            ]
        },
        created_at=datetime.now(UTC),
    )

    async def fake_search_content(query: str, video_ids=None, n_results: int = 0, **_kwargs):
        assert query == "Friday"
        return [
            {
                "text": "The budget review is due Friday R O C M is enabled on Strix Halo.",
                "metadata": {
                    "video_id": str(older_video.id),
                    "type": "transcript",
                    "timestamp": 0.0,
                    "speaker": "Speaker 1",
                },
                "score": 0.92,
            },
            {
                "text": "The budget review is due Friday ROCM is enabled on Strix Halo.",
                "metadata": {
                    "video_id": str(newer_video.id),
                    "type": "transcript",
                    "timestamp": 0.0,
                    "speaker": "Speaker 1",
                },
                "score": 0.92,
            },
        ]

    monkeypatch.setattr(search_module, "search_content", fake_search_content)

    db = DummySession([SequenceResult(items=[older_video, newer_video])])
    response = await search(q="Friday", tags=None, limit=5, db=db)

    assert response.total == 1
    assert response.results[0].video_id == str(newer_video.id)


@pytest.mark.asyncio
async def test_ask_question_falls_back_to_lexical_matches_when_semantic_is_unavailable(monkeypatch):
    video = Video(
        id=uuid.uuid4(),
        filename="meeting.mp4",
        original_filename="meeting.mp4",
        file_path="/tmp/meeting.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Roadmap Review",
        transcript={
            "segments": [
                {
                    "start": 12.0,
                    "end": 16.0,
                    "text": "The budget roadmap review is due Friday.",
                    "speaker": "Speaker 1",
                },
            ]
        },
    )
    settings = type(
        "Settings",
        (),
        {
            "model_loading": search_module.ModelLoadingStrategy.SEQUENTIAL,
            "gpu_backend": search_module.GPUBackend.ROCM,
        },
    )()
    captured = {}

    async def fake_complete(messages, max_tokens: int = 0, temperature: float = 0.0):
        captured["messages"] = messages
        return "The budget roadmap review is due Friday."

    monkeypatch.setattr(search_module, "get_settings", lambda: settings)
    monkeypatch.setattr(search_module, "get_model_manager", lambda: DummyManager())
    monkeypatch.setattr(search_module, "complete_with_summarization_model", fake_complete)

    db = DummySession([SequenceResult(items=[video])])
    response = await ask_question(
        RAGQuestion(question="When is the budget roadmap review due?"),
        db=db,
    )

    assert response.answer == "The budget roadmap review is due Friday."
    assert len(response.sources) == 1
    assert response.sources[0].text == "The budget roadmap review is due Friday."
    assert "Friday" in captured["messages"][1]["content"]


@pytest.mark.asyncio
async def test_find_similar_falls_back_to_lexical_similarity(monkeypatch):
    source_video = Video(
        id=uuid.uuid4(),
        filename="source.mp4",
        original_filename="source.mp4",
        file_path="/tmp/source.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Roadmap Review",
        summary={"executive_summary": "Budget roadmap review and planning decisions."},
    )
    similar_video = Video(
        id=uuid.uuid4(),
        filename="similar.mp4",
        original_filename="similar.mp4",
        file_path="/tmp/similar.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Planning Session",
        summary={"executive_summary": "Planning session covering roadmap and budget milestones."},
    )

    async def broken_similarity(*_args, **_kwargs):
        raise RuntimeError("semantic lookup failed")

    monkeypatch.setattr(search_module, "find_similar_content", broken_similarity)

    db = DummySession([
        SequenceResult(scalar=source_video),
        SequenceResult(items=[similar_video]),
        SequenceResult(items=[similar_video]),
    ])
    response = await find_similar(str(source_video.id), limit=3, db=db)

    assert response.total == 1
    assert response.results[0].video_id == str(similar_video.id)
    assert response.results[0].relevance_score > 0


@pytest.mark.asyncio
async def test_find_similar_skips_semantic_when_embedding_not_warm_on_sequential_rocm(monkeypatch):
    source_video = Video(
        id=uuid.uuid4(),
        filename="source.mp4",
        original_filename="source.mp4",
        file_path="/tmp/source.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Roadmap Review",
        summary={"executive_summary": "Budget roadmap review and planning decisions."},
    )
    similar_video = Video(
        id=uuid.uuid4(),
        filename="similar.mp4",
        original_filename="similar.mp4",
        file_path="/tmp/similar.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Planning Session",
        summary={"executive_summary": "Planning session covering roadmap and budget milestones."},
    )
    settings = type(
        "Settings",
        (),
        {
            "model_loading": search_module.ModelLoadingStrategy.SEQUENTIAL,
            "gpu_backend": search_module.GPUBackend.ROCM,
        },
    )()

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("semantic similarity should have been skipped")

    monkeypatch.setattr(search_module, "get_settings", lambda: settings)
    monkeypatch.setattr(search_module, "get_model_manager", lambda: DummyManager())
    monkeypatch.setattr(search_module, "find_similar_content", should_not_run)

    db = DummySession([
        SequenceResult(scalar=source_video),
        SequenceResult(items=[similar_video]),
        SequenceResult(items=[similar_video]),
    ])
    response = await find_similar(str(source_video.id), limit=3, db=db)

    assert response.total == 1
    assert response.results[0].video_id == str(similar_video.id)


@pytest.mark.asyncio
async def test_find_similar_merges_semantic_and_lexical_rankings(monkeypatch):
    source_video = Video(
        id=uuid.uuid4(),
        filename="source.mp4",
        original_filename="source.mp4",
        file_path="/tmp/source.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Budget Review",
        summary={"executive_summary": "Budget review and roadmap planning."},
    )
    semantic_only = Video(
        id=uuid.uuid4(),
        filename="semantic.mp4",
        original_filename="semantic.mp4",
        file_path="/tmp/semantic.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Other Review",
        summary={"executive_summary": "General planning notes and discussion."},
    )
    hybrid_match = Video(
        id=uuid.uuid4(),
        filename="hybrid.mp4",
        original_filename="hybrid.mp4",
        file_path="/tmp/hybrid.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Roadmap Budget Session",
        summary={"executive_summary": "Budget review and roadmap planning milestones."},
    )

    async def fake_find_similar_content(video_id: str, n_results: int = 0):
        assert video_id == str(source_video.id)
        return [
            {"video_id": str(semantic_only.id), "score": 0.88},
            {"video_id": str(hybrid_match.id), "score": 0.70},
        ]

    monkeypatch.setattr(search_module, "find_similar_content", fake_find_similar_content)

    db = DummySession([
        SequenceResult(scalar=source_video),
        SequenceResult(items=[semantic_only, hybrid_match]),
    ])
    response = await find_similar(str(source_video.id), limit=3, db=db)

    assert response.total == 2
    assert response.results[0].video_id == str(hybrid_match.id)


@pytest.mark.asyncio
async def test_find_similar_prefers_transcript_overlap_over_generic_summary(monkeypatch):
    source_video = Video(
        id=uuid.uuid4(),
        filename="source.mp4",
        original_filename="source.mp4",
        file_path="/tmp/source.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Deploy Verification",
        transcript={
            "segments": [
                {
                    "start": 0.0,
                    "end": 6.0,
                    "text": "Please verify deployment and confirm the action items after launch.",
                },
            ]
        },
        summary={
            "executive_summary": "A professional presentation with narration and background music.",
        },
    )
    transcript_match = Video(
        id=uuid.uuid4(),
        filename="good.mp4",
        original_filename="good.mp4",
        file_path="/tmp/good.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Launch Verification",
        transcript={
            "segments": [
                {
                    "start": 0.0,
                    "end": 6.0,
                    "text": "Confirm the deployment verification and review the remaining action items.",
                },
            ]
        },
        summary={
            "executive_summary": "A professional presentation with narration and background music.",
        },
    )
    generic_match = Video(
        id=uuid.uuid4(),
        filename="generic.mp4",
        original_filename="generic.mp4",
        file_path="/tmp/generic.mp4",
        file_size=1,
        mime_type="video/mp4",
        title="Generic Narration",
        summary={
            "executive_summary": "A professional presentation with narration and background music.",
        },
    )

    async def fake_find_similar_content(video_id: str, n_results: int = 0):
        assert video_id == str(source_video.id)
        return []

    monkeypatch.setattr(search_module, "find_similar_content", fake_find_similar_content)

    db = DummySession([
        SequenceResult(scalar=source_video),
        SequenceResult(items=[transcript_match, generic_match]),
    ])
    response = await find_similar(str(source_video.id), limit=3, db=db)

    assert response.total == 1
    assert response.results[0].video_id == str(transcript_match.id)


@pytest.mark.asyncio
async def test_warm_search_runtime_loads_embedding(monkeypatch):
    events: list[tuple[str, object]] = []

    class WarmManager:
        async def get_model(self, model_type):
            events.append(("get_model", model_type))
            return object()

        async def release_model(self, model_type):
            events.append(("release_model", model_type))

    monkeypatch.setattr(search_module, "get_model_manager", lambda: WarmManager())

    response = await warm_search_runtime(SearchWarmRequest(mode="search"))

    assert response.mode == "search"
    assert response.warmed == ["embedding"]
    assert events == [
        ("get_model", search_module.ModelType.EMBEDDING),
        ("release_model", search_module.ModelType.EMBEDDING),
    ]


@pytest.mark.asyncio
async def test_warm_search_runtime_loads_embedding_and_summarization(monkeypatch):
    events: list[tuple[str, object]] = []

    class WarmManager:
        async def get_model(self, model_type):
            events.append(("get_model", model_type))
            return object()

        async def release_model(self, model_type):
            events.append(("release_model", model_type))

    async def fake_warm_summarization():
        events.append(("warm_summarization", None))

    monkeypatch.setattr(search_module, "get_model_manager", lambda: WarmManager())
    monkeypatch.setattr(search_module, "_warm_summarization_runtime", fake_warm_summarization)

    response = await warm_search_runtime(SearchWarmRequest(mode="ask"))

    assert response.mode == "ask"
    assert response.warmed == ["embedding", "summarization"]
    assert events == [
        ("get_model", search_module.ModelType.EMBEDDING),
        ("release_model", search_module.ModelType.EMBEDDING),
        ("warm_summarization", None),
    ]
