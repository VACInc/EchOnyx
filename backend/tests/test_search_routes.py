import uuid
import asyncio

import pytest

from app.api.routes import search as search_module
from app.api.routes.search import RAGQuestion, ask_question, find_similar, search
from app.models.video import Video
from tests.helpers import SequenceResult


class DummySession:
    def __init__(self, results):
        self._results = list(results)

    async def execute(self, _query):
        return self._results.pop(0)


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

    assert response.total == 1
    assert response.results[0].video_id == str(kept_video.id)
    assert response.results[0].context == "summary section: executive_summary"
    assert captured["video_ids"] == [str(kept_video.id)]


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
        assert n_results == 3
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
    assert response.results[0].relevance_score == pytest.approx(0.79)


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
