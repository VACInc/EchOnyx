import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.core import embeddings


def _fake_embedding_for(text: str) -> list[float]:
    lowered = text.lower()
    return [
        float(lowered.count("budget")) + 0.1,
        float(lowered.count("roadmap")) + 0.1,
        float(lowered.count("operations")) + 0.1,
        0.1,
    ]


@pytest.mark.asyncio
async def test_semantic_index_search_and_similarity(monkeypatch, tmp_path):
    monkeypatch.setattr(
        embeddings,
        "_chroma_client",
        chromadb.PersistentClient(
            path=str(tmp_path / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        ),
    )

    async def fake_generate_embeddings(texts: list[str]) -> list[list[float]]:
        return [_fake_embedding_for(text) for text in texts]

    monkeypatch.setattr(embeddings, "generate_embeddings", fake_generate_embeddings)

    video_one = "11111111-1111-1111-1111-111111111111"
    video_two = "22222222-2222-2222-2222-222222222222"

    await embeddings.index_video_content(
        video_id=video_one,
        transcript={
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Budget review and roadmap planning.", "speaker": "Speaker 1"},
            ]
        },
        summary={
            "executive_summary": "Budget review for the roadmap.",
            "key_points": ["Budget approved"],
            "topics": [{"timestamp": "00:00:00", "topic": "Budget", "summary": "Budget review"}],
        },
        slides=[{"timestamp": 0.0, "title": "Budget", "content": "Budget roadmap", "key_points": []}],
    )
    await embeddings.index_video_content(
        video_id=video_two,
        transcript={
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Budget operations update.", "speaker": "Speaker 2"},
            ]
        },
        summary={
            "executive_summary": "Budget operations update.",
            "key_points": ["Budget tracked by operations"],
            "topics": [{"timestamp": "00:00:00", "topic": "Budget", "summary": "Operations budget update"}],
        },
        slides=[{"timestamp": 0.0, "title": "Ops", "content": "Budget operations", "key_points": []}],
    )

    matches = await embeddings.search_content(
        query="budget roadmap",
        video_ids=[video_one, video_two],
        n_results=3,
    )
    similar = await embeddings.find_similar_content(video_one, n_results=3)

    assert matches
    assert matches[0]["metadata"]["video_id"] == video_one
    assert similar[0]["video_id"] == video_two


@pytest.mark.asyncio
async def test_find_similar_rebuilds_query_embedding_from_documents(monkeypatch):
    calls: dict[str, object] = {}

    class FakeCollection:
        def get(self, where=None, include=None):
            calls["include"] = include
            return {"documents": ["Roadmap review and budget planning."]}

        def query(self, query_embeddings=None, n_results: int = 0, include=None):
            calls["query_embeddings"] = query_embeddings
            calls["n_results"] = n_results
            return {
                "ids": [["source_chunk", "other_chunk"]],
                "metadatas": [[
                    {"video_id": "source-video"},
                    {"video_id": "other-video"},
                ]],
                "distances": [[0.0, 0.2]],
            }

    async def fake_generate_embeddings(texts: list[str]) -> list[list[float]]:
        calls["texts"] = texts
        return [[0.4, 0.6]]

    monkeypatch.setattr(embeddings, "get_collection", lambda name="video_content": FakeCollection())
    monkeypatch.setattr(embeddings, "generate_embeddings", fake_generate_embeddings)

    similar = await embeddings.find_similar_content("source-video", n_results=1)

    assert calls["include"] == ["documents"]
    assert calls["texts"] == ["Roadmap review and budget planning."]
    assert calls["query_embeddings"] == [[0.4, 0.6]]
    assert similar == [{"video_id": "other-video", "score": pytest.approx(0.8)}]
