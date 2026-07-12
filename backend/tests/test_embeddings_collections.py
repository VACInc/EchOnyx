import types

from app.core import embeddings


class FakeCollectionInfo:
    def __init__(self, name):
        self.name = name


class FakeCollection:
    def __init__(self, name):
        self.name = name
        self.deleted_wheres = []

    def delete(self, where=None):
        self.deleted_wheres.append(where)


class FakeClient:
    def __init__(self, existing=()):
        self.collections = {name: FakeCollection(name) for name in existing}
        self.created = []

    def list_collections(self):
        return [FakeCollectionInfo(name) for name in self.collections]

    def get_collection(self, name):
        return self.collections[name]

    def get_or_create_collection(self, name, metadata=None):
        self.created.append(name)
        return self.collections.setdefault(name, FakeCollection(name))


def _use(monkeypatch, client, tmp_path, model="Qwen/Qwen3-Embedding-8B"):
    monkeypatch.setattr(embeddings, "get_chroma_client", lambda: client)
    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: types.SimpleNamespace(
            embedding_model=model, chroma_persist_dir=tmp_path
        ),
    )


def test_collection_slug_sanitizes_model_names():
    assert embeddings._collection_slug("Qwen/Qwen3-Embedding-8B") == "qwen-qwen3-embedding-8b"
    assert embeddings._collection_slug("nomic-ai/nomic-embed-text-v1.5") == (
        "nomic-ai-nomic-embed-text-v1-5"
    )


def test_fresh_deployment_creates_namespaced_collection(monkeypatch, tmp_path):
    client = FakeClient()
    _use(monkeypatch, client, tmp_path)

    collection = embeddings.get_collection()

    assert collection.name == "video_content--qwen-qwen3-embedding-8b"


def test_legacy_collection_is_adopted_and_owner_stamped(monkeypatch, tmp_path):
    client = FakeClient(existing=["video_content"])
    _use(monkeypatch, client, tmp_path)

    assert embeddings.get_collection().name == "video_content"
    assert embeddings._read_legacy_owner_model() == "Qwen/Qwen3-Embedding-8B"
    # Stays adopted for the owning model on later calls.
    assert embeddings.get_collection().name == "video_content"


def test_namespaced_collection_wins_over_legacy(monkeypatch, tmp_path):
    client = FakeClient(
        existing=["video_content", "video_content--nomic-ai-nomic-embed-text-v1-5"]
    )
    _use(monkeypatch, client, tmp_path, model="nomic-ai/nomic-embed-text-v1.5")

    assert embeddings.get_collection().name == (
        "video_content--nomic-ai-nomic-embed-text-v1-5"
    )


def test_model_switch_does_not_inherit_owned_legacy_collection(monkeypatch, tmp_path):
    client = FakeClient(existing=["video_content"])
    _use(monkeypatch, client, tmp_path)
    # Original model adopts legacy and stamps ownership.
    assert embeddings.get_collection().name == "video_content"

    _use(monkeypatch, client, tmp_path, model="nomic-ai/nomic-embed-text-v1.5")
    # A different model must get its own collection, never the legacy one
    # holding incompatible-dimension vectors.
    assert embeddings.get_collection().name == (
        "video_content--nomic-ai-nomic-embed-text-v1-5"
    )


def test_delete_video_content_covers_all_content_collections(monkeypatch, tmp_path):
    client = FakeClient(
        existing=[
            "video_content",
            "video_content--nomic-ai-nomic-embed-text-v1-5",
            "unrelated",
        ]
    )
    _use(monkeypatch, client, tmp_path)

    embeddings.delete_video_content("vid-1")

    assert client.collections["video_content"].deleted_wheres == [{"video_id": "vid-1"}]
    assert client.collections[
        "video_content--nomic-ai-nomic-embed-text-v1-5"
    ].deleted_wheres == [{"video_id": "vid-1"}]
    assert client.collections["unrelated"].deleted_wheres == []
