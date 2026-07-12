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


def test_collection_slug_sanitizes_and_digests_model_names():
    slug = embeddings._collection_slug("Qwen/Qwen3-Embedding-8B")
    assert slug.startswith("qwen-qwen3-embedding-8b-")
    assert len(slug.rsplit("-", 1)[1]) == 8


def test_long_prefix_models_never_collide():
    base = "org/some-extremely-long-embedding-model-name-variant"
    a = embeddings._collection_name_for_model(base + "-alpha")
    b = embeddings._collection_name_for_model(base + "-beta")
    assert a != b


def test_fresh_deployment_creates_namespaced_collection(monkeypatch, tmp_path):
    client = FakeClient()
    _use(monkeypatch, client, tmp_path)

    collection = embeddings.get_collection()

    assert collection.name == embeddings._collection_name_for_model("Qwen/Qwen3-Embedding-8B")


def test_legacy_collection_is_adopted_and_owner_stamped(monkeypatch, tmp_path):
    client = FakeClient(existing=["video_content"])
    _use(monkeypatch, client, tmp_path)

    assert embeddings.get_collection().name == "video_content"
    assert embeddings._read_legacy_owner_model() == "Qwen/Qwen3-Embedding-8B"
    # Stays adopted for the owning model on later calls.
    assert embeddings.get_collection().name == "video_content"


def test_namespaced_collection_wins_over_legacy(monkeypatch, tmp_path):
    client = FakeClient(
        existing=["video_content", embeddings._collection_name_for_model("nomic-ai/nomic-embed-text-v1.5")]
    )
    _use(monkeypatch, client, tmp_path, model="nomic-ai/nomic-embed-text-v1.5")

    assert embeddings.get_collection().name == (
        embeddings._collection_name_for_model("nomic-ai/nomic-embed-text-v1.5")
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
        embeddings._collection_name_for_model("nomic-ai/nomic-embed-text-v1.5")
    )


def test_delete_video_content_covers_all_content_collections(monkeypatch, tmp_path):
    client = FakeClient(
        existing=[
            "video_content",
            embeddings._collection_name_for_model("nomic-ai/nomic-embed-text-v1.5"),
            "unrelated",
        ]
    )
    _use(monkeypatch, client, tmp_path)

    embeddings.delete_video_content("vid-1")

    assert client.collections["video_content"].deleted_wheres == [{"video_id": "vid-1"}]
    assert client.collections[
        embeddings._collection_name_for_model("nomic-ai/nomic-embed-text-v1.5")
    ].deleted_wheres == [{"video_id": "vid-1"}]
    assert client.collections["unrelated"].deleted_wheres == []


def test_corrupt_or_non_object_marker_fails_closed(monkeypatch, tmp_path):
    client = FakeClient(existing=["video_content"])
    _use(monkeypatch, client, tmp_path)
    marker = tmp_path / "legacy_collection_owner.json"

    for payload in ("[]", "not-json", "{}"):
        marker.write_text(payload)
        # Owner unreadable/foreign-shaped: never adopt implicitly. The claim
        # path sees an existing file it cannot own, so namespaced wins.
        assert embeddings.get_collection().name == (
            embeddings._collection_name_for_model("Qwen/Qwen3-Embedding-8B")
        )


def test_claim_is_exclusive_across_models(monkeypatch, tmp_path):
    client = FakeClient(existing=["video_content"])
    _use(monkeypatch, client, tmp_path)
    assert embeddings._claim_legacy_owner_model("model-a") is True
    assert embeddings._claim_legacy_owner_model("model-b") is False
    assert embeddings._claim_legacy_owner_model("model-a") is True


def test_delete_skips_unrelated_prefix_and_survives_one_failure(monkeypatch, tmp_path):
    nomic_name = embeddings._collection_name_for_model("nomic-ai/nomic-embed-text-v1.5")
    client = FakeClient(existing=["video_content", nomic_name, "video_content_backup"])

    class ExplodingCollection(FakeCollection):
        def delete(self, where=None):
            raise RuntimeError("collection unavailable")

    client.collections["video_content"] = ExplodingCollection("video_content")
    _use(monkeypatch, client, tmp_path)

    embeddings.delete_video_content("vid-9")

    # Failure on the first collection must not abort the sweep...
    assert client.collections[nomic_name].deleted_wheres == [{"video_id": "vid-9"}]
    # ...and non-content prefixes are never touched.
    assert client.collections["video_content_backup"].deleted_wheres == []
