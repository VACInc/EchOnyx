import uuid
from datetime import datetime, timezone

import pytest

from app.api.routes import action_items as action_items_module
from app.api.routes.action_items import (
    ActionItemCreate,
    ActionItemUpdate,
    create_action_item,
    delete_action_item,
    list_action_items,
    update_action_item,
)
from app.models.action_item import ActionItem
from app.models.video import Video
from tests.helpers import ensure_timestamp_defaults


class DummyResult:
    def __init__(self, scalar=None, row=None, rows=None):
        self._scalar = scalar
        self._row = row
        self._rows = list(rows or [])

    def scalar_one(self):
        return self._scalar

    def scalar_one_or_none(self):
        return self._scalar

    def one_or_none(self):
        return self._row

    def all(self):
        return list(self._rows)


class DummySession:
    def __init__(self, results=None):
        self._results = list(results or [])
        self.added = []
        self.deleted = []
        self.commits = 0

    async def execute(self, _query):
        return self._results.pop(0)

    def add(self, item):
        if getattr(item, "id", None) is None:
            item.id = uuid.uuid4()
        ensure_timestamp_defaults(item)
        self.added.append(item)

    async def flush(self):
        return None

    async def commit(self):
        self.commits += 1

    async def refresh(self, _item):
        return None

    async def delete(self, item):
        self.deleted.append(item)


def _video() -> Video:
    video = Video(
        id=uuid.uuid4(),
        filename="sample.mp4",
        original_filename="sample.mp4",
        file_path="/tmp/sample.mp4",
        file_size=123,
        mime_type="video/mp4",
        title="Budget Review",
        tags=["Finance", "Review"],
        created_at=datetime.now(timezone.utc),
    )
    ensure_timestamp_defaults(video)
    return video


def _action_item(*, video_id: uuid.UUID, text: str, source: str = "manual", completed: bool = False) -> ActionItem:
    item = ActionItem(
        id=uuid.uuid4(),
        video_id=video_id,
        text=text,
        source=source,
        completed=completed,
    )
    ensure_timestamp_defaults(item)
    return item


@pytest.mark.asyncio
async def test_list_action_items_serializes_labels():
    video = _video()
    item = _action_item(video_id=video.id, text="Confirm the budget review", source="summary")
    db = DummySession([
        DummyResult(scalar=1),
        DummyResult(rows=[(item, video)]),
    ])

    response = await list_action_items(
        tags=None,
        status="all",
        sort="updated_at",
        order="desc",
        page=1,
        page_size=50,
        db=db,
    )

    assert response.total == 1
    assert response.items[0].text == "Confirm the budget review"
    assert response.items[0].labels == ["Finance", "Review"]
    assert response.items[0].video_title == "Budget Review"


@pytest.mark.asyncio
async def test_create_action_item_dedupes_same_video_text(monkeypatch):
    video = _video()
    existing = _action_item(video_id=video.id, text="Confirm the budget review", source="summary")

    async def fake_load_video(_db, _video_id):
        return video

    async def fake_find_existing(_db, *, video_id, normalized_text):
        assert video_id == video.id
        assert normalized_text == "confirm the budget review"
        return existing

    monkeypatch.setattr(action_items_module, "_load_video", fake_load_video)
    monkeypatch.setattr(action_items_module, "_find_existing_action_item", fake_find_existing)

    response = await create_action_item(
        ActionItemCreate(video_id=str(video.id), text="  Confirm   the budget review  ", source="summary"),
        db=DummySession(),
    )

    assert response.id == str(existing.id)
    assert response.text == "Confirm the budget review"


@pytest.mark.asyncio
async def test_create_action_item_adds_new_manual_item(monkeypatch):
    video = _video()

    async def fake_load_video(_db, _video_id):
        return video

    async def fake_find_existing(_db, *, video_id, normalized_text):
        assert video_id == video.id
        assert normalized_text == "follow up with finance"
        return None

    monkeypatch.setattr(action_items_module, "_load_video", fake_load_video)
    monkeypatch.setattr(action_items_module, "_find_existing_action_item", fake_find_existing)

    db = DummySession()
    response = await create_action_item(
        ActionItemCreate(video_id=str(video.id), text="Follow up with finance", source="manual"),
        db=db,
    )

    assert response.text == "Follow up with finance"
    assert response.source == "manual"
    assert db.commits == 1
    assert db.added[0].text == "Follow up with finance"


@pytest.mark.asyncio
async def test_update_action_item_marks_completed():
    video = _video()
    item = _action_item(video_id=video.id, text="Confirm the budget review", source="summary")
    db = DummySession([DummyResult(row=(item, video))])

    response = await update_action_item(str(item.id), ActionItemUpdate(completed=True), db=db)

    assert response.completed is True
    assert item.completed_at is not None
    assert db.commits == 1


@pytest.mark.asyncio
async def test_delete_action_item_removes_item():
    item = _action_item(video_id=uuid.uuid4(), text="Remove me")
    db = DummySession([DummyResult(scalar=item)])

    response = await delete_action_item(str(item.id), db=db)

    assert response["message"] == "Action item deleted successfully"
    assert db.deleted == [item]
    assert db.commits == 1
