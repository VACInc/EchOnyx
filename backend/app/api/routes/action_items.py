"""Action item endpoints."""

import re
import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal, Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import Text, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.action_item import ActionItem
from app.models.video import Video

router = APIRouter()

WHITESPACE_RE = re.compile(r"\s+")


class ActionItemResponse(BaseModel):
    """Single action item response."""

    id: str
    video_id: str
    video_title: str
    text: str
    source: str
    completed: bool
    completed_at: str | None
    labels: list[str]
    created_at: str
    updated_at: str


class ActionItemListResponse(BaseModel):
    """Paginated action item list response."""

    items: list[ActionItemResponse]
    total: int
    page: int
    page_size: int


class ActionItemCreate(BaseModel):
    """Create a new action item."""

    video_id: str
    text: str
    source: Literal["manual", "summary"] = "manual"


class ActionItemUpdate(BaseModel):
    """Update an action item."""

    text: str | None = None
    completed: bool | None = None


def _normalize_action_item_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip()).lower()


def _serialize_action_item(item: ActionItem, video: Video) -> ActionItemResponse:
    return ActionItemResponse(
        id=str(item.id),
        video_id=str(video.id),
        video_title=video.title or video.original_filename,
        text=item.text,
        source=item.source,
        completed=item.completed,
        completed_at=item.completed_at.isoformat() if item.completed_at else None,
        labels=video.tags or [],
        created_at=item.created_at.isoformat(),
        updated_at=item.updated_at.isoformat(),
    )


async def _load_video(db: AsyncSession, video_id: str) -> Video:
    try:
        vid = uuid.UUID(video_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid video ID") from exc

    result = await db.execute(select(Video).where(Video.id == vid))
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


async def _find_existing_action_item(db: AsyncSession, *, video_id: uuid.UUID, normalized_text: str) -> ActionItem | None:
    result = await db.execute(select(ActionItem).where(ActionItem.video_id == video_id))
    for item in result.scalars().all():
        if _normalize_action_item_text(item.text) == normalized_text:
            return item
    return None


@router.get("", response_model=ActionItemListResponse)
async def list_action_items(
    video_id: str | None = None,
    tags: Sequence[str] | None = Query(None),
    status: Literal["open", "completed", "all"] = Query("open"),
    search: str | None = None,
    sort: Literal["created_at", "updated_at", "completed_at", "video_title"] = Query("updated_at"),
    order: Literal["asc", "desc"] = Query("desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> ActionItemListResponse:
    query = select(ActionItem, Video).join(Video, ActionItem.video_id == Video.id)
    count_query = select(func.count(ActionItem.id)).select_from(ActionItem).join(Video, ActionItem.video_id == Video.id)

    filters = []
    if video_id:
        try:
            filters.append(Video.id == uuid.UUID(video_id))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid video ID") from exc
    if status == "open":
        filters.append(ActionItem.completed.is_(False))
    elif status == "completed":
        filters.append(ActionItem.completed.is_(True))
    if search:
        filters.append(ActionItem.text.ilike(f"%{search.strip()}%"))
    if tags:
        tag_filters = [func.lower(func.coalesce(Video.tags.cast(Text), "")).ilike(f'%"{tag.strip().lower()}"%') for tag in tags if tag.strip()]
        if tag_filters:
            filters.append(or_(*tag_filters))

    if filters:
        query = query.where(*filters)
        count_query = count_query.where(*filters)

    sort_map = {
        "created_at": ActionItem.created_at,
        "updated_at": ActionItem.updated_at,
        "completed_at": ActionItem.completed_at,
        "video_title": func.coalesce(Video.title, Video.original_filename),
    }
    sort_column = sort_map[sort]
    sort_column = sort_column.asc() if order == "asc" else sort_column.desc()
    query = query.order_by(sort_column, ActionItem.created_at.desc())

    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    result = await db.execute(query)
    rows = result.all()

    return ActionItemListResponse(
        items=[_serialize_action_item(item, video) for item, video in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=ActionItemResponse)
async def create_action_item(
    payload: ActionItemCreate,
    db: AsyncSession = Depends(get_db),
) -> ActionItemResponse:
    video = await _load_video(db, payload.video_id)
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Action item text is required")

    normalized_text = _normalize_action_item_text(text)
    existing = await _find_existing_action_item(db, video_id=video.id, normalized_text=normalized_text)
    if existing:
        return _serialize_action_item(existing, video)

    item = ActionItem(
        video_id=video.id,
        text=text,
        source=payload.source,
        completed=False,
    )
    db.add(item)
    await db.flush()
    await db.commit()
    await db.refresh(item)
    return _serialize_action_item(item, video)


@router.put("/{action_item_id}", response_model=ActionItemResponse)
async def update_action_item(
    action_item_id: str,
    payload: ActionItemUpdate,
    db: AsyncSession = Depends(get_db),
) -> ActionItemResponse:
    try:
        item_id = uuid.UUID(action_item_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid action item ID") from exc

    result = await db.execute(select(ActionItem, Video).join(Video, ActionItem.video_id == Video.id).where(ActionItem.id == item_id))
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Action item not found")
    item, video = row

    if payload.text is not None:
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Action item text is required")
        item.text = text
    if payload.completed is not None:
        item.completed = payload.completed
        item.completed_at = datetime.now(timezone.utc) if payload.completed else None

    await db.commit()
    await db.refresh(item)
    return _serialize_action_item(item, video)


@router.delete("/{action_item_id}")
async def delete_action_item(
    action_item_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    try:
        item_id = uuid.UUID(action_item_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid action item ID") from exc

    result = await db.execute(select(ActionItem).where(ActionItem.id == item_id))
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Action item not found")

    await db.delete(item)
    await db.commit()
    return {"message": "Action item deleted successfully"}
