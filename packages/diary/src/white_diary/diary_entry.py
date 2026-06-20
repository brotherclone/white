from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DiaryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    song_slug: str
    phase: str | None = None
    author: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str | None = None
    body: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
