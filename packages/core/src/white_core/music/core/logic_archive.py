"""Pydantic models for the Logic song-folder archive (`archive_manifest.yml`, `archive_keep.yml`)."""

from datetime import datetime

from pydantic import BaseModel, Field

from white_core.enums.archive_status import ArchiveStatus


class ArchiveKeepList(BaseModel):
    """Songs or whole threads kept on the primary drive even when not on a side."""

    songs: list[str] = Field(default_factory=list)
    threads: list[str] = Field(default_factory=list)


class ArchiveManifestEntry(BaseModel):
    source_path: str
    dest_path: str
    file_count: int
    total_bytes: int
    moved_at: datetime
    status: ArchiveStatus


class ArchiveManifest(BaseModel):
    """Keyed by song_id (`<thread>__<production_slug>`)."""

    songs: dict[str, ArchiveManifestEntry] = Field(default_factory=dict)
