from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from white_diary import (
    DiaryEntry,
    count_entries_by_song,
    delete_entry,
    list_entries,
    load_entry,
    write_entry,
)

_SAFE_SLUG = re.compile(r"^[a-zA-Z0-9_\-]+$")
_SAFE_UUID = re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")


def _check_slug(slug: str, label: str) -> None:
    if not _SAFE_SLUG.match(slug):
        raise HTTPException(status_code=400, detail=f"Invalid {label}")


def _check_entry_id(entry_id: str) -> None:
    if not _SAFE_UUID.match(entry_id):
        raise HTTPException(status_code=400, detail="Invalid entry_id")


def make_diary_router(entries_dir: Path) -> APIRouter:
    """Return an APIRouter for diary entry CRUD.

    entries_dir: root directory for all diary entries (e.g. white_diary.ENTRIES_DIR).
    Per-song entries live at <entries_dir>/<song_slug>/<entry_id>.yml.
    No production directory or shrink-wrapped dir required.
    """
    router = APIRouter(prefix="/diary", tags=["diary"])

    def _song_dir(song_slug: str) -> Path:
        _check_slug(song_slug, "song_slug")
        return entries_dir / song_slug

    @router.get("/counts")
    def get_entry_counts() -> dict[str, int]:
        """Entry counts for every song with a diary directory. Registered
        before /{song_slug} so 'counts' isn't swallowed as a song_slug."""
        return count_entries_by_song(entries_dir)

    @router.get("/{song_slug}")
    def list_song_entries(song_slug: str) -> list[dict]:
        return [e.model_dump(mode="json") for e in list_entries(_song_dir(song_slug))]

    @router.get("/{song_slug}/{entry_id}")
    def get_entry(song_slug: str, entry_id: str) -> dict:
        _check_entry_id(entry_id)
        try:
            return load_entry(entry_id, _song_dir(song_slug)).model_dump(mode="json")
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Diary entry '{entry_id}' not found"
            )

    @router.post("/{song_slug}", status_code=201)
    def create_entry(song_slug: str, body: DiaryEntry) -> dict:
        body.song_slug = song_slug
        write_entry(body, _song_dir(song_slug))
        return body.model_dump(mode="json")

    @router.put("/{song_slug}/{entry_id}")
    def update_entry(song_slug: str, entry_id: str, body: DiaryEntry) -> dict:
        _check_entry_id(entry_id)
        song_dir = _song_dir(song_slug)
        try:
            load_entry(entry_id, song_dir)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Diary entry '{entry_id}' not found"
            )
        body.id = entry_id
        body.song_slug = song_slug
        write_entry(body, song_dir)
        return body.model_dump(mode="json")

    @router.delete("/{song_slug}/{entry_id}", status_code=204)
    def remove_entry(song_slug: str, entry_id: str) -> None:
        _check_entry_id(entry_id)
        try:
            delete_entry(entry_id, _song_dir(song_slug))
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Diary entry '{entry_id}' not found"
            )

    return router
