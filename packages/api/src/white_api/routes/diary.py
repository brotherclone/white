from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from white_diary import DiaryEntry, delete_entry, list_entries, load_entry, write_entry


def make_diary_router(entries_dir: Path) -> APIRouter:
    """Return an APIRouter for diary entry CRUD.

    entries_dir: root directory for all diary entries (e.g. white_diary.ENTRIES_DIR).
    Per-song entries live at <entries_dir>/<song_slug>/<entry_id>.yml.
    No production directory or shrink-wrapped dir required.
    """
    router = APIRouter(prefix="/diary", tags=["diary"])

    def _song_dir(song_slug: str) -> Path:
        return entries_dir / song_slug

    @router.get("/{song_slug}")
    def list_song_entries(song_slug: str) -> list[dict]:
        return [e.model_dump(mode="json") for e in list_entries(_song_dir(song_slug))]

    @router.get("/{song_slug}/{entry_id}")
    def get_entry(song_slug: str, entry_id: str) -> dict:
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
        try:
            delete_entry(entry_id, _song_dir(song_slug))
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Diary entry '{entry_id}' not found"
            )

    return router
