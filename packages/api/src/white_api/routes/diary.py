from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException
from white_diary import DiaryEntry, delete_entry, list_entries, load_entry, write_entry


def make_diary_router(
    get_shrink_wrapped_dir: Callable[[], Path | None],
) -> APIRouter:
    """Return an APIRouter for diary entry CRUD.

    get_shrink_wrapped_dir: callable () -> Path | None — root of all thread/production dirs.
    """
    router = APIRouter(prefix="/diary", tags=["diary"])

    def _production_dir(song_slug: str) -> Path:
        swdir = get_shrink_wrapped_dir()
        if swdir is not None:
            for candidate in swdir.glob(f"*/production/{song_slug}"):
                if candidate.is_dir():
                    return candidate
        raise HTTPException(
            status_code=404,
            detail=f"Production directory not found for song '{song_slug}'",
        )

    @router.get("/{song_slug}")
    def list_song_entries(song_slug: str) -> list[dict]:
        prod_dir = _production_dir(song_slug)
        return [e.model_dump(mode="json") for e in list_entries(prod_dir)]

    @router.get("/{song_slug}/{entry_id}")
    def get_entry(song_slug: str, entry_id: str) -> dict:
        prod_dir = _production_dir(song_slug)
        try:
            return load_entry(entry_id, prod_dir).model_dump(mode="json")
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Diary entry '{entry_id}' not found"
            )

    @router.post("/{song_slug}", status_code=201)
    def create_entry(song_slug: str, body: DiaryEntry) -> dict:
        prod_dir = _production_dir(song_slug)
        body.song_slug = song_slug
        write_entry(body, prod_dir)
        return body.model_dump(mode="json")

    @router.put("/{song_slug}/{entry_id}")
    def update_entry(song_slug: str, entry_id: str, body: DiaryEntry) -> dict:
        prod_dir = _production_dir(song_slug)
        try:
            load_entry(entry_id, prod_dir)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Diary entry '{entry_id}' not found"
            )
        body.id = entry_id
        body.song_slug = song_slug
        write_entry(body, prod_dir)
        return body.model_dump(mode="json")

    @router.delete("/{song_slug}/{entry_id}", status_code=204)
    def remove_entry(song_slug: str, entry_id: str) -> None:
        prod_dir = _production_dir(song_slug)
        try:
            delete_entry(entry_id, prod_dir)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Diary entry '{entry_id}' not found"
            )

    return router
