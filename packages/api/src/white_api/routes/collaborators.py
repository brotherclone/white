from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from white_production.collaborator_registry import (
    delete_collaborator,
    list_collaborators,
    load_collaborator,
    save_collaborator,
)
from white_production.work_order_store import list_work_orders

from white_core.enums.work_order_status import WorkOrderStatus
from white_core.music.core.collaborator import Collaborator


def make_collaborators_router(
    get_shrink_wrapped_dir,
    *,
    registry_dir: Path | None = None,
) -> APIRouter:
    """Return an APIRouter for collaborator CRUD.

    get_shrink_wrapped_dir: callable () -> Path | None — for checking active work orders.
    registry_dir: override registry location (tests only).
    """
    router = APIRouter(prefix="/collaborators", tags=["collaborators"])

    def _active_work_order_slugs(collaborator_id: str) -> list[str]:
        swdir = get_shrink_wrapped_dir()
        if swdir is None:
            return []
        slugs = []
        for prod_dir in swdir.glob("*/production/*"):
            if not prod_dir.is_dir():
                continue
            for wo in list_work_orders(prod_dir):
                if wo.collaborator_id == collaborator_id and wo.status not in (
                    WorkOrderStatus.ACCEPTED,
                ):
                    slugs.append(wo.song_slug)
        return slugs

    @router.get("")
    def list_all() -> list[dict]:
        return [c.model_dump(mode="json") for c in list_collaborators(registry_dir)]

    @router.get("/{collaborator_id}")
    def get_one(collaborator_id: str) -> dict:
        try:
            return load_collaborator(collaborator_id, registry_dir).model_dump(
                mode="json"
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Collaborator '{collaborator_id}' not found"
            )

    @router.post("", status_code=201)
    def create(body: Collaborator) -> dict:
        try:
            load_collaborator(body.id, registry_dir)
            raise HTTPException(
                status_code=409, detail=f"Collaborator '{body.id}' already exists"
            )
        except FileNotFoundError:
            pass
        save_collaborator(body, registry_dir)
        return body.model_dump(mode="json")

    @router.put("/{collaborator_id}")
    def update(collaborator_id: str, body: Collaborator) -> dict:
        if body.id != collaborator_id:
            raise HTTPException(
                status_code=422,
                detail="Body id must match URL collaborator_id",
            )
        save_collaborator(body, registry_dir)
        return body.model_dump(mode="json")

    @router.delete("/{collaborator_id}", status_code=204)
    def delete(collaborator_id: str) -> None:
        active_slugs = _active_work_order_slugs(collaborator_id)
        try:
            delete_collaborator(
                collaborator_id, registry_dir, active_song_slugs=active_slugs
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Collaborator '{collaborator_id}' not found"
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    return router
