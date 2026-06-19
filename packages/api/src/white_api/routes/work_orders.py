from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from white_production.collaborator_registry import load_collaborator
from white_production.work_order_generator import generate_work_order
from white_production.work_order_store import (
    list_work_orders,
    load_work_order,
    save_work_order,
)

from white_core.enums.collaborator_platform import CollaboratorPlatform
from white_core.enums.collaborator_role import CollaboratorRole
from white_core.music.core.work_order import WorkOrder

log = logging.getLogger(__name__)


class GenerateBody(BaseModel):
    collaborator_id: str
    role: str
    platform: str = "direct"


def make_work_orders_router(
    require_production_dir: Callable[[], Path],
    get_shrink_wrapped_dir: Callable[[], Path | None],
    *,
    registry_dir: Path | None = None,
) -> APIRouter:
    """Return an APIRouter for work order CRUD + generate + draft-email.

    require_production_dir: callable () -> Path that raises HTTPException(503) when none active.
    get_shrink_wrapped_dir: callable () -> Path | None.
    registry_dir: override for tests.
    """
    router = APIRouter(prefix="/production/work-orders", tags=["work-orders"])

    @router.get("")
    def list_all() -> list[dict]:
        prod = require_production_dir()
        return [wo.model_dump(mode="json") for wo in list_work_orders(prod)]

    @router.get("/{collaborator_id}")
    def get_one(collaborator_id: str) -> dict:
        prod = require_production_dir()
        try:
            return load_work_order(prod, collaborator_id).model_dump(mode="json")
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Work order for '{collaborator_id}' not found"
            )

    @router.post("/generate")
    def generate(body: GenerateBody) -> dict:
        prod = require_production_dir()
        try:
            role = CollaboratorRole(body.role)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Unknown role: {body.role!r}")
        try:
            platform = CollaboratorPlatform(body.platform)
        except ValueError:
            raise HTTPException(
                status_code=422, detail=f"Unknown platform: {body.platform!r}"
            )
        wo = generate_work_order(prod, body.collaborator_id, role, platform)
        return wo.model_dump(mode="json")

    @router.put("/{collaborator_id}")
    def update(collaborator_id: str, body: WorkOrder) -> dict:
        if body.collaborator_id != collaborator_id:
            raise HTTPException(
                status_code=422,
                detail="Body collaborator_id must match URL collaborator_id",
            )
        prod = require_production_dir()
        save_work_order(prod, body)
        return body.model_dump(mode="json")

    @router.post("/{collaborator_id}/draft-email")
    async def draft_email(collaborator_id: str) -> dict:
        prod = require_production_dir()
        try:
            wo = load_work_order(prod, collaborator_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Work order for '{collaborator_id}' not found"
            )
        try:
            collaborator = load_collaborator(collaborator_id, registry_dir)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Collaborator '{collaborator_id}' not found"
            )

        if not collaborator.email:
            raise HTTPException(
                status_code=422, detail="Collaborator has no email address"
            )

        body_text = _render_work_order_email(wo, collaborator.name)
        subject = f"Work Order — {wo.song_slug} ({wo.role.value})"

        try:

            log.info("Gmail MCP: creating draft for %s", collaborator.email)
        except Exception:
            pass

        return {
            "to": collaborator.email,
            "subject": subject,
            "body": body_text,
            "status": "ready_to_draft",
            "note": "Call Gmail MCP create_draft with to/subject/body from this response",
        }

    return router


def _render_work_order_email(wo: WorkOrder, collaborator_name: str) -> str:
    lines = [
        f"Hi {collaborator_name},",
        "",
        f"Here are the details for the {wo.role.value} work order on '{wo.song_slug}'.",
        "",
        "--- SONG DETAILS ---",
        f"Key: {wo.key or 'TBD'}",
        f"BPM: {wo.bpm or 'TBD'}",
        f"Time signature: {wo.time_signature}",
    ]
    if wo.sections:
        lines += ["", "Sections:"] + [f"  • {s}" for s in wo.sections]
    if wo.creative_direction:
        lines += ["", "Creative direction:", wo.creative_direction]
    if wo.part_notes:
        lines += ["", "Part notes:", wo.part_notes]
    if wo.deliverable_format:
        lines += ["", f"Deliverable format: {wo.deliverable_format}"]
    if wo.deadline:
        lines += [f"Deadline: {wo.deadline.isoformat()}"]
    if wo.budget_agreed is not None:
        lines += [f"Agreed budget: {wo.budget_agreed} {wo.budget_currency}"]
    lines += ["", "Looking forward to working with you!", ""]
    return "\n".join(lines)
