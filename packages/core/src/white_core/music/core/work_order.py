from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from white_core.enums.budget_status import BudgetStatus
from white_core.enums.chain_artifact_type import ChainArtifactType
from white_core.enums.collaborator_platform import CollaboratorPlatform
from white_core.enums.collaborator_role import CollaboratorRole
from white_core.enums.work_order_status import WorkOrderStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RoyaltySplit(BaseModel):
    collaborator_id: str
    song_slug: str
    mechanical_pct: float = 0.0
    performance_pct: float = 0.0
    sync_pct: float = 0.0
    notes: str = ""


class WorkOrder(BaseModel):
    id: str
    song_slug: str
    collaborator_id: str
    role: CollaboratorRole
    platform: CollaboratorPlatform = CollaboratorPlatform.DIRECT
    status: WorkOrderStatus = WorkOrderStatus.DRAFT

    # Musical context
    key: str = ""
    bpm: Optional[int] = None
    time_signature: str = "4/4"
    sections: list[str] = []
    creative_direction: str = ""
    part_notes: str = ""

    # Artifacts to include in packet
    artifact_types: list[ChainArtifactType] = Field(default_factory=list)
    deliverable_format: str = ""

    # Scheduling
    deadline: Optional[date] = None
    follow_up_date: Optional[date] = None
    follow_up_reason: str = ""

    # Budget
    budget_agreed: Optional[float] = None
    budget_paid: Optional[float] = None
    budget_currency: str = "USD"
    budget_status: BudgetStatus = BudgetStatus.PENDING

    # Calendar integration
    calendar_event_id: Optional[str] = None

    # Royalties
    royalty_split: Optional[RoyaltySplit] = None

    # Metadata
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
