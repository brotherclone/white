import datetime

from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field, computed_field

from app.structures.concepts.biographical_event import BiographicalEvent
from app.structures.enums.biographical_timeline_detail_level import (
    BiographicalTimelineDetailLevel,
)


class BiographicalPeriod(BaseModel):
    """A span of time in the biographical timeline."""

    model_config = {"arbitrary_types_allowed": True}

    start_date: datetime.date
    end_date: datetime.date
    age_range: tuple[int, int]

    # Content
    description: str
    known_events: List[BiographicalEvent] = Field(default_factory=list)
    detail_level: BiographicalTimelineDetailLevel = Field(
        default=BiographicalTimelineDetailLevel.MEDIUM
    )

    # Context
    location: Optional[str] = None
    primary_activity: Optional[str] = None  # "college", "working at X", "traveling"
    key_relationships: List[str] = Field(default_factory=list)
    creative_output: List[str] = Field(default_factory=list)

    # Metadata
    emotional_tone: Optional[str] = None
    trauma_level: Optional[Literal["none", "low", "medium", "high"]] = None

    @computed_field
    @property
    def duration_months(self) -> int:
        """Calculate duration in months."""
        return (self.end_date.year - self.start_date.year) * 12 + (
            self.end_date.month - self.start_date.month
        )

    @computed_field
    @property
    def is_forgotten(self) -> bool:
        """Is this period suitable for taping over?"""
        return (
            self.detail_level
            in [
                BiographicalTimelineDetailLevel.LOW,
                BiographicalTimelineDetailLevel.MINIMAL,
            ]
            and self.trauma_level in ["none", "low", None]
            and self.duration_months >= 6
        )


class PeriodSelectionResult(BaseModel):
    """Result of selecting a forgotten period."""

    period: BiographicalPeriod
    duration_months: int
    actual_known_details: List[BiographicalEvent]
    adjacent_context: Dict[str, Optional[BiographicalPeriod]]
    selection_reason: str
