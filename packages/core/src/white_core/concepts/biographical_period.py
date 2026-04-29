import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, computed_field

from white_core.concepts.biographical_event import BiographicalEvent
from white_core.enums.biographical_timeline_detail_level import (
    BiographicalTimelineDetailLevel,
)


class BiographicalPeriod(BaseModel):
    """A span of time in the biographical timeline."""

    model_config = {"arbitrary_types_allowed": True}

    start_date: datetime.date = Field(
        description="Start date of the period", examples=["1997-09-01"]
    )
    end_date: datetime.date = Field(
        description="End date of the period", examples=["1999-06-30"]
    )
    age_range: tuple[int, int] = Field(
        description="Subject's age at start and end of period", examples=[(21, 23)]
    )
    description: str = Field(
        description="Narrative description of the period",
        examples=["College years in Portland"],
    )
    known_events: List[BiographicalEvent] = Field(
        default_factory=list, description="Documented events within this period"
    )
    detail_level: BiographicalTimelineDetailLevel = Field(
        default=BiographicalTimelineDetailLevel.MEDIUM,
        description="How much is known about this period",
    )
    location: Optional[str] = Field(
        default=None,
        description="Primary geographic location",
        examples=["Portland, OR"],
    )
    primary_activity: Optional[str] = Field(
        default=None,
        description="Main occupation or activity",
        examples=["college", "working at X", "traveling"],
    )
    key_relationships: List[str] = Field(
        default_factory=list, description="Significant people during this period"
    )
    creative_output: List[str] = Field(
        default_factory=list, description="Creative works produced during this period"
    )
    emotional_tone: Optional[str] = Field(
        default=None,
        description="Overall emotional character of the period",
        examples=["restless", "melancholic", "hopeful"],
    )
    trauma_level: Optional[Literal["none", "low", "medium", "high"]] = Field(
        default=None,
        description="Degree of traumatic content in this period",
        examples=["low"],
    )

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

    period: BiographicalPeriod = Field(description="The selected biographical period")
    duration_months: int = Field(
        description="Duration of the period in months", examples=[18]
    )
    actual_known_details: List[BiographicalEvent] = Field(
        description="Known events within the selected period"
    )
    adjacent_context: Dict[str, Optional[BiographicalPeriod]] = Field(
        description="Periods immediately before and after the selection"
    )
    selection_reason: str = Field(
        description="Why this period was chosen for taping over",
        examples=["Low detail, minimal trauma, sufficient duration"],
    )
