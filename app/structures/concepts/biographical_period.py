import datetime

from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field, field_validator

from app.structures.concepts.biographical_event import BiographicalEvent
from app.structures.enums.biographical_timeline_detail_level import (
    BiographicalTimelineDetailLevel,
)


class BiographicalPeriod(BaseModel):
    """A span of time in the biographical timeline."""

    model_config = {"arbitrary_types_allowed": True}

    start_date: datetime.date
    end_date: datetime.date
    age_range: range = None  # Computed from dates

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

    @property
    def duration_months(self) -> int:
        """Calculate duration in months."""
        return (self.end_date.year - self.start_date.year) * 12 + (
            self.end_date.month - self.start_date.month
        )

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

    @field_validator("age_range", mode="before")
    def parse_age_range(cls, v):
        if v is None or isinstance(v, range):
            return v

        # string like "6..12" or "6-12" (inclusive)
        if isinstance(v, str):
            parts = None
            if ".." in v:
                parts = v.split("..")
            elif "-" in v:
                parts = v.split("-")
            if parts and len(parts) == 2:
                start, end = map(int, map(str.strip, parts))
                return range(start, end + 1)  # treat end as inclusive

        # list/tuple like [6, 12] or [6, 12, 2]
        if isinstance(v, (list, tuple)):
            nums = list(map(int, v))
            if len(nums) == 2:
                return range(nums[0], nums[1] + 1)
            if len(nums) == 3:
                return range(nums[0], nums[1] + 1, nums[2])

        # mapping like {start: 6, stop: 13, step: 1}
        if isinstance(v, dict):
            start = int(v["start"])
            stop = int(v["stop"])
            step = int(v.get("step", 1))
            # assume `stop` here is exclusive; if your YAML uses inclusive stop, add +1
            return range(start, stop, step)

        raise TypeError("age_range must be a range or a parsable representation")


class PeriodSelectionResult(BaseModel):
    """Result of selecting a forgotten period."""

    period: BiographicalPeriod
    duration_months: int
    actual_known_details: List[BiographicalEvent]
    adjacent_context: Dict[str, Optional[BiographicalPeriod]]
    selection_reason: str
