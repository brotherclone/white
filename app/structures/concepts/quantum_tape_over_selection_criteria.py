from typing import List

from pydantic import BaseModel, Field

from app.structures.enums.biographical_timeline_detail_level import (
    BiographicalTimelineDetailLevel,
)


class QuantumTapeOverSelectionCriteria(BaseModel):
    """Criteria for selecting forgotten period to tape over."""

    min_duration_months: int = 6
    preferred_age_range: range = range(20, 45)
    allowed_detail_levels: List[BiographicalTimelineDetailLevel] = Field(
        default_factory=lambda: [
            BiographicalTimelineDetailLevel.LOW,
            BiographicalTimelineDetailLevel.MINIMAL,
        ]
    )
    max_trauma_level: str = "low"
    require_adjacent_context: bool = True
