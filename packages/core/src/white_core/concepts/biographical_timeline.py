from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from white_core.concepts.biographical_period import BiographicalPeriod
from white_core.concepts.biographical_timeline_summary import (
    BiographicalTimelineSummary,
)


class BiographicalTimeline(BaseModel):
    """Complete biographical timeline."""

    periods: List[BiographicalPeriod] = Field(
        description="All periods in chronological order"
    )
    total_span_years: Optional[int] = Field(
        default=None, description="Total years covered by the timeline"
    )
    high_detail_periods: List[BiographicalPeriod] = Field(
        default_factory=list, description="Periods with high detail level"
    )
    low_detail_periods: List[BiographicalPeriod] = Field(
        default_factory=list, description="Periods with low detail level"
    )
    forgotten_periods: List[BiographicalPeriod] = Field(
        default_factory=list, description="Periods that are suitable for taping over"
    )

    def get_surrounding_periods(
        self, target: BiographicalPeriod
    ) -> Dict[str, Optional[BiographicalPeriod]]:
        """Get periods immediately before and after the target."""
        idx = self.periods.index(target)
        return {
            "preceding": self.periods[idx - 1] if idx > 0 else None,
            "following": self.periods[idx + 1] if idx < len(self.periods) - 1 else None,
        }

    def filter_by_age_range(
        self, min_age: int, max_age: int
    ) -> List[BiographicalPeriod]:
        """Get periods within age range."""
        return [p for p in self.periods if min_age <= p.age_range[0] <= max_age]


class BiographicalTimelineLoadResult(BaseModel):
    """Result of loading biographical timeline."""

    full_timeline: BiographicalTimeline = Field(
        description="The complete biographical timeline"
    )
    forgotten_periods: List[BiographicalPeriod] = Field(
        description="Periods suitable for taping over"
    )
    high_detail_periods: List[BiographicalPeriod] = Field(
        description="Periods with high detail level"
    )
    low_detail_periods: List[BiographicalPeriod] = Field(
        description="Periods with low detail level"
    )
    summary: BiographicalTimelineSummary = Field(
        description="Summary statistics for the timeline"
    )
