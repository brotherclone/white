from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from app.structures.concepts.biographical_period import BiographicalPeriod
from app.structures.concepts.biographical_timeline_summary import (
    BiographicalTimelineSummary,
)


class BiographicalTimeline(BaseModel):
    """Complete biographical timeline."""

    periods: List[BiographicalPeriod]

    # Computed properties
    total_span_years: Optional[int] = None
    high_detail_periods: List[BiographicalPeriod] = Field(default_factory=list)
    low_detail_periods: List[BiographicalPeriod] = Field(default_factory=list)
    forgotten_periods: List[BiographicalPeriod] = Field(default_factory=list)

    def get_surrounding_periods(
        self, target: BiographicalPeriod
    ) -> Dict[str, Optional[BiographicalPeriod]]:
        """Get periods immediately before and after target."""
        idx = self.periods.index(target)
        return {
            "preceding": self.periods[idx - 1] if idx > 0 else None,
            "following": self.periods[idx + 1] if idx < len(self.periods) - 1 else None,
        }

    def filter_by_age_range(
        self, min_age: int, max_age: int
    ) -> List[BiographicalPeriod]:
        """Get periods within age range."""
        return [p for p in self.periods if min_age <= p.age_range.start <= max_age]


class BiographicalTimelineLoadResult(BaseModel):
    """Result of loading biographical timeline."""

    full_timeline: BiographicalTimeline
    forgotten_periods: List[BiographicalPeriod]
    high_detail_periods: List[BiographicalPeriod]
    low_detail_periods: List[BiographicalPeriod]
    summary: BiographicalTimelineSummary
