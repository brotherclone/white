import datetime
from typing import Optional

from pydantic import BaseModel, computed_field

from app.structures.concepts.biographical_metrics import BiographicalMetrics
from app.structures.concepts.timeline_breakage_checks import TimelineBreakageChecks


class TimelineEvaluationResult(BaseModel):
    """Result of breakage evaluation for a time period."""

    is_suitable: bool  # Overall verdict
    checks: TimelineBreakageChecks  # Detailed check results
    breakage_score: float  # 0.0-1.0 weighted score
    reason: str  # Human-readable explanation
    metrics: BiographicalMetrics  # Source metrics from tools
    year: Optional[int] = None
    evaluation_timestamp: datetime.datetime

    @computed_field
    @property
    def passed_checks_count(self) -> int:
        """Count how many checks passed."""
        checks_dict = self.checks.model_dump()
        return sum(1 for v in checks_dict.values() if v)

    @computed_field
    @property
    def total_checks_count(self) -> int:
        """Total number of checks."""
        return len(self.checks.model_dump())

    @computed_field
    @property
    def passed_percentage(self) -> float:
        """Percentage of checks that passed."""
        return (
            self.passed_checks_count / self.total_checks_count
            if self.total_checks_count > 0
            else 0.0
        )
