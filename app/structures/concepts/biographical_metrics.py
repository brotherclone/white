from typing import Literal, Optional

from pydantic import BaseModel


class BiographicalMetrics(BaseModel):
    """Quantum biographical metrics from analysis tools."""

    taped_over_coefficient: float  # 0.0-1.0
    narrative_malleability: float  # 0.0-1.0
    choice_point_density: int  # 0-N (typically 0-10)
    temporal_significance: Literal["low", "medium", "high"]
    identity_collapse_risk: Literal["low", "moderate", "high"]
    influence_complexity: Optional[int] = None
    forgotten_self_potential: Optional[str] = None
