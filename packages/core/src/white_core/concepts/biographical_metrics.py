from typing import Literal, Optional

from pydantic import BaseModel, Field


class BiographicalMetrics(BaseModel):
    """Quantum biographical metrics from analysis tools."""

    taped_over_coefficient: float = Field(
        ge=0.0,
        le=1.0,
        description="How much of the original self has been recorded over",
        examples=[0.72],
    )
    narrative_malleability: float = Field(
        ge=0.0,
        le=1.0,
        description="How readily the biographical narrative can be reshaped",
        examples=[0.45],
    )
    choice_point_density: int = Field(
        ge=0,
        description="Number of significant fork-in-the-road moments (typically 0–10)",
        examples=[7],
    )
    temporal_significance: Literal["low", "medium", "high"] = Field(
        description="Significance of this period in the overall timeline",
        examples=["high"],
    )
    identity_collapse_risk: Literal["low", "moderate", "high"] = Field(
        description="Risk of identity dissolution under alternate-timeline exposure",
        examples=["moderate"],
    )
    influence_complexity: Optional[int] = Field(
        default=None,
        description="Number of interlocking influences shaping the subject",
        examples=[4],
    )
    forgotten_self_potential: Optional[str] = Field(
        default=None,
        description="Qualitative description of the self that was lost or suppressed",
        examples=["The musician who never left Portland"],
    )
