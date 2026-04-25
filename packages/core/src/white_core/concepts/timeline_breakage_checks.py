from pydantic import BaseModel


class TimelineBreakageChecks(BaseModel):
    """Individual checks for breakage evaluation."""

    sufficient_malleability: bool = False  # taped_over_coefficient >= 0.4
    narrative_flexibility: bool = False  # narrative_malleability >= 0.4
    choice_point_range: bool = False  # 2 <= choice_point_density <= 8
    low_temporal_significance: bool = (
        False  # temporal_significance in ['low', 'medium']
    )
    safe_identity_risk: bool = False  # identity_collapse_risk in ['low', 'moderate']
    has_some_context: bool = False  # choice_point_density >= 2
