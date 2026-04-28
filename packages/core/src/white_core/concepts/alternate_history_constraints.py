from pydantic import BaseModel, Field


class AlternateHistoryConstraints(BaseModel):
    """Constraints for generating plausible alternate histories."""

    must_fit_temporally: bool = Field(default=True, description="Must fit temporally")
    must_fit_geographically: bool = Field(
        default=True, description="Must fit geographically"
    )
    require_concrete_details: bool = Field(
        default=True, description="Require concrete details"
    )
    minimum_specificity_score: float = Field(
        default=0.7, description="Minimum specificity score"
    )
    minimum_plausibility_score: float = Field(
        default=0.6, description="Minimum plausibility score"
    )
    avoid_fantasy_elements: bool = Field(
        default=True, description="Avoid fantasy elements"
    )
    avoid_wish_fulfillment: bool = Field(
        default=True, description="Avoid wish fulfillment"
    )
    require_names_and_places: bool = Field(
        default=True, description="Require names and places"
    )
