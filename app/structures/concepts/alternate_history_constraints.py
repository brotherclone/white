from pydantic import BaseModel


class AlternateHistoryConstraints(BaseModel):
    """Constraints for generating plausible alternate histories."""

    must_fit_temporally: bool = True
    must_fit_geographically: bool = True
    require_concrete_details: bool = True
    minimum_specificity_score: float = 0.7
    minimum_plausibility_score: float = 0.6

    # Content guidelines
    avoid_fantasy_elements: bool = True
    avoid_wish_fulfillment: bool = True
    require_names_and_places: bool = True
