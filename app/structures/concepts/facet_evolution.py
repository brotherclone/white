from typing import Any, Dict, List
from pydantic import BaseModel, Field, ConfigDict

from app.structures.enums.white_facet import WhiteFacet


class FacetEvolution(BaseModel):
    """
    Tracks how the White Facet (cognitive lens) evolves through the spectrum.

    Like white light refracting through successive prisms - each agent's methodology
    shifts the angle of perception, creating interference patterns.
    """

    initial_facet: WhiteFacet
    initial_metadata: Dict[str, Any] = Field(default_factory=dict)
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_refraction_angle: str = "initial"  # Metaphorical angle of perception

    model_config = ConfigDict(arbitrary_types_allowed=True)
