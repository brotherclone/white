from typing import Literal, Union

from pydantic import BaseModel


class PopulationData(BaseModel):
    """Population trajectory over time"""

    year: int
    population: Union[
        int, str
    ]  # Can be numeric or descriptive (e.g., "abundant throughout Caribbean")
    source: str = "estimated"
    confidence: Literal[
        "high",
        "medium",
        "low",
        "medium-high",
        "medium-low",
        "definitive",
        "estimated",
        "uncertain",
    ] = "medium"
