from pydantic import BaseModel
from typing import Literal


class PopulationData(BaseModel):
    """Population trajectory over time"""

    year: int
    population: int
    source: str = "estimated"
    confidence: Literal["high", "medium", "low"] = "medium"
