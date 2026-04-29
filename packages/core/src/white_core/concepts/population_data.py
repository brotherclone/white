from typing import Literal, Union

from pydantic import BaseModel, Field


class PopulationData(BaseModel):
    """Population trajectory over time"""

    year: int = Field(
        description="Year of the population observation",
        examples=[1900, 1975, 2023],
    )
    population: Union[int, str] = Field(
        description="Population count or descriptive estimate",
        examples=[5000, "abundant throughout Caribbean"],
    )
    source: str = Field(
        default="estimated",
        description="Source or method of the population estimate",
        examples=["IUCN Red List", "field survey", "estimated"],
    )
    confidence: Literal[
        "high",
        "medium",
        "low",
        "medium-high",
        "medium-low",
        "definitive",
        "estimated",
        "uncertain",
    ] = Field(
        default="medium",
        description="Confidence level of the population data",
        examples=["high", "medium", "low"],
    )
