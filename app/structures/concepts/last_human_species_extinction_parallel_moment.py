from pydantic import Field, BaseModel


class LastHumanSpeciesExtinctionParallelMoment(BaseModel):
    """A moment where species and human narratives intersect"""

    species_moment: str = Field(..., description="Ecological data point")
    human_moment: str = Field(..., description="Personal experience point")
    thematic_connection: str = Field(..., description="How they mirror each other")
    timestamp_relative: str = Field(
        ..., description="e.g., 'Three months before extinction'"
    )
