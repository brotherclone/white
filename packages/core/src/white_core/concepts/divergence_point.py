from pydantic import BaseModel, Field


class DivergencePoint(BaseModel):
    """Where the alternate timeline splits from actual history."""

    when: str = Field(
        description="When the divergence occurs in the timeline",
        examples=["After graduating college in 1997"],
    )
    what_changed: str = Field(
        description="The specific decision or event that differed",
        examples=["Took the Greyhound to Portland instead of returning to NJ"],
    )
    why_plausible: str = Field(
        description="Why this divergence is believable given the circumstances",
        examples=["Had been offered a job at Powell's Books"],
    )
