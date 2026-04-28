from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class AlternateLifeDetail(BaseModel):
    """Specific concrete detail from alternate history."""

    category: Literal[
        "career", "relationship", "location", "creative", "daily_routine", "outcome"
    ] = Field(description="Category of the alternate life detail")
    detail: str = Field(description="Specific concrete detail from alternate history")
    sensory_elements: Optional[List[str]] = Field(
        default=None, description="Sights, sounds, textures associated with this detail"
    )
