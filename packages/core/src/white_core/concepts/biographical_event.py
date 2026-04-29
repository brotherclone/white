import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class BiographicalEvent(BaseModel):
    """A single event in the timeline."""

    date: Optional[datetime.date] = Field(default=None, examples=["1998-06-15"])
    year: Optional[int] = Field(default=None, examples=[1998])
    approximate_date: Optional[str] = Field(
        default=None, examples=["Summer 1998", "Late 1993"]
    )
    description: str = Field(
        examples=["Moved to Portland", "Finished recording the album"]
    )
    category: Optional[str] = Field(
        default=None, examples=["career", "relationship", "creative", "location"]
    )
    emotional_weight: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, examples=[0.8]
    )
    tags: List[str] = Field(
        default_factory=list, examples=[["music", "Portland", "1998"]]
    )
