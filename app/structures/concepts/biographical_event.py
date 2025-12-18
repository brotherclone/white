import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class BiographicalEvent(BaseModel):
    """A single event in the timeline."""

    date: Optional[datetime.date] = None
    year: Optional[int] = None
    approximate_date: Optional[str] = None  # "Summer 1998", "Late 1993"
    description: str
    category: Optional[str] = None  # "career", "relationship", "creative", "location"
    emotional_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
