from typing import Literal, Optional, List

from pydantic import BaseModel


class AlternateLifeDetail(BaseModel):
    """Specific concrete detail from alternate history."""

    category: Literal[
        "career", "relationship", "location", "creative", "daily_routine", "outcome"
    ]
    detail: str
    sensory_elements: Optional[List[str]] = None  # Sights, sounds, textures
