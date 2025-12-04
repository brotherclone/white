from typing import Optional

from pydantic import BaseModel, Field


class PulsarPalaceEncounter(BaseModel):
    """An encounter within a room"""

    encounter_id: str = Field(description="The ID of the encounter")
    room_id: Optional[str] = Field(default=None, description="The ID of the room")
    characters_involved: Optional[list[str]] = Field(
        default=None, description="The IDs of the characters involved"
    )  # Characters don't have IDs yet!
    actions: Optional[list[str]] = Field(
        default=None, description="The actions taken in the encounter"
    )
    narrative: Optional[str] = Field(
        default=None, description="The narrative of the encounter"
    )
    tension_level: Optional[int] = Field(
        default=None, description="The tension level of the encounter"
    )  # Don't know scale

    def __init__(self, **data):
        super().__init__(**data)
