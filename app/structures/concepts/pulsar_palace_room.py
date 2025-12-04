from typing import Optional
from pydantic import BaseModel, Field


class PulsarPalaceRoom(BaseModel):
    """A generated room in the Pulsar Palace"""

    room_id: str = Field(description="The ID of the room")
    name: Optional[str] = Field(default=None, description="The name of the room")
    description: Optional[str] = Field(
        default=None, description="The description of the room"
    )
    atmosphere: Optional[str] = Field(
        default=None, description="The atmosphere of the room"
    )
    room_type: Optional[str] = Field(default=None, description="The type of room")
    exits: Optional[list[str]] = Field(
        default=None, description="The exits of the room"
    )
    inhabitants: Optional[list[str]] = Field(
        default=None, description="The inhabitants of the room"
    )
    features: Optional[list[str]] = Field(
        default=None, description="The features of the room"
    )
    narrative_beat: Optional[str] = Field(
        default=None, description="The narrative beat of the room"
    )  # ?

    def __init__(self, **data):
        super().__init__(**data)
