from pydantic import BaseModel


class PulsarPalaceRoom(BaseModel):
    """A generated room in the Pulsar Palace"""

    room_id: str
    name: str
    description: str
    atmosphere: str
    room_type: str
    exits: list[str]
    inhabitants: list[str]
    features: list[str]
    narrative_beat: str

    def __init__(self, **data):
        super().__init__(**data)


class PulsarPalaceEncounter(BaseModel):
    """An encounter within a room"""

    encounter_id: str
    room_id: str
    characters_involved: list[str]
    actions: list[str]
    narrative: str
    tension_level: int

    def __init__(self, **data):
        super().__init__(**data)
