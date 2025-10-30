from pydantic import BaseModel, field_validator

from app.structures.enums.player import RainbowPlayer


class ManifestTrack(BaseModel):

    id: int
    description: str
    audio_file: str | None = None
    midi_file: str | None = None
    group: str | None = None
    midi_group_file: str | None = None
    player: RainbowPlayer = RainbowPlayer.GABE

    def __init__(self, **data):
        super().__init__(**data)

    @field_validator("player", mode="before")
    def accept_name_or_value(cls,v):
        if v is None:
            return None
        if isinstance(v, RainbowPlayer):
            return v
        if isinstance(v, str):
            try:
                return RainbowPlayer[v]
            except KeyError:
                for member in RainbowPlayer:
                    if member.value == v:
                        return member
        raise ValueError(f"Unknown player: {v}")