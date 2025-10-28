from pydantic import BaseModel

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