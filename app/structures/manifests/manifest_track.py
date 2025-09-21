from pydantic import BaseModel


class ManifestTrack(BaseModel):

    id: int
    description: str
    audio_file: str | None = None
    midi_file: str | None = None
    group: str | None = None
    midi_group_file: str | None = None

    def __init__(self, **data):
        super().__init__(**data)