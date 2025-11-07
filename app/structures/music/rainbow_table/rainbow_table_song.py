from pydantic import BaseModel

from app.structures.music.core.key_signature import KeySignature
from app.structures.music.core.time_signature import TimeSignature


class RainbowTableSong(BaseModel):

    bpm: int | None = None
    tempo: TimeSignature | None = None
    key: KeySignature | None = None
    total_running_time: int | None = None  # in milliseconds
    album_id: str | int | None = None
    sequence_on_album: int | None = None
