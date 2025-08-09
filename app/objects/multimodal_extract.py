import datetime
from typing import Any
from enum import Enum
from pydantic import BaseModel


class ExtractionContentType(Enum):
    TRACK_AUDIO = "track_audio"
    MIX_AUDIO = "mix_audio"
    MIDI = "midi"
    SHARED_MIDI = "shared_midi"


class MultimodalExtractEventModel(BaseModel):
    start_time: datetime.timedelta | float | None = None
    end_time: datetime.timedelta | float | None = None
    type: ExtractionContentType | str | None = None
    content: Any


class MultimodalExtractModel(BaseModel):
    events: list[MultimodalExtractEventModel] | None = None
    start_time: datetime.timedelta | None = None
    end_time: datetime.timedelta | None = None
    duration: datetime.timedelta | None = None
    sequence: int | None = None
    section_name: str | None = None
    extract_lyrics: str | None = None
    extract_lrc: str | None = None
    section_description: str | None = None
    midi_group: str | None = None


class MultimodalExtract(BaseModel):
    extract_data: MultimodalExtractModel

    def __init__(self, /, **data):
        super().__init__(**data)
