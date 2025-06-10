import datetime
from typing import List, Any

from pydantic import BaseModel


class MultimodalExtractEventModel(BaseModel):
    time: str
    type: str
    content: Any

class MultimodalExtractModel(BaseModel):
    events: List[MultimodalExtractEventModel] | None = None
    start_time:  datetime.timedelta | None = None
    end_time:  datetime.timedelta | None = None
    duration: datetime.timedelta | None = None
    sequence: int | None = None
    section_name: str | None = None
    audio_segments: List[str] | None = None # Todo: Fix type
    midi_segments: List[str] | None = None   # Todo: Fix type
    lyric_segment: list[dict[str, datetime.timedelta | str | Any]] | None = None

class MultimodalExtract(BaseModel):
    extract_data: MultimodalExtractModel # Why not just put the model here?
    def __init__(self, /, **data):
        super().__init__(**data)

