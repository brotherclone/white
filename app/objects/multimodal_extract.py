from typing import List

from pydantic import BaseModel

from app.objects.rainbow_song_meta import RainbowSongMeta


class MultimodalExtractEventModel(BaseModel):
    time: str
    type: str
    content: str


class MultimodalExtractModel(BaseModel):
    events: List[MultimodalExtractEventModel]


class MultimodalExtract(BaseModel):
    def __init__(self, /, **data):
        super().__init__(**data)

    def extract(self) -> MultimodalExtractModel:
        """
        Extracts the multimodal data and returns a structured model.
        """
        return MultimodalExtractModel(events=self.events)