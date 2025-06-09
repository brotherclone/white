from typing import Any

from pydantic import BaseModel

from app.objects.multimodal_extract import MultimodalExtract
from app.objects.rainbow_song_meta import RainbowSongMeta, RainbowMetaDataModel


class RainbowSongModel(BaseModel):
    meta_data: RainbowMetaDataModel
    extracts: list[MultimodalExtract] | None = None

class RainbowSong(BaseModel):

    meta_data: RainbowMetaDataModel

    def __init__(self, /, **data: Any):
        super().__init__(**data)


    def extract(self):
        for song_section in self.meta_data.structure:
            print(song_section)
        pass
