from pydantic import BaseModel

from app.structures.music.core.duration import Duration


class ManifestSongStructure(BaseModel):

    section_name: str
    start_time: str | Duration
    end_time: str | Duration
    description: str | None = None

    def __init__(self, **data):
        super().__init__(**data)

