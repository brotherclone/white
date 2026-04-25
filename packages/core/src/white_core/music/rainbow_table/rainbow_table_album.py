from datetime import datetime

from pydantic import BaseModel


class RainbowTableAlbum(BaseModel):

    title: str
    release_date: datetime
    ef_id: str | int | None = None
    rainbow_mnemonic_character_value: str | None = None

    def __init__(self, **data):
        super().__init__(**data)
