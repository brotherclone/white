import uuid

from pydantic import BaseModel


class RainbowSoundsLike(BaseModel):

    artist_name_a: str | None = None
    artist_a_local_id: uuid.UUID | None = None
    artist_a_discogs_id: str | None = None
    artist_name_b: str | None = None
    artist_b_local_id: uuid.UUID | None = None
    artist_b_discogs_id: str | None = None
    descriptor_a: str | None = None
    descriptor_b: str | None = None
    location: str | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_random_arist_name(self):
        pass
