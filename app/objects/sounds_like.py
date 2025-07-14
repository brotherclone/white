import uuid

from pydantic import BaseModel

class RainbowSoundsLikeArtist(BaseModel):
    name: str | None = None
    local_id: uuid.UUID | None = None
    discogs_id: str | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RainbowSoundsLike(BaseModel):

    artist_a: RainbowSoundsLikeArtist | None = None
    artist_b: RainbowSoundsLikeArtist | None = None
    descriptor_a: str | None = None
    descriptor_b: str | None = None
    location: str | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
