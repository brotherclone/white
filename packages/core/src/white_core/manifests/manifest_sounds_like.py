from pydantic import BaseModel


class ManifestSoundsLike(BaseModel):

    discogs_id: int
    name: str

    def __init__(self, **data):
        super().__init__(**data)
