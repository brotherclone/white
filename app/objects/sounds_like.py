from app.objects.db_models.artist_schema import RainbowArtist
from pydantic import BaseModel

from app.utils.db_util import get_artist_by_name, db_arist_to_rainbow_artist


class RainbowSoundsLike(BaseModel):
    artist_a: RainbowArtist | None = None
    artist_b: RainbowArtist | None = None
    descriptor_a: str | None = None
    descriptor_b: str | None = None
    location: str | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_statement(self):
        if self.artist_a:
            statement = f"Sounds like {self.artist_a.name}"
            if self.artist_b:
                statement = f"{statement} meets {self.artist_b.name}"
            if self.descriptor_a:
                statement = f"{statement} playing {self.descriptor_a}"
                if self.descriptor_b:
                    statement = f"{statement} and {self.descriptor_b}"
                if self.location:
                    statement = f"{statement} in {self.location}"
            return f"{statement}."
        return None
