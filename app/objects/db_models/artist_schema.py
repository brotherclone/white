from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, ForeignKey, Table, Text, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RainbowArtist(BaseModel):
    id: int | None = None
    discogs_id: int | None = None
    name: str
    profile: str | None = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.discogs_id is None:
            self.discogs_id = 0  # Default value if not provided


class ArtistSchema(Base):
    __tablename__ = 'artists'

    id = Column(Integer, primary_key=True)
    discogs_id = Column(Integer, unique=True)
    name = Column(String(255), nullable=False)
    profile = Column(Text)

    def __repr__(self):
        return f"<Artist(id={self.id}, name='{self.name}', discogs_id='{self.discogs_id}')>"