from typing import List

from pydantic import BaseModel, Field

from app.structures.concepts.rainbow_table_color import RainbowTableColor
from app.structures.music.core.key_signature import KeySignature
from app.structures.music.core.time_signature import TimeSignature

class SongProposalIteration(BaseModel):

    iteration_id: str = Field(..., description="Unique identifier for the iteration")
    bpm: int = Field(..., description="Beats per minute for the song")
    tempo: str | TimeSignature = Field(..., description="Tempo or time signature of the song (e.g., 4/4, 3/4)")
    key: str | KeySignature = Field(..., description="Key signature of the song (e.g., C Major, A Minor)")
    rainbow_color: str | RainbowTableColor = Field(..., description="Associated rainbow table color for the iteration")
    title: str = Field(..., description="Title of the song")
    mood: list[str] = Field(..., description="List of moods for the song (e.g., happy, sad, energetic)")
    genres: list[str] = Field(..., description="List of genres for the song (e.g., rock, jazz, electronic)")
    concept: str = Field(..., description="Most important part! This concept or theme of the song")

    def __init__(self, **data):
        super().__init__(**data)


class SongProposal(BaseModel):
    iterations: List[SongProposalIteration] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)