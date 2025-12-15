from typing import Optional

from pydantic import BaseModel, Field


class MethodologyFeature(BaseModel):
    concept_length: int = Field(description="Length of concept in characters")
    word_count: int = Field(description="Number of words")
    sentence_count: int = Field(description="Number of sentences")
    avg_word_length: float = Field(description="Average word length")
    question_marks: int = Field(description="Count of ?")
    exclamation_marks: int = Field(description="Count of !")
    uncertainty_level: float = Field(description="Boundary fluidity")
    narrative_complexity: float = Field(description="Temporal complexity")
    discrepancy_intensity: float = Field(description="Memory discrepancy severity")
    has_rebracketing_markers: bool = Field(description="Any core markers present")
    track_duration: Optional[float] = Field(
        default=None, description="Duration in seconds"
    )
    track_position: Optional[int] = Field(
        default=None, description="Position in album (1-indexed)"
    )
    album_sequence: Optional[int] = Field(
        default=None, description="Album number in Rainbow Table (1-9)"
    )
