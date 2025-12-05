from typing import Literal, Optional, List, Dict

from pydantic import BaseModel, Field


class ExtinctionMusicalParameters(BaseModel):
    """
    New Music / Classical aesthetic parameters for The Empty Fields.
    """

    bpm: int = Field(..., ge=40, le=200)
    key: str = Field(..., description="Musical key, can be atonal or microtonal")
    time_signature: str = Field(default="4/4")

    # ToDo: Refactor to align with Song Proposal

    # Texture
    note_density: str = Field(..., description="sparse → moderate → dense over time")
    silence_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=0.8,
        description="Percentage of piece that is rests/gaps",
    )

    # Harmony
    dissonance_level: Literal["consonant", "moderate", "dissonant", "microtonal"] = (
        "moderate"
    )
    harmonic_progression: Optional[str] = None

    # Instrumentation
    primary_instruments: List[str] = Field(default_factory=list)
    texture_description: str = Field(
        default="sparse, with long silences between phrases"
    )

    # Structure
    structure: str = Field(default="parallel_duet")
    movement_count: int = Field(default=1)

    # Specific to Empty Fields aesthetic
    ecological_sound_source: Optional[str] = Field(
        None, description="Natural sound that could be sampled (whale song, wind, etc)"
    )

    human_sound_source: Optional[str] = Field(
        None, description="Human sound that parallels (breathing, footsteps, etc)"
    )

    def to_artifact_dict(self) -> Dict:
        return self.model_dump()
