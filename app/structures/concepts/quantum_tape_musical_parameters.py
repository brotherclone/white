from typing import List, Optional

from pydantic import BaseModel, Field

from app.structures.concepts.quantum_tape_instrumentation import (
    QuantumTapeInstrumentationConfig,
)
from app.structures.concepts.quantum_tape_production_aesthetic import (
    QuantumTapeProductionAesthetic,
)


class QuantumTapeMusicalParameters(BaseModel):
    """Complete musical production specification."""

    # Core parameters
    bpm: int = Field(ge=60, le=180)
    key: str  # "G_major", "D_minor", etc.
    time_signature: str = Field(default="4/4")

    # Genre and style
    primary_genre: str = "folk_rock"
    genre_influences: List[str] = Field(default_factory=list)
    mood: str

    # Instrumentation
    instrumentation: QuantumTapeInstrumentationConfig

    # Production
    production_aesthetic: QuantumTapeProductionAesthetic

    # Lyrical themes
    lyrical_themes: List[str] = Field(default_factory=list)
    narrative_style: Optional[str] = None  # "springsteen_specific", "dylan_cryptic"

    # References
    reference_artists: List[str] = Field(default_factory=list)
    reference_songs: List[str] = Field(default_factory=list)
