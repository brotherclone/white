from abc import ABC
from typing import List, Optional

from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.concepts.alternate_life_detail import AlternateLifeDetail
from app.structures.concepts.biographical_period import BiographicalPeriod
from app.structures.concepts.divergence_point import DivergencePoint
from app.structures.enums.quantum_tape_emotional_tone import QuantumTapeEmotionalTone


class AlternateTimelineArtifact(ChainArtifact, ABC):
    """The fictional-but-plausible alternate life."""

    # Core narrative
    period: BiographicalPeriod
    title: str  # "Summer in Portland, 1998"
    narrative: str  # Full prose description

    # Structure
    divergence_point: DivergencePoint
    key_differences: List[str]  # Bullet points of what changed
    specific_details: List[AlternateLifeDetail]

    # Tone
    emotional_tone: QuantumTapeEmotionalTone
    mood_description: str

    # Context
    preceding_events: List[str]  # What happened before in actual timeline
    following_events: List[str]  # What happened after in actual timeline

    # Quality metrics
    plausibility_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    specificity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    divergence_magnitude: Optional[float] = Field(None, ge=0.0, le=1.0)

    def __init__(self, **data):
        super().__init__(**data)

    def flatten(self):
        pass

    def save_file(self):
        pass
