from abc import ABC
from typing import Optional

from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.quantum_tape_recording_quality import (
    QuantumTapeRecordingQuality,
)


class QuantumTapeLabelArtifact(ChainArtifact, ABC):
    """VHS/Cassette tape label metadata."""

    # Primary label
    title: str  # "Summer in Portland - 1998"
    date_range: str  # "1998-06 to 1999-11"

    # Technical specs
    recording_quality: QuantumTapeRecordingQuality
    counter_start: int = Field(ge=0, le=9999)
    counter_end: Optional[int] = Field(None, ge=0, le=9999)

    # Handwritten notes
    notes: Optional[str] = None  # "Wrote novel. Didn't show anyone."

    # Analog artifacts
    original_label_visible: bool = True  # Gabe Walsh showing through
    original_label_text: Optional[str] = None
    tape_degradation: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Visual metadata
    tape_brand: Optional[str] = Field(default="MEMOREX T-120")
    handwriting_style: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)

    def flatten(self):
        pass

    def save_file(self):
        pass
