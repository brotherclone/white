from typing import List

from pydantic import BaseModel, Field, computed_field


class QuantumTapeInstrumentationConfig(BaseModel):
    """Instrumentation for folk rock production."""

    core: List[str] = Field(
        default_factory=lambda: [
            "acoustic_guitar_fingerpicked",
            "upright_bass",
            "brushes_on_snare",
        ]
    )
    color: List[str] = Field(default_factory=list)  # Additional textures

    @computed_field
    @property
    def all_instruments(self) -> List[str]:
        return self.core + self.color
