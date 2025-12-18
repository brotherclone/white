from typing import Optional

from pydantic import BaseModel, Field


class QuantumTapeProductionAesthetic(BaseModel):
    """Analog tape production settings."""

    tape_simulation: bool = True
    hiss_level: float = Field(0.15, ge=0.0, le=1.0)
    wow_flutter: float = Field(0.05, ge=0.0, le=1.0)

    # Audio processing
    compression: str = "analog_tape_saturation"
    eq: str = "vintage_radio_curve"
    reverb: Optional[str] = None

    # Degradation artifacts
    dropouts: bool = False
    clicks_pops: bool = True
    tracking_noise: bool = True
