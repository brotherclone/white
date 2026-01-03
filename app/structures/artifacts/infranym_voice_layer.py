from typing import Optional
from pydantic import BaseModel, Field

from app.structures.enums.infranym_voice_profile import InfranymVoiceProfile


class InfranymVoiceLayer(BaseModel):
    text: str = Field(
        min_length=1, max_length=1000, description="Text to be synthesized into speech"
    )
    voice_profile: Optional[InfranymVoiceProfile] = Field(
        description="Voice profile to use for synthesis",
        default=InfranymVoiceProfile.PROCLAMATION,
    )
    rate: Optional[int] = Field(
        default=150, description="The pace of the speech", ge=50, le=300
    )
    pitch: Optional[float] = Field(default=1.0, description="Pitch shift multiplier")
    volume_db: Optional[float] = Field(
        default=0.0, description="Volume adjustment in dB"
    )
    reverse: bool = Field(default=False, description="Reverse the speech direction")
    stereo_pan: Optional[float] = Field(
        default=0.0, description="Pan -1.0 left, 0.0 center, 1.0 right"
    )
    freq_filter: Optional[tuple] = Field(
        default=None, description="(low_hz, high_hz) bandpass filter"
    )
