from typing import Optional
from pydantic import BaseModel, Field

from app.structures.artifacts.infranym_voice_layer import InfranymVoiceLayer


class InfranymVoiceComposition(BaseModel):
    surface_layer: InfranymVoiceLayer = Field(
        description="Surface layer of the composition"
    )
    reverse_layer: InfranymVoiceLayer = Field(
        description="Reverse layer of the composition"
    )
    submerged_layer: InfranymVoiceLayer = Field(
        description="Submerged layer of the composition"
    )
    title: str = Field(description="Title of the composition")
    tempo_bpm: Optional[int] = Field(
        default=None, description="For musical timing reference"
    )
    key_signature: Optional[str] = Field(
        default=None, description="For musical context"
    )
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")
