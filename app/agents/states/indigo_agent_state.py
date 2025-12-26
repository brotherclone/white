from typing import Optional, Dict, Any

from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.infranym_audio_artifact import InfranymAudioArtifact
from app.structures.artifacts.infranym_image_artifact import InfranymImageArtifact
from app.structures.artifacts.infranym_midi_artifact import InfranymMidiArtifact
from app.structures.artifacts.infranym_text_artifact import InfranymTextArtifact
from app.structures.enums.infranym_medium import InfranymMedium
from app.structures.enums.infranym_method import InfranymMethod


class IndigoAgentState(BaseRainbowAgentState):

    secret_name: Optional[str] = Field(
        default=None, description="A secret name for the song."
    )
    infranym_medium: Optional[InfranymMedium] = Field(
        default=None, description="The medium (MIDI/AUDIO/TEXT/IMAGE) for the infranym."
    )
    infranym_method: Optional[InfranymMethod] = Field(
        default=None, description="The specific encoding method for the infranym."
    )
    infranym_image: Optional[InfranymImageArtifact] = Field(
        default=None, description="The image of the infranym method."
    )
    infranym_text: Optional[InfranymTextArtifact] = Field(
        default=None, description="The text of the infranym method."
    )
    infranym_audio: Optional[InfranymAudioArtifact] = Field(
        default=None, description="The audio of the infranym method."
    )
    infranym_midi: Optional[InfranymMidiArtifact] = Field(
        default=None, description="The midi of the infranym method."
    )
    concepts: Optional[str] = Field(
        default=None, description="The concepts from other proposals"
    )
    letter_bank: Optional[str] = Field(
        default=None, description="The letters from the secret name."
    )
    surface_name: Optional[str] = Field(
        default=None, description="The surface name derived from the secret name."
    )
    anagram_attempts: int = Field(
        default=0,
        description="The current number of attempts to guess the secret name.",
    )
    anagram_attempt_max: int = Field(
        default=3,
        description="The maximum number of attempts to guess the secret name.",
    )
    anagram_valid: bool = Field(
        default=False, description="Whether the anagram is actually valid."
    )
    method_constraints: Optional[Dict[str, Any]] = Field(
        default_factory=list, description="The constraints for the infranym method."
    )

    def __init__(self, **data):
        super().__init__(**data)
