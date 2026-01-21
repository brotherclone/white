from typing import Optional, Dict, Any, List, Annotated
from langchain_anthropic import ChatAnthropic
from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.infranym_audio_artifact import InfranymAudioArtifact
from app.structures.artifacts.infranym_encoded_image_artifact import (
    InfranymEncodedImageArtifact,
)
from app.structures.artifacts.infranym_text_render_artifact import (
    InfranymTextRenderArtifact,
)
from app.structures.artifacts.infranym_midi_artifact import InfranymMidiArtifact
from app.structures.artifacts.infranym_text_artifact import InfranymTextArtifact
from app.structures.enums.infranym_medium import InfranymMedium
from app.structures.enums.infranym_method import InfranymMethod
from app.util.agent_state_utils import safe_add


class IndigoAgentState(BaseRainbowAgentState):

    secret_name: Annotated[Optional[str], lambda x, y: y or x] = None
    infranym_medium: Annotated[Optional[InfranymMedium], lambda x, y: y or x] = Field(
        default=None, description="The medium (MIDI/AUDIO/TEXT/IMAGE) for the infranym."
    )
    infranym_method: Annotated[Optional[InfranymMethod], lambda x, y: y or x] = Field(
        default=None, description="The specific encoding method for the infranym."
    )
    infranym_encoded_image: Annotated[
        Optional[InfranymEncodedImageArtifact], lambda x, y: y or x
    ] = Field(
        default=None, description="The encoded image of the infranym secret word."
    )
    infranym_text_render: Annotated[
        Optional[InfranymTextRenderArtifact], lambda x, y: y or x
    ] = Field(default=None, description="The image of the infranym secret word.")
    infranym_text: Annotated[Optional[InfranymTextArtifact], lambda x, y: y or x] = (
        Field(default=None, description="The text of the infranym method.")
    )
    infranym_audio: Annotated[Optional[InfranymAudioArtifact], lambda x, y: y or x] = (
        Field(default=None, description="The audio of the infranym method.")
    )
    infranym_midi: Annotated[Optional[InfranymMidiArtifact], lambda x, y: y or x] = (
        Field(default=None, description="The midi of the infranym method.")
    )
    concepts: Annotated[Optional[str], lambda x, y: y or x] = Field(
        default=None, description="The concepts from other proposals"
    )
    letter_bank: Annotated[Optional[str], lambda x, y: y or x] = Field(
        default=None, description="The letters from the secret name."
    )
    surface_name: Annotated[Optional[str], lambda x, y: y or x] = Field(
        default=None, description="The surface name derived from the secret name."
    )
    anagram_attempts: Annotated[int, lambda x, y: y if y is not None else x] = Field(
        default=0,
        description="The current number of attempts to guess the secret name.",
    )
    anagram_attempt_max: Annotated[int, lambda x, y: y if y is not None else x] = Field(
        default=3,
        description="The maximum number of attempts to guess the secret name.",
    )
    anagram_valid: Annotated[bool, lambda x, y: y if y is not None else x] = Field(
        default=False, description="Whether the anagram is actually valid."
    )
    method_constraints: Annotated[Optional[Dict[str, Any]], lambda x, y: y or x] = (
        Field(
            default_factory=list, description="The constraints for the infranym method."
        )
    )

    text_infranym_method: Annotated[Optional[str], lambda x, y: y or x] = Field(
        default=None, description="Text method chosen: acrostic/riddle/anagram"
    )
    generated_text_lines: Annotated[Optional[List[str]], safe_add] = Field(
        default=None, description="LLM-generated acrostic lines (one per letter)"
    )
    generated_riddle_text: Annotated[Optional[str], lambda x, y: y or x] = Field(
        default=None, description="LLM-generated riddle text"
    )
    text_infranym_difficulty: Annotated[Optional[str], lambda x, y: y or x] = Field(
        default=None, description="Riddle difficulty level: easy/medium/hard"
    )
    llm: Annotated[Optional[ChatAnthropic], lambda x, y: y or x] = Field(
        default=None, description="LLM instance passed from agent for text generation"
    )

    def __init__(self, **data):
        super().__init__(**data)
