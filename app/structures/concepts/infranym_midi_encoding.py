from abc import ABC
from typing import Dict

from pydantic import BaseModel, Field

from app.structures.enums.infranym_method import InfranymMethod


class InfranymMidiEncoding(BaseModel, ABC):
    """Base for MIDI encoding"""

    method: InfranymMethod
    secret_word: str = Field(..., description="The word/phrase to encode")
    encoding_map: Dict[str, int] = Field(
        default_factory=dict, description="Letter->MIDI value mapping"
    )
