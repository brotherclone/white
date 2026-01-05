from enum import Enum


class InfranymMedium(str, Enum):
    """The medium/format for hiding infranyms (secret names)"""

    MIDI = "midi"
    AUDIO = "audio"
    TEXT = "text"
    IMAGE = "image"
