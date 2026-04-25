from enum import Enum


class ChainArtifactFileType(str, Enum):
    MARKDOWN = "md"
    AUDIO = "wav"
    PNG = "png"
    YML = "yml"
    MIDI = "midi"
    TXT = "txt"
