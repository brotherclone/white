from enum import Enum


class ChainArtifactFileType(str, Enum):
    MARKDOWN = "md"
    AUDIO = "wav"
    JSON = "json"
    PNG = "png"
    YML = "yml"
    HTML = "html"
    MIDI = "midi"
    TXT = "txt"
