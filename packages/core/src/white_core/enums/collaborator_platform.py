from enum import Enum


class CollaboratorPlatform(str, Enum):
    AIRGIGS = "airgigs"
    SOUNDBETTER = "soundbetter"
    DIRECT = "direct"
    OTHER = "other"
