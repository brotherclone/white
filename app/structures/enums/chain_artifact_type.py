from enum import Enum


class ChainArtifactType(str, Enum):

    TRANSCRIPT = "transcript"
    INSTRUCTIONS_TO_HUMAN = "instructions_to_human"
    SIGIL_DESCRIPTION = "sigil_description"
    DOCUMENT = "doc"
    AUDIO_MOSIAC = "audio_mos"
    RANDOM_AUDIO_BY_COLOR_SEGMENT = "col_audio_"
    NOISE_MIXED_AUDIO = "noise_mixed"
