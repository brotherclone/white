from enum import Enum


class InfranymMethod(str, Enum):
    """Specific encoding methods for hiding infranyms within different mediums"""

    # MIDI methods
    NOTE_CIPHER = "note_cipher"
    MORSE_DURATION = "morse_duration"
    # Audio methods
    BACKMASK_WHISPER = "backmask_whisper"
    STENOGRAPH_SPECTROGRAM = "stenograph_spectrogram"
    # Text methods
    RIDDLE_POEM = "riddle_poem"
    ACROSTIC_LYRICS = "acrostic_lyrics"
    # Image methods
    LSB_STEGANOGRAPHY = "lsb_steganography"
    ANTI_SIGIL = "anti_sigil"

    DEFAULT = "default"
