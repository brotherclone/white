import logging

from pydantic import Field

from app.reference.encodings.morse_code import MORSE_CODE
from app.structures.concepts.infranym_midi_encoding import InfranymMidiEncoding
from app.structures.enums.infranym_method import InfranymMethod

logger = logging.getLogger(__name__)


class MorseDurationEncoding(InfranymMidiEncoding):
    """
    Encode secret as note durations using Morse code.

    Example:
        "SOS" -> "... --- ..." (dot-dot-dot, dash-dash-dash, dot-dot-dot)
        - Dot = 240 ticks (eighth note)
        - Dash = 720 ticks (dotted quarter)
        - Letter gap = 240 tick rest
        - Word gap = 960 tick rest

    The rhythm, not the pitch, contains the message.
    """

    method: InfranymMethod = InfranymMethod.MORSE_DURATION
    morse_pattern: str = Field(
        default="", description="Morse code representation (e.g., '... --- ...')"
    )
    carrier_note: int = Field(
        default=60,
        description="Single pitch to use for all morse notes (C4 by default)",
    )
    dot_duration: int = Field(
        default=240, description="MIDI ticks for dot (default: eighth note)"
    )
    dash_duration: int = Field(
        default=720, description="MIDI ticks for dash (default: dotted quarter)"
    )
    letter_gap: int = Field(default=240, description="MIDI ticks between letters")
    word_gap: int = Field(default=960, description="MIDI ticks between words")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.morse_pattern:
            self.morse_pattern = self._encode_to_morse()

    def _encode_to_morse(self) -> str:
        """Convert secret word to Morse code"""
        morse_chars = []
        for char in self.secret_word.upper():
            if char in MORSE_CODE:
                morse_chars.append(MORSE_CODE[char])
            else:
                logger.warning(f"Character '{char}' not in Morse code, skipping")
        return " ".join(morse_chars)
