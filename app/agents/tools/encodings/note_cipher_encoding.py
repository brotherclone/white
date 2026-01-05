from typing import List, Optional

from pydantic import Field

from app.reference.encodings.letter_to_midi import LETTER_TO_MIDI_MAP
from app.structures.concepts.infranym_midi_encoding import InfranymMidiEncoding
from app.structures.enums.infranym_method import InfranymMethod


class NoteCipherEncoding(InfranymMidiEncoding):
    """
    Encode secret as note pitches using direct letter->MIDI mapping.

    Example:
        "TEMPORAL" -> [79, 64, 72, 75, 74, 77, 60, 71]
        (T=79, E=64, M=72, P=75, O=74, R=77, A=60, L=71)

    The melody literally spells the word if you know the cipher.
    """

    method: InfranymMethod = InfranymMethod.NOTE_CIPHER
    note_sequence: List[int] = Field(
        default_factory=list, description="MIDI note numbers spelling secret"
    )
    track_number: int = Field(default=1, description="Which track contains message")
    octave_offset: int = Field(
        default=0, description="Octave shift for camouflage (-2 to +2)"
    )
    velocity_pattern: Optional[List[int]] = Field(
        default=None, description="Custom velocity for each note (default: uniform 64)"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not self.encoding_map:
            self.encoding_map = LETTER_TO_MIDI_MAP
        if not self.note_sequence:
            self.note_sequence = self._encode_to_notes()

    def _encode_to_notes(self) -> List[int]:
        """Convert secret word to MIDI notes"""
        notes = []
        for char in self.secret_word.upper():
            if char in self.encoding_map:
                base_note = self.encoding_map[char]
                shifted_note = base_note + (self.octave_offset * 12)
                # Clamp to valid MIDI range (0-127)
                notes.append(max(0, min(127, shifted_note)))
        return notes
