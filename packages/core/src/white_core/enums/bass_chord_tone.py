from enum import Enum


class BassChordTone(str, Enum):
    """Chord-tone selection rule for a bass pattern note.

    Used in BassPattern.notes tuples and resolved to a MIDI pitch by
    resolve_tone() in bass_patterns.py.
    """

    ROOT = "root"
    FIFTH = "5th"
    THIRD = "3rd"
    OCTAVE_UP = "octave_up"
    OCTAVE_DOWN = "octave_down"
    CHROMATIC_APPROACH = "chromatic_approach"
    PASSING_TONE = "passing_tone"
