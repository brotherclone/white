from enum import Enum


class BassStyle(str, Enum):
    """Structural style of a bass line pattern template."""

    ROOT = "root"
    WALKING = "walking"
    PEDAL = "pedal"
    ARPEGGIATED = "arpeggiated"
    OCTAVE = "octave"
    SYNCOPATED = "syncopated"
    REGGAE = "reggae"
