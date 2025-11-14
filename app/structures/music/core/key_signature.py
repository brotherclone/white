from pydantic import BaseModel
from enum import Enum
from app.structures.music.core.notes import Note

class ModeName(Enum):
    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"

class Mode(BaseModel):

    name: ModeName
    intervals: list[int] | None = None  # e.g., ['2', '1', '2', '2', '2', '1', '2'] for Major

    def __init__(self, **data):
        super().__init__(**data)
        if self.name not in ModeName:
            raise ValueError(f"Mode must be one of the following: {', '.join([mode.value for mode in ModeName])}")

    def __str__(self):
        return self.name.value

class KeySignature(BaseModel):

    note: Note
    mode: Mode

    def __init__(self, **data):
        super().__init__(**data)

tempered_modes = {
    #ToDo: Check these - they are slopped in
    'major': Mode(name=ModeName.MAJOR, intervals=[2, 2, 1, 2, 2, 2, 1]),
    'minor': Mode(name=ModeName.MINOR, intervals=[2, 1, 2, 2, 1, 2, 2]),
    'dorian': Mode(name=ModeName.DORIAN, intervals=[2, 1, 2, 2, 2, 1, 2]),
    'phrygian': Mode(name=ModeName.PHRYGIAN, intervals=[1, 2, 2, 2, 1, 2, 2]),
    'lydian': Mode(name=ModeName.LYDIAN, intervals=[2, 2, 2, 1, 2, 2, 1]),
    'mixolydian': Mode(name=ModeName.MIXOLYDIAN, intervals=[2, 2, 1, 2, 2, 1, 2]),
    'aeolian': Mode(name=ModeName.AEOLIAN, intervals=[2, 1, 2, 2, 1, 2, 2]),
}

def get_mode(mode_str: str) -> Mode:
    # Make mode parsing case-insensitive
    mode_str_lower = mode_str.lower()
    if mode_str_lower in tempered_modes:
        return tempered_modes[mode_str_lower]
    else:
        raise ValueError(f"Mode {mode_str} is not a valid tempered mode.")