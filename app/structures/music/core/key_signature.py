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