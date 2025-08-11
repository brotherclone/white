from pydantic import BaseModel


class Note(BaseModel):

    pitch_name: str
    pitch_alias: list[str] | None = None  # e.g., ['C', 'Do']
    accidental: str | None = None  # e.g., 'sharp', 'flat',
    frequency: int | None = None
    octave: int | None = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.pitch_name not in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
            raise ValueError("Pitch name must be one of the following: C, D, E, F, G, A, B")

