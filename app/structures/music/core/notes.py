from pydantic import BaseModel


class Note(BaseModel):

    pitch_name: str
    pitch_alias: list[str] | None = None  # e.g., ['C', 'Do']
    accidental: str | None = None  # e.g., 'sharp', 'flat',
    frequency: int | None = None
    octave: int | None = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.pitch_name not in ["C", "D", "E", "F", "G", "A", "B"]:
            raise ValueError(
                "Pitch name must be one of the following: C, D, E, F, G, A, B"
            )


tempered_notes = {
    "C": Note(pitch_name="C", pitch_alias=[], accidental=None, frequency=261, octave=4),
    "C#": Note(
        pitch_name="C", pitch_alias=["D♭"], accidental="sharp", frequency=277, octave=4
    ),
    "D": Note(pitch_name="D", pitch_alias=[], accidental=None, frequency=293, octave=4),
    "D#": Note(
        pitch_name="D", pitch_alias=["E♭"], accidental="sharp", frequency=311, octave=4
    ),
    "E": Note(pitch_name="E", pitch_alias=[], accidental=None, frequency=329, octave=4),
    "F": Note(pitch_name="F", pitch_alias=[], accidental=None, frequency=349, octave=4),
    "F#": Note(
        pitch_name="F", pitch_alias=["G♭"], accidental="sharp", frequency=370, octave=4
    ),
    "G": Note(pitch_name="G", pitch_alias=[], accidental=None, frequency=392, octave=4),
    "G#": Note(
        pitch_name="G", pitch_alias=["A♭"], accidental="sharp", frequency=415, octave=4
    ),
    "A": Note(pitch_name="A", pitch_alias=[], accidental=None, frequency=440, octave=4),
    "A#": Note(
        pitch_name="A", pitch_alias=["B♭"], accidental="sharp", frequency=466, octave=4
    ),
    "B": Note(pitch_name="B", pitch_alias=[], accidental=None, frequency=493, octave=4),
}


def get_note(note_str: str) -> Note:
    if note_str in tempered_notes:
        return tempered_notes[note_str]
    else:
        raise ValueError(f"Note {note_str} is not a valid tempered note.")
