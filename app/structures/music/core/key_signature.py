from enum import Enum

from pydantic import BaseModel, model_validator

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
    intervals: list[int] | None = None

    def __str__(self):
        return self.name.value


class KeySignature(BaseModel):
    note: Note
    mode: Mode

    @model_validator(mode="before")
    @classmethod
    def parse_key_string(cls, data):
        """
        Parse key signature from string format (e.g., 'C# minor') or structured dict.
        Handles YAML entries like: key: "C# minor"
        """
        # If already structured with note and mode, pass through
        if isinstance(data, dict) and "note" in data and "mode" in data:
            return data

        # Parse string format: "NOTE MODE"
        if isinstance(data, str):
            parts = data.strip().split()
            if len(parts) != 2:
                raise ValueError(f"Key signature must be 'NOTE MODE', got: {data}")

            note_str, mode_str = parts
            mode_str = mode_str.lower()

            # Note mapping: string -> (pitch_name, accidental)
            note_map = {
                "C": ("C", None),
                "C#": ("C", "sharp"),
                "Db": ("D", "flat"),
                "D": ("D", None),
                "D#": ("D", "sharp"),
                "Eb": ("E", "flat"),
                "E": ("E", None),
                "F": ("F", None),
                "F#": ("F", "sharp"),
                "Gb": ("G", "flat"),
                "G": ("G", None),
                "G#": ("G", "sharp"),
                "Ab": ("A", "flat"),
                "A": ("A", None),
                "A#": ("A", "sharp"),
                "Bb": ("B", "flat"),
                "B": ("B", None),
            }

            if note_str not in note_map:
                raise ValueError(f"Invalid note: {note_str}")

            pitch_name, accidental = note_map[note_str]

            # Get the mode from tempered_modes to include intervals
            if mode_str not in tempered_modes:
                raise ValueError(
                    f"Invalid mode: {mode_str}. Must be one of: {', '.join(tempered_modes.keys())}"
                )

            mode_obj = tempered_modes[mode_str]

            return {
                "note": {"pitch_name": pitch_name, "accidental": accidental},
                "mode": {
                    "name": mode_obj.name.value,  # Convert enum to string for Pydantic
                    "intervals": mode_obj.intervals,
                },
            }

        return data


tempered_modes = {
    "major": Mode(name=ModeName.MAJOR, intervals=[2, 2, 1, 2, 2, 2, 1]),
    "minor": Mode(name=ModeName.MINOR, intervals=[2, 1, 2, 2, 1, 2, 2]),
    "dorian": Mode(name=ModeName.DORIAN, intervals=[2, 1, 2, 2, 2, 1, 2]),
    "phrygian": Mode(name=ModeName.PHRYGIAN, intervals=[1, 2, 2, 2, 1, 2, 2]),
    "lydian": Mode(name=ModeName.LYDIAN, intervals=[2, 2, 2, 1, 2, 2, 1]),
    "mixolydian": Mode(name=ModeName.MIXOLYDIAN, intervals=[2, 2, 1, 2, 2, 1, 2]),
    "aeolian": Mode(name=ModeName.AEOLIAN, intervals=[2, 1, 2, 2, 1, 2, 2]),
    "locrian": Mode(
        name=ModeName.LOCRIAN, intervals=[1, 2, 2, 1, 2, 2, 2]
    ),  # Added Locrian!
    "harmonic_minor": Mode(
        name=ModeName.HARMONIC_MINOR, intervals=[2, 1, 2, 2, 1, 3, 1]
    ),  # Added Harmonic Minor!
    "melodic_minor": Mode(
        name=ModeName.MELODIC_MINOR, intervals=[2, 1, 2, 2, 2, 2, 1]
    ),  # Added Melodic Minor!
}


def get_mode(mode_str: str) -> Mode:
    # Make mode parsing case-insensitive
    mode_str_lower = mode_str.lower()
    if mode_str_lower in tempered_modes:
        return tempered_modes[mode_str_lower]
    else:
        raise ValueError(f"Mode {mode_str} is not a valid tempered mode.")
