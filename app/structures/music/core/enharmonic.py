# Flat ↔ sharp enharmonic mappings.
# No heavy dependencies — safe to import from both .venv and .venv312.

# flat_to_sharp: normalise flat spellings to the sharp keys used in tempered_notes.
flat_to_sharp: dict[str, str] = {
    "Db": "C#",
    "D♭": "C#",
    "Eb": "D#",
    "E♭": "D#",
    "Gb": "F#",
    "G♭": "F#",
    "Ab": "G#",
    "A♭": "G#",
    "Bb": "A#",
    "B♭": "A#",
    "Cb": "B",
    "C♭": "B",
    "Fb": "E",
    "F♭": "E",
}

# sharp_to_flat: normalise sharp spellings to the flat spellings preferred by
# the chord database and most downstream systems.
sharp_to_flat: dict[str, str] = {
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
    "A#": "Bb",
}


def normalize_to_flat(note_str: str) -> str:
    """Return the flat-spelled enharmonic equivalent, or the note unchanged.

    Used when targeting systems (e.g. the chord database) that store
    accidentals as flats rather than sharps.
    """
    return sharp_to_flat.get(note_str, note_str)
