from typing import Optional
from pydantic import BaseModel

from app.enums.chord_quality import ChordQuality


class Chord(BaseModel):
    root: str
    quality: ChordQuality
    inversion: Optional[int] = None
    extensions: list[str] = []

    def __str__(self):
        chord_str = self.root
        if self.quality == ChordQuality.MINOR:
            chord_str += "m"
        elif self.quality == ChordQuality.DIMINISHED:
            chord_str += "dim"
        elif self.quality == ChordQuality.AUGMENTED:
            chord_str += "aug"
        elif self.quality == ChordQuality.MAJOR7:
            chord_str += "maj7"
        elif self.quality == ChordQuality.MINOR7:
            chord_str += "m7"
        elif self.quality == ChordQuality.DOMINANT7:
            chord_str += "7"
        elif self.quality == ChordQuality.SUSPENDED2:
            chord_str += "sus2"
        elif self.quality == ChordQuality.SUSPENDED4:
            chord_str += "sus4"

        if self.extensions:
            chord_str += "(" + ",".join(self.extensions) + ")"

        return chord_str
