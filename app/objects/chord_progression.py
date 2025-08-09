from pydantic import BaseModel

from app.objects.chord import Chord


class ChordProgression(BaseModel):
    chords: list[Chord]
    section_name: str
    bars_per_chord: list[int]  # How many bars each chord lasts
    key: str
    mode: str = "major"

    def total_bars(self) -> int:
        return sum(self.bars_per_chord)

    def to_chord_chart(self) -> str:
        """Generate a simple chord chart representation"""
        chart = f"{self.section_name} ({self.key} {self.mode}):\n"
        for i, (chord, bars) in enumerate(zip(self.chords, self.bars_per_chord)):
            chart += f"| {str(chord):<8} "
            if bars > 1:
                chart += f"({bars} bars) "
            if (i + 1) % 4 == 0:  # New line every 4 chords
                chart += "|\n"
        if len(self.chords) % 4 != 0:
            chart += "|\n"
        return chart
