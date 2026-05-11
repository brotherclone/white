import os
from pathlib import Path

from dotenv import load_dotenv

from white_composition.promote_part import register_part

load_dotenv()


PHASES = ["bass", "melody", "chords", "drums", "quartet"]

if __name__ == "__main__":

    midi_root = Path(
        os.getenv(
            "NON_GEN_MOVER_MIDI_ROOT",
            "/Users/gabrielwalsh/Documents/Music Production/Earthly Frames/White/Tracks/",
        )
    )
    for phase in PHASES:
        phase_dir = midi_root / phase
        if not phase_dir.exists():
            continue
        for midi_file in sorted(phase_dir.glob("*.mid")):
            label = midi_file.stem
            try:
                register_part(
                    midi_path=midi_file,
                    phase=phase,
                    section=label,
                    label=label,
                    production_dir=os.getenv(
                        "NON_GEN_MOVER_PRODUCTION_DIR",
                        "/Volumes/LucidNonsense/White/packages/generation/shrink_wrapped/",
                    ),
                )
                print(f"  ✓  {phase}/{label}")
            except ValueError as e:
                print(f"  ✗  {phase}/{label}: {e}")
