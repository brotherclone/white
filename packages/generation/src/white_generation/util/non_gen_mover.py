# Configuration for non-generative MIDI processing

from pathlib import Path

from white_composition.promote_part import register_part

MIDI_ROOT = "/Users/gabrielwalsh/Documents/Music Production/Earthly Frames/White/Tracks/violet-fallback-defensive-violet-response/The Cataloguer's Lament/MIDI"
PRODUCTION_DIR = "/Volumes/LucidNonsense/White/packages/generation/shrink_wrapped/violet-fallback-defensive-violet-response/production/flesh_circuit_taxonomy_v2"
PHASES = ["bass", "melody", "chords", "drums"]

if __name__ == "__main__":

    midi_root = Path(MIDI_ROOT)
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
                    production_dir=PRODUCTION_DIR,
                )
                print(f"  ✓  {phase}/{label}")
            except ValueError as e:
                print(f"  ✗  {phase}/{label}: {e}")
