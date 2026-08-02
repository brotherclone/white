"""
MIDI Parser for chord library.
Extracts chord data, progressions, and relationships from MIDI files.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido

# ToDo: Use Pydantic Note model for better structure and validation
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_note_to_name(midi_note: int) -> str:
    """Convert MIDI note number to note name with octave."""
    octave = (midi_note // 12) - 1
    note_name = NOTE_NAMES[midi_note % 12]
    return f"{note_name}{octave}"


def parse_key_from_path(path: Path) -> Tuple[str, str, str]:
    """
    Extract key information from directory structure.
    Returns: (key_root, key_quality, relative_minor_or_major)

    Example: '01 - C Major - A Minor' -> ('C', 'Major', 'A Minor')
    """
    parts = path.parts

    # Find the key directory (e.g., '01 - C Major - A Minor')
    key_dir = None
    for part in parts:
        if " Major - " in part or " Minor - " in part:
            key_dir = part
            break

    if not key_dir:
        return ("Unknown", "Unknown", "Unknown")

    # Parse: "01 - C Major - A Minor"
    match = re.match(
        r"^\d+\s*-\s*([A-G][b#]?)\s+(Major|Minor)\s*-\s*([A-G][b#]?)\s+(Major|Minor)",
        key_dir,
    )
    if match:
        primary_root = match.group(1)
        primary_quality = match.group(2)
        relative_root = match.group(3)
        relative_quality = match.group(4)
        return (primary_root, primary_quality, f"{relative_root} {relative_quality}")

    return ("Unknown", "Unknown", "Unknown")


def parse_chord_metadata(path: Path) -> Dict:
    """
    Extract metadata from file path and name.

    Path structure:
    chords/01 - C Major - A Minor/1 Triads/01 - C Major/I - C Maj.mid
    chords/01 - C Major - A Minor/2 Extended Chords/01 - C Major/I - Cmaj7.mid
    """
    parts = path.parts
    filename = path.stem

    # Get key information
    key_root, key_quality, relative_key = parse_key_from_path(path)

    # Determine category from directory
    category = "unknown"
    for part in parts:
        if "Triad" in part:
            category = "triad"
        elif "Extended" in part:
            category = "extended"
        elif "Borrowed" in part or "Modal" in part:
            category = "modal"
        elif "Progression" in part:
            category = "progression"

    # Determine if major or minor based on subdirectory
    mode_in_key = None
    for part in parts:
        if re.match(r"^\d+\s*-\s*[A-G][b#]?\s+Major", part):
            mode_in_key = "Major"
        elif re.match(r"^\d+\s*-\s*[A-G][b#]?\s+Minor", part):
            mode_in_key = "Minor"

    # Parse filename for function and chord name
    # Example: "I - C Maj.mid" or "ii - Dm.mid" or "I - Cmaj7.mid"
    function = None
    chord_name = None

    # Try to match roman numeral patterns, allowing a leading accidental for
    # borrowed/modal functions (e.g. "bII - Db13", "#vi - ...").
    roman_match = re.match(r"^([b#]?[ivIV]+(?:sus)?)\s*-\s*(.+)", filename)
    if roman_match:
        function = roman_match.group(1)
        chord_name = roman_match.group(2).strip()
    else:
        # Might be a progression or other format
        chord_name = filename

    # The directory label uses the major key as the primary root (e.g.
    # "C Major - A Minor").  For minor-mode chords, the correct key_root is
    # the relative-minor root, not the major one.
    effective_key_root = key_root
    if mode_in_key == "Minor" and relative_key and relative_key != "Unknown":
        minor_root = relative_key.split()[0]
        effective_key_root = minor_root

    return {
        "key_root": effective_key_root,
        "key_quality": key_quality,
        "relative_key": relative_key,
        "mode_in_key": mode_in_key,
        "category": category,
        "function": function,
        "chord_name": chord_name,
        "source_file": str(path),
    }


def extract_chord_from_midi(midi_file: Path) -> Optional[Dict]:
    """
    Extract chord notes and timing from a single-chord MIDI file.
    Returns chord data or None if parsing fails.
    """
    try:
        mid = mido.MidiFile(midi_file)
    except Exception as e:
        print(f"Error reading {midi_file}: {e}")
        return None

    # Collect all note_on events
    notes = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append(msg.note)

    if not notes:
        return None

    # Remove duplicates and sort
    notes = sorted(set(notes))

    # Calculate intervals from root (lowest note)
    root = min(notes)
    intervals = [note - root for note in notes]

    # Get metadata
    metadata = parse_chord_metadata(midi_file)

    # Determine chord quality from intervals
    quality = infer_chord_quality(intervals)

    return {
        **metadata,
        "midi_notes": notes,
        "root_note": root,
        "bass_note": root,  # For basic chords, root = bass
        "intervals": intervals,
        "quality": quality,
        "num_notes": len(notes),
        "note_names": [midi_note_to_name(n) for n in notes],
        "duration_seconds": mid.length,
        "ticks_per_beat": mid.ticks_per_beat,
    }


def infer_chord_quality(intervals: List[int]) -> str:
    """
    Infer chord quality from intervals, handling octave doublings in voicings.

    Raw intervals from multi-octave MIDI voicings (e.g. [0,7,12,16]) contain
    duplicated pitch classes once octave-reduced.  Reducing to mod-12 before
    matching means C3-G3-C4-E4 correctly identifies as "major" rather than
    "unknown".  More specific (complex) chords are checked first so that, for
    example, an add9 [0,2,4,7] is not misidentified as sus2 [0,2,7].
    """
    # Reduce to pitch classes to collapse octave doublings
    pc = set(i % 12 for i in intervals)

    # Ninth chords (most specific — check before 7ths to avoid partial matches)
    if {0, 2, 4, 7, 11} <= pc:
        return "maj9"
    if {0, 2, 3, 7, 10} <= pc:
        return "min9"
    if {0, 2, 4, 7, 10} <= pc:
        return "dom9"

    # Seventh chords
    if {0, 4, 7, 11} <= pc:
        return "maj7"
    if {0, 3, 7, 10} <= pc:
        return "min7"
    if {0, 4, 7, 10} <= pc:
        return "dom7"
    if {0, 3, 6, 10} <= pc:
        return "min7b5"
    if {0, 3, 6, 9} <= pc:
        return "dim7"

    # Add chords (before triads+sus — {0,2,4,7} contains {0,2,7} as subset)
    if {0, 2, 4, 7} <= pc:
        return "add9"
    if {0, 4, 7, 9} <= pc:
        return "add6"

    # Triads (exact pitch-class match after extended chords are ruled out)
    if pc == {0, 4, 7}:
        return "major"
    if pc == {0, 3, 7}:
        return "minor"
    if pc == {0, 3, 6}:
        return "diminished"
    if pc == {0, 4, 8}:
        return "augmented"

    # Suspended
    if {0, 5, 7} <= pc:
        return "sus4"
    if {0, 2, 7} <= pc:
        return "sus2"

    return "unknown"


def extract_progression_from_midi(midi_file: Path) -> Optional[Dict]:
    """
    Extract chord progression from a MIDI file.
    Returns progression data with individual chords and timing.
    """
    try:
        mid = mido.MidiFile(midi_file)
    except Exception as e:
        print(f"Error reading {midi_file}: {e}")
        return None

    # Track chords over time
    current_notes = set()
    abs_time = 0
    chord_events = []

    for track in mid.tracks:
        for msg in track:
            abs_time += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                # If starting a new chord (first note), save previous
                if not current_notes and chord_events:
                    # Mark previous chord end time
                    if chord_events:
                        chord_events[-1]["end_time"] = abs_time

                current_notes.add(msg.note)

                # If this is first note of a new chord
                if len(current_notes) == 1:
                    chord_events.append(
                        {"start_time": abs_time, "notes": set(), "end_time": None}
                    )

                # Add note to current chord
                if chord_events:
                    chord_events[-1]["notes"].add(msg.note)

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                if msg.note in current_notes:
                    current_notes.remove(msg.note)

                # If all notes released, chord is complete
                if (
                    not current_notes
                    and chord_events
                    and chord_events[-1]["end_time"] is None
                ):
                    chord_events[-1]["end_time"] = abs_time

    # Clean up last chord
    if chord_events and chord_events[-1]["end_time"] is None:
        chord_events[-1]["end_time"] = abs_time

    # Convert to chord list
    chords = []
    for event in chord_events:
        if not event["notes"]:
            continue

        notes = sorted(event["notes"])
        root = min(notes)
        intervals = [n - root for n in notes]
        duration_ticks = event["end_time"] - event["start_time"]

        chords.append(
            {
                "midi_notes": notes,
                "root_note": root,
                "bass_note": root,
                "intervals": intervals,
                "quality": infer_chord_quality(intervals),
                "start_time_ticks": event["start_time"],
                "duration_ticks": duration_ticks,
                "note_names": [midi_note_to_name(n) for n in notes],
            }
        )

    if not chords:
        return None

    # Get metadata
    metadata = parse_chord_metadata(midi_file)

    return {
        **metadata,
        "num_chords": len(chords),
        "total_duration_seconds": mid.length,
        "ticks_per_beat": mid.ticks_per_beat,
        "chords": chords,
    }


def parse_all_chords(chords_dir: Path, exclude_progressions: bool = True) -> List[Dict]:
    """
    Parse all individual chord MIDI files.
    """
    chords = []

    for midi_file in chords_dir.rglob("*.mid"):
        # Skip progression files if requested
        if exclude_progressions and "Progression" in str(midi_file):
            continue

        # Skip per-category reference files (e.g. "All Borrowed & Modal Chords
        # (F Minor).mid") — these stack every chord tone in the category into
        # one sustained cluster and are not a playable chord.
        if midi_file.stem.startswith("All "):
            continue

        chord_data = extract_chord_from_midi(midi_file)
        if chord_data:
            chords.append(chord_data)

    return chords


def parse_all_progressions(chords_dir: Path) -> List[Dict]:
    """
    Parse all progression MIDI files.
    """
    progressions = []

    for midi_file in chords_dir.rglob("*.mid"):
        # Only process progression files
        if "Progression" not in str(midi_file):
            continue

        prog_data = extract_progression_from_midi(midi_file)
        if prog_data:
            progressions.append(prog_data)

    return progressions
