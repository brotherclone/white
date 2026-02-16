#!/usr/bin/env python3
"""
Bass line pattern template library for the Music Production Pipeline.

Provides bass line templates organized by style and energy level. Templates
use tone-selection rules (root, 5th, 3rd, etc.) rather than absolute pitches,
so the pipeline resolves them to actual MIDI notes from any chord voicing.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASS_REGISTER_MIN = 24  # C1
BASS_REGISTER_MAX = 60  # C4
BASS_CHANNEL = 0

VELOCITY = {
    "accent": 100,
    "normal": 80,
    "ghost": 50,
}

ENERGY_ORDER = ["low", "medium", "high"]

# ---------------------------------------------------------------------------
# Template data structure
# ---------------------------------------------------------------------------


@dataclass
class BassPattern:
    """A single-bar bass pattern template.

    Attributes:
        name: Unique pattern identifier (e.g., "root_whole").
        style: Pattern style — root, walking, pedal, arpeggiated, octave, syncopated.
        energy: Energy level — "low", "medium", or "high".
        time_sig: Tuple of (numerator, denominator).
        description: Human-readable description.
        notes: List of (beat_position, tone_selection, velocity_level).
               beat_position: float relative to bar start (0 = beat 1).
               tone_selection: "root", "5th", "3rd", "octave_up", "octave_down",
                               "chromatic_approach", "passing_tone".
               velocity_level: key into VELOCITY dict.
        note_durations: Optional list of durations in beats, parallel to notes.
                        If None, each note sustains until the next note onset
                        (or end of bar for the last note).
    """

    name: str
    style: str
    energy: str
    time_sig: tuple[int, int]
    description: str
    notes: list[tuple[float, str, str]] = field(default_factory=list)
    note_durations: list[float] | None = None

    def bar_length_beats(self) -> float:
        """Bar length in quarter-note beats."""
        num, den = self.time_sig
        return num * (4.0 / den)


# ---------------------------------------------------------------------------
# Tone resolution
# ---------------------------------------------------------------------------


def clamp_to_bass_register(note: int) -> int:
    """Clamp a MIDI note to the bass register (24-60), transposing by octaves."""
    while note > BASS_REGISTER_MAX:
        note -= 12
    while note < BASS_REGISTER_MIN:
        note += 12
    return note


def extract_root(voicing: list[int]) -> int:
    """Extract the root note from a chord voicing (lowest note), clamped to bass register."""
    if not voicing:
        return 36  # C2 fallback
    return clamp_to_bass_register(min(voicing))


def extract_chord_tones(voicing: list[int]) -> dict[str, int]:
    """Extract named chord tones relative to the root from a voicing.

    Returns dict with keys: root, 3rd, 5th (if identifiable from the voicing).
    All values clamped to bass register.
    """
    if not voicing:
        root = 36
        return {"root": root, "3rd": root + 4, "5th": root + 7}

    sorted_notes = sorted(voicing)
    root = extract_root(voicing)
    root_pc = root % 12  # pitch class

    tones = {"root": root}

    # Find intervals relative to root pitch class
    intervals = set()
    for note in sorted_notes:
        interval = (note % 12 - root_pc) % 12
        intervals.add(interval)

    # Identify 3rd (minor=3, major=4)
    if 3 in intervals:
        tones["3rd"] = clamp_to_bass_register(root + 3)
    elif 4 in intervals:
        tones["3rd"] = clamp_to_bass_register(root + 4)
    else:
        tones["3rd"] = clamp_to_bass_register(root + 4)  # default major 3rd

    # Identify 5th (diminished=6, perfect=7, augmented=8)
    if 7 in intervals:
        tones["5th"] = clamp_to_bass_register(root + 7)
    elif 6 in intervals:
        tones["5th"] = clamp_to_bass_register(root + 6)
    elif 8 in intervals:
        tones["5th"] = clamp_to_bass_register(root + 8)
    else:
        tones["5th"] = clamp_to_bass_register(root + 7)  # default perfect 5th

    return tones


def resolve_tone(
    tone_selection: str,
    voicing: list[int],
    next_voicing: list[int] | None = None,
) -> int:
    """Resolve a tone selection to a MIDI note number.

    Args:
        tone_selection: One of root, 5th, 3rd, octave_up, octave_down,
                        chromatic_approach, passing_tone.
        voicing: Current chord voicing (list of MIDI note numbers).
        next_voicing: Next chord's voicing (for approach/passing tones).

    Returns:
        MIDI note number in bass register.
    """
    tones = extract_chord_tones(voicing)
    root = tones["root"]

    if tone_selection == "root":
        return root
    elif tone_selection == "5th":
        return tones["5th"]
    elif tone_selection == "3rd":
        return tones["3rd"]
    elif tone_selection == "octave_up":
        return clamp_to_bass_register(root + 12)
    elif tone_selection == "octave_down":
        return clamp_to_bass_register(root - 12)
    elif tone_selection == "chromatic_approach":
        if next_voicing:
            next_root = extract_root(next_voicing)
            return clamp_to_bass_register(next_root - 1)
        return root  # fallback for last chord
    elif tone_selection == "passing_tone":
        if next_voicing:
            next_root = extract_root(next_voicing)
            # Step toward next root: if next is higher go up by 2, else down by 2
            diff = next_root - root
            if diff > 0:
                return clamp_to_bass_register(root + 2)
            elif diff < 0:
                return clamp_to_bass_register(root - 2)
            return root  # same root
        return root  # fallback for last chord
    else:
        return root  # unknown selection


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------


def energy_distance(a: str, b: str) -> int:
    """Distance between two energy levels (0, 1, or 2)."""
    try:
        return abs(ENERGY_ORDER.index(a) - ENERGY_ORDER.index(b))
    except ValueError:
        return 2


def select_templates(
    all_templates: list[BassPattern],
    time_sig: tuple[int, int],
    target_energy: str,
) -> list[BassPattern]:
    """Select templates matching time signature and energy (exact + adjacent).

    Returns templates sorted by energy distance (exact first).
    """
    results = []
    for t in all_templates:
        if t.time_sig != time_sig:
            continue
        dist = energy_distance(t.energy, target_energy)
        if dist <= 1:
            results.append((dist, t))
    results.sort(key=lambda x: x[0])
    return [t for _, t in results]


def make_fallback_pattern(time_sig: tuple[int, int]) -> BassPattern:
    """Generate a minimal fallback pattern for unsupported time signatures."""
    return BassPattern(
        name="fallback_root",
        style="root",
        energy="medium",
        time_sig=time_sig,
        description=f"Fallback — root on 1 for {time_sig[0]}/{time_sig[1]}",
        notes=[(0, "root", "normal")],
    )


# ---------------------------------------------------------------------------
# Theory scoring
# ---------------------------------------------------------------------------


def root_adherence(
    bass_notes: list[tuple[float, int]],
    chord_root: int,
    time_sig: tuple[int, int] = (4, 4),
) -> float:
    """Fraction of notes on strong beats that are the chord root.

    Strong beats: beat 0 and beat 2 in 4/4, beat 0 in 7/8.
    bass_notes: list of (beat_position, midi_note).
    chord_root: pitch class (0-11) of the chord root.
    """
    num, den = time_sig
    if num == 4 and den == 4:
        strong_beats = {0.0, 2.0}
    elif num == 7 and den == 8:
        strong_beats = {0.0}
    else:
        strong_beats = {0.0}

    strong_notes = [n for pos, n in bass_notes if pos in strong_beats]
    if not strong_notes:
        return 0.0

    root_pc = chord_root % 12
    matches = sum(1 for n in strong_notes if n % 12 == root_pc)
    return matches / len(strong_notes)


def kick_alignment(
    bass_onsets: list[float],
    kick_onsets: list[float],
    tolerance: float = 0.25,  # 1/16 note in quarter-note beats
) -> float:
    """Fraction of bass note onsets coinciding with kick drum hits.

    Args:
        bass_onsets: beat positions of bass note attacks.
        kick_onsets: beat positions of kick drum hits.
        tolerance: maximum distance (in beats) to count as aligned.
    """
    if not bass_onsets or not kick_onsets:
        return 0.0

    aligned = 0
    for bass_pos in bass_onsets:
        for kick_pos in kick_onsets:
            if abs(bass_pos - kick_pos) <= tolerance:
                aligned += 1
                break
    return aligned / len(bass_onsets)


def voice_leading_score(intervals: list[int]) -> float:
    """Score the smoothness of bass movement between adjacent chords.

    Args:
        intervals: list of absolute semitone distances between consecutive
                   bass notes at chord boundaries.
    """
    if not intervals:
        return 1.0

    INTERVAL_SCORES = {
        0: 1.0,  # unison
        1: 1.0,  # semitone
        2: 0.9,  # whole step
        3: 0.8,  # minor 3rd
        4: 0.7,  # major 3rd
        5: 0.5,  # perfect 4th
        6: 0.5,  # tritone
        7: 0.5,  # perfect 5th
    }

    total = 0.0
    for iv in intervals:
        total += INTERVAL_SCORES.get(abs(iv), 0.3)
    return total / len(intervals)


def bass_theory_score(
    root_score: float,
    kick_score: float | None,
    vl_score: float,
) -> float:
    """Compute the mean of available theory components.

    If kick_score is None (no drum data), averages only root + voice_leading.
    """
    if kick_score is not None:
        return (root_score + kick_score + vl_score) / 3.0
    return (root_score + vl_score) / 2.0


# ===========================================================================
# PATTERN TEMPLATES
# ===========================================================================

# ---------------------------------------------------------------------------
# 4/4 Root
# ---------------------------------------------------------------------------

TEMPLATES_4_4_ROOT = [
    BassPattern(
        name="root_whole",
        style="root",
        energy="low",
        time_sig=(4, 4),
        description="Root on beat 1, held for whole bar",
        notes=[(0, "root", "normal")],
        note_durations=[4.0],
    ),
    BassPattern(
        name="root_half",
        style="root",
        energy="medium",
        time_sig=(4, 4),
        description="Root on 1 and 3",
        notes=[(0, "root", "accent"), (2, "root", "normal")],
        note_durations=[2.0, 2.0],
    ),
    BassPattern(
        name="root_quarter",
        style="root",
        energy="high",
        time_sig=(4, 4),
        description="Root on every beat",
        notes=[
            (0, "root", "accent"),
            (1, "root", "normal"),
            (2, "root", "normal"),
            (3, "root", "normal"),
        ],
        note_durations=[1.0, 1.0, 1.0, 1.0],
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Octave
# ---------------------------------------------------------------------------

TEMPLATES_4_4_OCTAVE = [
    BassPattern(
        name="octave_sparse",
        style="octave",
        energy="low",
        time_sig=(4, 4),
        description="Root on 1, octave on 3",
        notes=[(0, "root", "normal"), (2, "octave_up", "ghost")],
        note_durations=[2.0, 2.0],
    ),
    BassPattern(
        name="octave_bounce",
        style="octave",
        energy="medium",
        time_sig=(4, 4),
        description="Root-octave alternating on quarters",
        notes=[
            (0, "root", "accent"),
            (1, "octave_up", "normal"),
            (2, "root", "normal"),
            (3, "octave_up", "normal"),
        ],
        note_durations=[1.0, 1.0, 1.0, 1.0],
    ),
    BassPattern(
        name="octave_eighth",
        style="octave",
        energy="high",
        time_sig=(4, 4),
        description="Root-octave alternating on eighths",
        notes=[
            (0, "root", "accent"),
            (0.5, "octave_up", "ghost"),
            (1, "root", "normal"),
            (1.5, "octave_up", "ghost"),
            (2, "root", "normal"),
            (2.5, "octave_up", "ghost"),
            (3, "root", "normal"),
            (3.5, "octave_up", "ghost"),
        ],
        note_durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Walking
# ---------------------------------------------------------------------------

TEMPLATES_4_4_WALKING = [
    BassPattern(
        name="walking_quarter",
        style="walking",
        energy="medium",
        time_sig=(4, 4),
        description="Root-3rd-5th-approach on quarters",
        notes=[
            (0, "root", "accent"),
            (1, "3rd", "normal"),
            (2, "5th", "normal"),
            (3, "chromatic_approach", "normal"),
        ],
        note_durations=[1.0, 1.0, 1.0, 1.0],
    ),
    BassPattern(
        name="walking_eighth",
        style="walking",
        energy="high",
        time_sig=(4, 4),
        description="Walking quarters with eighth-note passing tones",
        notes=[
            (0, "root", "accent"),
            (0.5, "passing_tone", "ghost"),
            (1, "3rd", "normal"),
            (1.5, "passing_tone", "ghost"),
            (2, "5th", "normal"),
            (2.5, "passing_tone", "ghost"),
            (3, "chromatic_approach", "normal"),
            (3.5, "root", "ghost"),
        ],
        note_durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Arpeggiated
# ---------------------------------------------------------------------------

TEMPLATES_4_4_ARP = [
    BassPattern(
        name="arp_root_5th",
        style="arpeggiated",
        energy="low",
        time_sig=(4, 4),
        description="Root on 1, 5th on 3",
        notes=[(0, "root", "normal"), (2, "5th", "ghost")],
        note_durations=[2.0, 2.0],
    ),
    BassPattern(
        name="arp_triad",
        style="arpeggiated",
        energy="medium",
        time_sig=(4, 4),
        description="Root-3rd-5th on quarters, rest on 4",
        notes=[
            (0, "root", "accent"),
            (1, "3rd", "normal"),
            (2, "5th", "normal"),
        ],
        note_durations=[1.0, 1.0, 1.0],
    ),
    BassPattern(
        name="arp_full",
        style="arpeggiated",
        energy="high",
        time_sig=(4, 4),
        description="Root-3rd-5th-octave on quarters",
        notes=[
            (0, "root", "accent"),
            (1, "3rd", "normal"),
            (2, "5th", "normal"),
            (3, "octave_up", "normal"),
        ],
        note_durations=[1.0, 1.0, 1.0, 1.0],
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Pedal
# ---------------------------------------------------------------------------

TEMPLATES_4_4_PEDAL = [
    BassPattern(
        name="pedal_whole",
        style="pedal",
        energy="low",
        time_sig=(4, 4),
        description="Root held entire bar",
        notes=[(0, "root", "normal")],
        note_durations=[4.0],
    ),
    BassPattern(
        name="pedal_pulse",
        style="pedal",
        energy="medium",
        time_sig=(4, 4),
        description="Root repeated on quarters (re-attacked)",
        notes=[
            (0, "root", "accent"),
            (1, "root", "normal"),
            (2, "root", "normal"),
            (3, "root", "normal"),
        ],
        note_durations=[0.9, 0.9, 0.9, 0.9],  # slight gap for re-attack
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Syncopated
# ---------------------------------------------------------------------------

TEMPLATES_4_4_SYNCOPATED = [
    BassPattern(
        name="syncopated_offbeat",
        style="syncopated",
        energy="medium",
        time_sig=(4, 4),
        description="Root on 1, 5th on 2.5",
        notes=[(0, "root", "accent"), (2.5, "5th", "normal")],
        note_durations=[2.5, 1.5],
    ),
    BassPattern(
        name="syncopated_funk",
        style="syncopated",
        energy="high",
        time_sig=(4, 4),
        description="Root on 1, ghost on 1.5, 5th on 2.5, root on 3.5",
        notes=[
            (0, "root", "accent"),
            (1.5, "root", "ghost"),
            (2.5, "5th", "normal"),
            (3.5, "root", "normal"),
        ],
        note_durations=[1.0, 1.0, 1.0, 0.5],
    ),
]

# ---------------------------------------------------------------------------
# 7/8 Templates
# ---------------------------------------------------------------------------
# 7/8 = 3.5 quarter-note beats. Group positions: 0, 1.5, 2.5 (3+2+2 grouping)

TEMPLATES_7_8 = [
    BassPattern(
        name="root_7_sparse",
        style="root",
        energy="low",
        time_sig=(7, 8),
        description="Root on 1 only",
        notes=[(0, "root", "normal")],
        note_durations=[3.5],
    ),
    BassPattern(
        name="root_7_322",
        style="root",
        energy="medium",
        time_sig=(7, 8),
        description="Root on group starts (0, 1.5, 2.5)",
        notes=[
            (0, "root", "accent"),
            (1.5, "root", "normal"),
            (2.5, "root", "normal"),
        ],
        note_durations=[1.5, 1.0, 1.0],
    ),
    BassPattern(
        name="octave_7_bounce",
        style="octave",
        energy="medium",
        time_sig=(7, 8),
        description="Root-octave on group boundaries",
        notes=[
            (0, "root", "accent"),
            (1.5, "octave_up", "normal"),
            (2.5, "root", "normal"),
        ],
        note_durations=[1.5, 1.0, 1.0],
    ),
    BassPattern(
        name="walking_7",
        style="walking",
        energy="medium",
        time_sig=(7, 8),
        description="Walking on group start positions",
        notes=[
            (0, "root", "accent"),
            (1.5, "3rd", "normal"),
            (2.5, "5th", "normal"),
        ],
        note_durations=[1.5, 1.0, 1.0],
    ),
    BassPattern(
        name="arp_7_322",
        style="arpeggiated",
        energy="medium",
        time_sig=(7, 8),
        description="Root-5th-3rd on group starts",
        notes=[
            (0, "root", "accent"),
            (1.5, "5th", "normal"),
            (2.5, "3rd", "normal"),
        ],
        note_durations=[1.5, 1.0, 1.0],
    ),
]

# ---------------------------------------------------------------------------
# All templates registry
# ---------------------------------------------------------------------------

ALL_TEMPLATES: list[BassPattern] = [
    # 4/4
    *TEMPLATES_4_4_ROOT,
    *TEMPLATES_4_4_OCTAVE,
    *TEMPLATES_4_4_WALKING,
    *TEMPLATES_4_4_ARP,
    *TEMPLATES_4_4_PEDAL,
    *TEMPLATES_4_4_SYNCOPATED,
    # 7/8
    *TEMPLATES_7_8,
]
