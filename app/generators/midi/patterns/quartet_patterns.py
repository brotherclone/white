#!/usr/bin/env python3
"""
Four-part (SATB/quartet) voice pattern library for the Music Production Pipeline.

Expresses alto, tenor, and bass-voice as signed semitone offset templates
relative to the soprano (melody) voice. Each VoicePattern provides an offset
per beat that the pipeline resolves to absolute MIDI pitches and clamps to the
target voice range.

Counterpoint constraints:
  - No parallel perfect 5ths (interval 7) or octaves (interval 12) between
    soprano and any lower voice on consecutive beats.
  - Voice crossing detected and corrected post-generation (alto must stay below
    soprano; tenor below alto; bass-voice below tenor).
  - Per-beat offset change capped at ±4 semitones to prevent awkward leaps.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Voice ranges
# ---------------------------------------------------------------------------

VOICE_RANGES: dict[str, tuple[int, int]] = {
    "alto": (48, 67),  # C3–G4
    "tenor": (43, 62),  # G2–D4
    "bass_voice": (36, 55),  # C2–G3
}

VOICE_ORDER = ["soprano", "alto", "tenor", "bass_voice"]

# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

VELOCITY = {
    "accent": 100,
    "normal": 80,
    "ghost": 50,
}

# ---------------------------------------------------------------------------
# MIDI channel assignments
# ---------------------------------------------------------------------------

QUARTET_CHANNELS = {
    "soprano": 0,
    "alto": 1,
    "tenor": 2,
    "bass_voice": 3,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VoicePattern:
    """A single-bar counterpoint template for one lower voice.

    Attributes:
        name: Unique identifier (e.g. 'alto_thirds_descend').
        voice_type: 'alto', 'tenor', or 'bass_voice'.
        interval_offsets: Signed semitone offsets from soprano per beat.
            Length equals beats_per_bar (typically 4 for 4/4).
        rhythm_offsets: Per-beat duration multipliers (1.0 = full beat,
            0.5 = half beat). Defaults to all 1.0 (sustained per beat).
        energy: 'low', 'medium', or 'high' — used for section-energy matching.
    """

    name: str
    voice_type: str  # alto | tenor | bass_voice
    interval_offsets: list[int]  # semitones below soprano per beat
    rhythm_offsets: list[float] = field(default_factory=list)
    energy: str = "medium"

    def __post_init__(self) -> None:
        if not self.rhythm_offsets:
            self.rhythm_offsets = [1.0] * len(self.interval_offsets)


# ---------------------------------------------------------------------------
# Counterpoint constraint checker
# ---------------------------------------------------------------------------


def _interval(soprano_note: int, voice_note: int) -> int:
    """Absolute semitone interval between soprano and a lower voice."""
    return abs(soprano_note - voice_note) % 12


PARALLEL_PERFECT = {7, 0}  # P5 (mod 12 = 7) and P8/unison (mod 12 = 0)


def check_parallels(
    soprano_notes: list[int],
    voice_notes: list[int],
) -> list[str]:
    """Detect parallel perfect 5ths or octaves between soprano and a voice.

    Compares consecutive beat pairs. Returns a list of human-readable
    violation strings (empty list = no violations).

    Args:
        soprano_notes: Absolute MIDI pitches for the soprano per beat.
        voice_notes: Absolute MIDI pitches for the lower voice per beat.
    """
    violations: list[str] = []
    for i in range(len(soprano_notes) - 1):
        s0, s1 = soprano_notes[i], soprano_notes[i + 1]
        v0, v1 = voice_notes[i], voice_notes[i + 1]
        i0 = _interval(s0, v0)
        i1 = _interval(s1, v1)
        if i0 == i1 and i0 in PARALLEL_PERFECT:
            kind = "P8" if i0 == 0 else "P5"
            violations.append(f"beat {i}→{i + 1}: parallel {kind}")
    return violations


# ---------------------------------------------------------------------------
# Range clamping
# ---------------------------------------------------------------------------


def clamp_to_voice_range(note: int, voice_type: str) -> int:
    """Transpose note by octaves until it fits the voice range.

    If the note is above the range ceiling it is shifted down by octaves;
    if below the floor it is shifted up.  Clamped to [floor, ceil] as
    a last resort.
    """
    low, high = VOICE_RANGES[voice_type]
    while note > high:
        note -= 12
    while note < low:
        note += 12
    return max(low, min(high, note))


# ---------------------------------------------------------------------------
# Voice crossing guard
# ---------------------------------------------------------------------------


def fix_voice_crossing(
    soprano: list[int],
    alto: list[int],
    tenor: list[int],
    bass_voice: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Clamp voices beat-by-beat to enforce soprano ≥ alto ≥ tenor ≥ bass_voice.

    When a voice exceeds the one above it, it is pulled down to one semitone
    below that voice, then range-clamped.  This is a greedy downward push
    rather than a true swap — it preserves the soprano and prioritises the
    upper voices when cascading corrections is needed.

    Returns corrected (alto, tenor, bass_voice) lists.
    """
    n = len(soprano)
    alto = list(alto)
    tenor = list(tenor)
    bass_voice = list(bass_voice)

    for i in range(n):
        # Alto must not exceed soprano
        if alto[i] > soprano[i]:
            alto[i] = soprano[i] - 1
        # Tenor must not exceed alto
        if tenor[i] > alto[i]:
            tenor[i] = alto[i] - 1
        # Bass-voice must not exceed tenor
        if bass_voice[i] > tenor[i]:
            bass_voice[i] = tenor[i] - 1
        # Clamp after adjustments
        alto[i] = clamp_to_voice_range(alto[i], "alto")
        tenor[i] = clamp_to_voice_range(tenor[i], "tenor")
        bass_voice[i] = clamp_to_voice_range(bass_voice[i], "bass_voice")

    return alto, tenor, bass_voice


# ---------------------------------------------------------------------------
# Template library — offset patterns per voice type
# ---------------------------------------------------------------------------
#
# Interval offsets are negative (lower than soprano).  Templates are designed
# for 4/4 — a 4-element offset list covers one bar of quarter-note soprano.
# For other time signatures the pipeline tiles or truncates to match note count.

ALTO_PATTERNS: list[VoicePattern] = [
    VoicePattern(
        "alto_thirds",
        "alto",
        [-4, -3, -4, -3],
        energy="medium",
    ),
    VoicePattern(
        "alto_thirds_descend",
        "alto",
        [-3, -4, -5, -4],
        energy="medium",
    ),
    VoicePattern(
        "alto_contrary",
        "alto",
        [-5, -4, -3, -4],
        energy="medium",
    ),
    VoicePattern(
        "alto_sixth_pedal",
        "alto",
        [-9, -9, -8, -9],
        energy="low",
    ),
    VoicePattern(
        "alto_active",
        "alto",
        [-3, -5, -4, -3],
        energy="high",
    ),
    VoicePattern(
        "alto_sus",
        "alto",
        [-4, -4, -5, -4],
        energy="low",
    ),
]

TENOR_PATTERNS: list[VoicePattern] = [
    VoicePattern(
        "tenor_fifths",
        "tenor",
        [-7, -7, -7, -8],
        energy="medium",
    ),
    VoicePattern(
        "tenor_contrary",
        "tenor",
        [-9, -8, -7, -8],
        energy="medium",
    ),
    VoicePattern(
        "tenor_thirds_below_alto",
        "tenor",
        [-8, -7, -8, -7],
        energy="medium",
    ),
    VoicePattern(
        "tenor_active",
        "tenor",
        [-7, -9, -8, -7],
        energy="high",
    ),
    VoicePattern(
        "tenor_pedal",
        "tenor",
        [-12, -12, -12, -11],
        energy="low",
    ),
    VoicePattern(
        "tenor_step",
        "tenor",
        [-8, -9, -8, -7],
        energy="medium",
    ),
]

BASS_VOICE_PATTERNS: list[VoicePattern] = [
    VoicePattern(
        "bass_voice_root_fifth",
        "bass_voice",
        [-12, -12, -11, -12],
        energy="medium",
    ),
    VoicePattern(
        "bass_voice_octave",
        "bass_voice",
        [-12, -14, -12, -11],
        energy="medium",
    ),
    VoicePattern(
        "bass_voice_contrary",
        "bass_voice",
        [-14, -12, -11, -12],
        energy="medium",
    ),
    VoicePattern(
        "bass_voice_active",
        "bass_voice",
        [-12, -11, -14, -12],
        energy="high",
    ),
    VoicePattern(
        "bass_voice_pedal",
        "bass_voice",
        [-12, -12, -12, -12],
        energy="low",
    ),
    VoicePattern(
        "bass_voice_step",
        "bass_voice",
        [-11, -12, -14, -12],
        energy="medium",
    ),
]

ALL_VOICE_PATTERNS: dict[str, list[VoicePattern]] = {
    "alto": ALTO_PATTERNS,
    "tenor": TENOR_PATTERNS,
    "bass_voice": BASS_VOICE_PATTERNS,
}


def get_patterns_for_voice(
    voice_type: str, energy: str = "medium"
) -> list[VoicePattern]:
    """Return all templates for a voice type, filtered by energy if possible."""
    patterns = ALL_VOICE_PATTERNS.get(voice_type, [])
    matched = [p for p in patterns if p.energy == energy]
    return matched if matched else patterns


# ---------------------------------------------------------------------------
# Counterpoint score
# ---------------------------------------------------------------------------


def counterpoint_score(
    soprano_notes: list[int],
    voice_notes: list[int],
) -> float:
    """Score a voice against the soprano (0.0–1.0, higher is better).

    Penalises:
      - Parallel perfect 5ths/octaves
      - Consecutive identical notes (static voice)
      - Excessive leaps (>7 semitones between consecutive voice notes)
    """
    if not soprano_notes or not voice_notes:
        return 0.0

    n = min(len(soprano_notes), len(voice_notes))
    violations = len(check_parallels(soprano_notes[:n], voice_notes[:n]))
    parallel_penalty = violations / max(n - 1, 1)

    static_count = sum(1 for i in range(1, n) if voice_notes[i] == voice_notes[i - 1])
    static_penalty = static_count / max(n - 1, 1)

    leap_count = sum(
        1 for i in range(1, n) if abs(voice_notes[i] - voice_notes[i - 1]) > 7
    )
    leap_penalty = leap_count / max(n - 1, 1)

    score = 1.0 - (0.5 * parallel_penalty + 0.3 * static_penalty + 0.2 * leap_penalty)
    return round(max(0.0, score), 3)
