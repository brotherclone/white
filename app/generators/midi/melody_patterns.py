#!/usr/bin/env python3
"""
Melody contour pattern template library for the Music Production Pipeline.

Provides melody templates organised by contour type and energy level. Templates
use relative interval sequences (signed semitone deltas) rather than absolute
pitches, so the pipeline resolves them to actual MIDI pitches within any
singer's vocal range.
"""

from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MELODY_CHANNEL = 0

VELOCITY = {
    "accent": 110,
    "normal": 90,
    "ghost": 60,
}

ENERGY_ORDER = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SingerRange:
    """A singer's vocal range constraint.

    Attributes:
        name: Display name.
        low: Lowest MIDI note.
        high: Highest MIDI note.
        voice_type: baritone, tenor, alto, etc.
    """

    name: str
    low: int
    high: int
    voice_type: str

    @property
    def mid(self) -> int:
        return (self.low + self.high) // 2


SINGERS: dict[str, SingerRange] = {
    "busyayo": SingerRange("Busyayo", 45, 64, "baritone"),
    "gabriel": SingerRange("Gabriel", 48, 67, "tenor"),
    "robbie": SingerRange("Robbie", 48, 67, "tenor"),
    "shirley": SingerRange("Shirley", 53, 72, "alto"),
    "katherine": SingerRange("Katherine", 57, 76, "alto"),
}


@dataclass
class MelodyPattern:
    """A single-bar melodic contour template.

    Attributes:
        name: Unique pattern identifier.
        contour: Contour type — stepwise, arpeggiated, repeated,
                 leap_step, pentatonic, scalar_run.
        energy: Energy level — "low", "medium", or "high".
        time_sig: Tuple of (numerator, denominator).
        description: Human-readable description.
        intervals: Signed semitone deltas from previous note.
                   First element is always 0 (resolved from chord).
        rhythm: Onset positions in beats (same length as intervals).
        durations: Note durations in beats; None = sustain to next onset.
    """

    name: str
    contour: str
    energy: str
    time_sig: tuple[int, int]
    description: str
    intervals: list[int] = field(default_factory=list)
    rhythm: list[float] = field(default_factory=list)
    durations: list[float] | None = None

    def bar_length_beats(self) -> float:
        num, den = self.time_sig
        return num * (4.0 / den)


# ---------------------------------------------------------------------------
# Singer range helpers
# ---------------------------------------------------------------------------


def clamp_to_singer_range(note: int, singer: SingerRange) -> int:
    """Clamp a MIDI note into the singer's range by octave transposition.

    If octave transposition can't fix it, clamp to the nearest boundary.
    """
    while note > singer.high and note - 12 >= singer.low:
        note -= 12
    while note < singer.low and note + 12 <= singer.high:
        note += 12
    return max(singer.low, min(singer.high, note))


def infer_singer(tonic_midi: int) -> SingerRange:
    """Pick the singer whose mid-range best covers the given tonic pitch.

    tonic_midi: MIDI note number of the song's tonic (any octave).
    """
    tonic_pc = tonic_midi % 12
    best_singer = SINGERS["gabriel"]  # default
    best_distance = 999

    for singer in SINGERS.values():
        mid_pc = singer.mid % 12
        dist = min(abs(tonic_pc - mid_pc), 12 - abs(tonic_pc - mid_pc))
        if dist < best_distance:
            best_distance = dist
            best_singer = singer

    return best_singer


# ---------------------------------------------------------------------------
# Melody resolution
# ---------------------------------------------------------------------------


def resolve_melody_notes(
    pattern: MelodyPattern,
    chord_voicing: list[int],
    singer: SingerRange,
    next_voicing: list[int] | None = None,
) -> list[tuple[float, int, float]]:
    """Resolve a contour template to concrete MIDI notes.

    Args:
        pattern: Melody contour template.
        chord_voicing: Current chord voicing (list of MIDI note numbers).
        singer: Singer vocal range constraint.
        next_voicing: Next chord voicing (for phrase ending resolution).

    Returns:
        List of (onset_beat, midi_note, duration_beats).
    """
    if not chord_voicing:
        chord_voicing = [60]  # C4 fallback

    # Starting pitch: highest chord tone within singer's range
    # Melody sits on top of the harmony
    in_range = [n for n in chord_voicing if singer.low <= n <= singer.high]
    if in_range:
        start_note = max(in_range)
    else:
        # No chord tones in range — transpose the highest chord tone into range
        start_note = clamp_to_singer_range(max(chord_voicing), singer)

    # Extract chord tones for snapping (pitch classes)
    chord_tones_pc = set(n % 12 for n in chord_voicing)

    # Walk intervals
    notes = []
    current = start_note
    bar_len = pattern.bar_length_beats()

    for i, interval in enumerate(pattern.intervals):
        if i == 0:
            note = current
        else:
            candidate = current + interval
            if candidate < singer.low or candidate > singer.high:
                # Mirror the interval
                candidate = current - interval
            note = clamp_to_singer_range(candidate, singer)

        # Determine onset and duration
        onset = pattern.rhythm[i] if i < len(pattern.rhythm) else 0.0
        if pattern.durations and i < len(pattern.durations):
            dur = pattern.durations[i]
        elif i + 1 < len(pattern.rhythm):
            dur = pattern.rhythm[i + 1] - onset
        else:
            dur = bar_len - onset

        notes.append((onset, note, dur))
        current = note

    # Strong-beat chord-tone snap
    notes = strong_beat_chord_snap(notes, chord_tones_pc, pattern.time_sig, singer)

    # Phrase ending: last note resolves to root or 5th
    if notes:
        onset, last_note, dur = notes[-1]
        root_pc = min(chord_voicing) % 12
        fifth_pc = (root_pc + 7) % 12
        last_pc = last_note % 12
        if last_pc != root_pc and last_pc != fifth_pc:
            # Snap to nearest root or 5th
            candidates = []
            for target_pc in [root_pc, fifth_pc]:
                for offset in range(-6, 7):
                    cand = last_note + offset
                    if cand % 12 == target_pc and singer.low <= cand <= singer.high:
                        candidates.append((abs(offset), cand))
            if candidates:
                candidates.sort()
                notes[-1] = (onset, candidates[0][1], dur)

    return notes


def strong_beat_chord_snap(
    notes: list[tuple[float, int, float]],
    chord_tones_pc: set[int],
    time_sig: tuple[int, int],
    singer: SingerRange,
) -> list[tuple[float, int, float]]:
    """Snap strong-beat notes to nearest chord tone within 2 semitones.

    Strong beats: beat 0 and beat 2 in 4/4, beat 0 in 7/8.
    """
    num, den = time_sig
    if num == 4 and den == 4:
        strong_beats = {0.0, 2.0}
    else:
        strong_beats = {0.0}

    result = []
    for onset, note, dur in notes:
        if onset in strong_beats and (note % 12) not in chord_tones_pc:
            # Find nearest chord tone within 2 semitones
            best = note
            best_dist = 999
            for offset in range(-2, 3):
                cand = note + offset
                if (cand % 12) in chord_tones_pc and singer.low <= cand <= singer.high:
                    if abs(offset) < best_dist:
                        best_dist = abs(offset)
                        best = cand
            if best_dist <= 2:
                note = best
        result.append((onset, note, dur))
    return result


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------


def energy_distance(a: str, b: str) -> int:
    try:
        return abs(ENERGY_ORDER.index(a) - ENERGY_ORDER.index(b))
    except ValueError:
        return 2


def select_templates(
    all_templates: list[MelodyPattern],
    time_sig: tuple[int, int],
    target_energy: str,
) -> list[MelodyPattern]:
    """Select templates matching time signature and energy (exact + adjacent)."""
    results = []
    for t in all_templates:
        if t.time_sig != time_sig:
            continue
        dist = energy_distance(t.energy, target_energy)
        if dist <= 1:
            results.append((dist, t))
    results.sort(key=lambda x: x[0])
    return [t for _, t in results]


def make_fallback_pattern(time_sig: tuple[int, int]) -> MelodyPattern:
    """Generate a minimal fallback pattern for unsupported time signatures."""
    logger.warning("No melody templates for %s/%s — using fallback", *time_sig)
    return MelodyPattern(
        name="fallback_repeated",
        contour="repeated",
        energy="medium",
        time_sig=time_sig,
        description=f"Fallback — repeated root on beat 1 for {time_sig[0]}/{time_sig[1]}",
        intervals=[0],
        rhythm=[0.0],
    )


# ---------------------------------------------------------------------------
# Theory scoring
# ---------------------------------------------------------------------------


def singability_score(
    notes: list[tuple[float, int, float]],
    singer: SingerRange,
) -> float:
    """Score melody singability (0.0–1.0).

    Components:
    - Interval penalty: large leaps (> octave) penalised, stepwise rewarded
    - Range usage: melodies using < 50% of available range penalised
    - Rest placement: at least one rest per 4 bars
    """
    if len(notes) < 2:
        return 0.5

    pitches = [n for _, n, _ in notes]

    # --- Interval penalty ---
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    interval_scores = []
    for iv in intervals:
        if iv <= 2:
            interval_scores.append(1.0)  # stepwise — ideal
        elif iv <= 4:
            interval_scores.append(0.8)  # small leap
        elif iv <= 7:
            interval_scores.append(0.6)  # medium leap
        elif iv <= 12:
            interval_scores.append(0.4)  # octave leap
        else:
            interval_scores.append(0.1)  # > octave — bad
    interval_component = (
        sum(interval_scores) / len(interval_scores) if interval_scores else 0.5
    )

    # --- Range usage ---
    available_range = singer.high - singer.low
    if available_range == 0:
        range_component = 0.5
    else:
        used_range = max(pitches) - min(pitches)
        usage = used_range / available_range
        if usage < 0.5:
            range_component = usage  # penalty scales linearly
        else:
            range_component = 1.0

    # --- Rest placement ---
    # Check if there are gaps between notes (onset + dur < next onset)
    bar_length = 4.0  # assume 4/4 for this heuristic
    total_beats = notes[-1][0] + notes[-1][2] if notes else 0.0
    bars_count = max(1, total_beats / bar_length)

    gap_count = 0
    for i in range(len(notes) - 1):
        onset_i, _, dur_i = notes[i]
        onset_next = notes[i + 1][0]
        if onset_next > onset_i + dur_i + 0.1:  # small tolerance
            gap_count += 1

    rest_ratio = gap_count / bars_count
    rest_component = min(1.0, rest_ratio * 4.0)  # 1 rest per 4 bars = full score

    return (interval_component + range_component + rest_component) / 3.0


def chord_tone_alignment(
    notes: list[tuple[float, int, float]],
    chord_tones_pc: set[int],
    time_sig: tuple[int, int] = (4, 4),
) -> float:
    """Fraction of strong-beat notes that are chord tones (0.0–1.0)."""
    num, den = time_sig
    if num == 4 and den == 4:
        strong_beats = {0.0, 2.0}
    else:
        strong_beats = {0.0}

    strong_notes = [n for onset, n, _ in notes if onset in strong_beats]
    if not strong_notes:
        return 0.5  # neutral if no strong-beat notes

    matches = sum(1 for n in strong_notes if (n % 12) in chord_tones_pc)
    return matches / len(strong_notes)


def contour_quality(notes: list[tuple[float, int, float]]) -> float:
    """Score melodic contour quality (0.0–1.0).

    Components:
    - Arch shape: high point roughly 2/3 through
    - Variety: penalise excessive repetition (> 4 consecutive same pitches)
    - Resolution: final note should be stable
    """
    if len(notes) < 2:
        return 0.5

    pitches = [n for _, n, _ in notes]

    # --- Arch shape ---
    peak_idx = pitches.index(max(pitches))
    ideal_peak_pos = len(pitches) * 2 / 3
    peak_distance = abs(peak_idx - ideal_peak_pos) / max(len(pitches), 1)
    arch_score = max(0.0, 1.0 - peak_distance)

    # --- Variety ---
    max_consecutive = 1
    current_run = 1
    for i in range(1, len(pitches)):
        if pitches[i] == pitches[i - 1]:
            current_run += 1
            max_consecutive = max(max_consecutive, current_run)
        else:
            current_run = 1

    if max_consecutive > 4:
        variety_score = max(0.0, 1.0 - (max_consecutive - 4) * 0.2)
    else:
        variety_score = 1.0

    # --- Resolution ---
    # Final note stability is handled by phrase ending resolution in resolve_melody_notes,
    # so here we just check it's not the highest or an extreme note
    final = pitches[-1]
    is_extreme = final == max(pitches) or final == min(pitches)
    resolution_score = 0.5 if is_extreme else 1.0

    return (arch_score + variety_score + resolution_score) / 3.0


def melody_theory_score(
    sing: float,
    chord_tone: float,
    contour: float,
) -> float:
    """Compute the mean of the three theory components."""
    return (sing + chord_tone + contour) / 3.0


# ===========================================================================
# PATTERN TEMPLATES
# ===========================================================================

# ---------------------------------------------------------------------------
# 4/4 Stepwise
# ---------------------------------------------------------------------------

TEMPLATES_4_4_STEPWISE = [
    MelodyPattern(
        name="stepwise_ascend_low",
        contour="stepwise",
        energy="low",
        time_sig=(4, 4),
        description="Gentle ascending stepwise on half notes",
        intervals=[0, 2, 2, -1],
        rhythm=[0.0, 2.0, 3.0, 3.5],
        durations=[2.0, 1.0, 0.5, 0.5],
    ),
    MelodyPattern(
        name="stepwise_ascend_med",
        contour="stepwise",
        energy="medium",
        time_sig=(4, 4),
        description="Ascending stepwise on quarters with passing tone",
        intervals=[0, 2, 1, 2, -2, 1, -1, -2],
        rhythm=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
    MelodyPattern(
        name="stepwise_descend_med",
        contour="stepwise",
        energy="medium",
        time_sig=(4, 4),
        description="Descending stepwise motion",
        intervals=[0, -2, -1, -2, 2, 1, -1, -2],
        rhythm=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
    MelodyPattern(
        name="stepwise_wave_high",
        contour="stepwise",
        energy="high",
        time_sig=(4, 4),
        description="Wave-like stepwise eighth notes",
        intervals=[0, 2, 1, -1, -2, 2, 3, -1],
        rhythm=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Arpeggiated
# ---------------------------------------------------------------------------

TEMPLATES_4_4_ARPEGGIATED = [
    MelodyPattern(
        name="arp_up_med",
        contour="arpeggiated",
        energy="medium",
        time_sig=(4, 4),
        description="Arpeggiated ascent through chord tones",
        intervals=[0, 4, 3, 5],
        rhythm=[0.0, 1.0, 2.0, 3.0],
        durations=[1.0, 1.0, 1.0, 1.0],
    ),
    MelodyPattern(
        name="arp_down_high",
        contour="arpeggiated",
        energy="high",
        time_sig=(4, 4),
        description="Arpeggiated descent with pickup",
        intervals=[0, -3, -4, 7, -3, -4, 3, -2],
        rhythm=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Repeated
# ---------------------------------------------------------------------------

TEMPLATES_4_4_REPEATED = [
    MelodyPattern(
        name="repeated_monotone_low",
        contour="repeated",
        energy="low",
        time_sig=(4, 4),
        description="Speech-like repeated note, sparse",
        intervals=[0, 0, 0, 2],
        rhythm=[0.0, 1.0, 2.5, 3.0],
        durations=[1.0, 1.0, 0.5, 1.0],
    ),
    MelodyPattern(
        name="repeated_oscillating_med",
        contour="repeated",
        energy="medium",
        time_sig=(4, 4),
        description="Oscillating between two notes",
        intervals=[0, 2, -2, 2, -2, 2, -2, 0],
        rhythm=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Leap-step
# ---------------------------------------------------------------------------

TEMPLATES_4_4_LEAP_STEP = [
    MelodyPattern(
        name="leap_up_step_down_med",
        contour="leap_step",
        energy="medium",
        time_sig=(4, 4),
        description="Leap up then stepwise descent",
        intervals=[0, 7, -2, -1, -2, -1],
        rhythm=[0.0, 1.0, 1.5, 2.0, 2.5, 3.0],
        durations=[1.0, 0.5, 0.5, 0.5, 0.5, 1.0],
    ),
    MelodyPattern(
        name="leap_down_step_up_high",
        contour="leap_step",
        energy="high",
        time_sig=(4, 4),
        description="Dramatic leap down then stepwise recovery",
        intervals=[0, -7, 2, 1, 2, 1, 2, -1],
        rhythm=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Pentatonic
# ---------------------------------------------------------------------------

TEMPLATES_4_4_PENTATONIC = [
    MelodyPattern(
        name="penta_major_med",
        contour="pentatonic",
        energy="medium",
        time_sig=(4, 4),
        description="Major pentatonic ascent-descent",
        intervals=[0, 2, 2, 3, 2, -2, -3, -2],
        rhythm=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
    MelodyPattern(
        name="penta_minor_low",
        contour="pentatonic",
        energy="low",
        time_sig=(4, 4),
        description="Minor pentatonic fragment",
        intervals=[0, 3, 2, -2, -3],
        rhythm=[0.0, 1.0, 2.0, 2.5, 3.0],
        durations=[1.0, 1.0, 0.5, 0.5, 1.0],
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Scalar run
# ---------------------------------------------------------------------------

TEMPLATES_4_4_SCALAR = [
    MelodyPattern(
        name="scalar_run_high",
        contour="scalar_run",
        energy="high",
        time_sig=(4, 4),
        description="Scale-wise ascending run for bridge/transition",
        intervals=[0, 2, 2, 1, 2, 2, 1, -5],
        rhythm=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
]

# ---------------------------------------------------------------------------
# 7/8 Templates
# ---------------------------------------------------------------------------
# 7/8 = 3.5 quarter-note beats.

TEMPLATES_7_8 = [
    MelodyPattern(
        name="stepwise_7_322",
        contour="stepwise",
        energy="low",
        time_sig=(7, 8),
        description="Stepwise on 3+2+2 group starts",
        intervals=[0, 2, -1],
        rhythm=[0.0, 1.5, 2.5],
        durations=[1.5, 1.0, 1.0],
    ),
    MelodyPattern(
        name="stepwise_7_223",
        contour="stepwise",
        energy="medium",
        time_sig=(7, 8),
        description="Stepwise on 2+2+3 group starts",
        intervals=[0, 2, -2, 1, -1],
        rhythm=[0.0, 1.0, 1.5, 2.0, 2.5],
        durations=[1.0, 0.5, 0.5, 0.5, 1.0],
    ),
    MelodyPattern(
        name="arp_7_up",
        contour="arpeggiated",
        energy="medium",
        time_sig=(7, 8),
        description="Arpeggiated on 3+2+2 groups",
        intervals=[0, 4, 3],
        rhythm=[0.0, 1.5, 2.5],
        durations=[1.5, 1.0, 1.0],
    ),
    MelodyPattern(
        name="arp_7_down",
        contour="arpeggiated",
        energy="high",
        time_sig=(7, 8),
        description="Arpeggiated descent on 7/8 eighths",
        intervals=[0, -3, -4, 7, -3, -4, 3],
        rhythm=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        durations=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ),
    MelodyPattern(
        name="repeated_7_pulse",
        contour="repeated",
        energy="medium",
        time_sig=(7, 8),
        description="Driving repeated note on group starts",
        intervals=[0, 0, 0, 2, -2],
        rhythm=[0.0, 1.0, 1.5, 2.0, 2.5],
        durations=[1.0, 0.5, 0.5, 0.5, 1.0],
    ),
    MelodyPattern(
        name="penta_7_modal",
        contour="pentatonic",
        energy="low",
        time_sig=(7, 8),
        description="Modal pentatonic on asymmetric groups",
        intervals=[0, 3, 2, -2],
        rhythm=[0.0, 1.0, 2.0, 2.5],
        durations=[1.0, 1.0, 0.5, 1.0],
    ),
]

# ---------------------------------------------------------------------------
# All templates registry
# ---------------------------------------------------------------------------

ALL_TEMPLATES: list[MelodyPattern] = [
    # 4/4
    *TEMPLATES_4_4_STEPWISE,
    *TEMPLATES_4_4_ARPEGGIATED,
    *TEMPLATES_4_4_REPEATED,
    *TEMPLATES_4_4_LEAP_STEP,
    *TEMPLATES_4_4_PENTATONIC,
    *TEMPLATES_4_4_SCALAR,
    # 7/8
    *TEMPLATES_7_8,
]
