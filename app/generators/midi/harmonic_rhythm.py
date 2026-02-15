#!/usr/bin/env python3
"""
Harmonic rhythm generation — variable chord durations on a half-bar grid.

Generates candidate duration distributions for chord progressions, scores them
against drum accent alignment and ChromaticScorer temporal mode.

Core functions (no pipeline orchestration — see harmonic_rhythm_pipeline.py):
- enumerate_distributions: all valid half-bar distributions for N chords
- extract_drum_accents: parse approved drum MIDI for strong beat positions
- drum_alignment_score: fraction of chord onsets on strong beats
- distribution_to_midi_bytes: block-chord MIDI with variable durations
"""

import io
import random
from itertools import product as iterproduct

import mido


# ---------------------------------------------------------------------------
# Distribution enumeration
# ---------------------------------------------------------------------------

# Duration grid: multiples of 0.5 bars
GRID_UNIT = 0.5  # half a bar
MIN_CHORD_DURATION = 0.5  # minimum bars per chord
MAX_CHORD_DURATION_FACTOR = 2.0  # max total = N * this
MAX_CANDIDATES = 200


def enumerate_distributions(
    n_chords: int,
    seed: int = 42,
) -> list[list[float]]:
    """Enumerate valid chord duration distributions on a half-bar grid.

    Each chord gets at least 0.5 bars. Total section length ranges from
    N * 0.5 to N * 2.0 bars. Durations are multiples of 0.5.

    If more than MAX_CANDIDATES distributions exist, randomly samples
    MAX_CANDIDATES (always including the uniform baseline).

    Returns list of distributions, each a list of N floats (bars per chord).
    """
    if n_chords <= 0:
        return []

    # Possible durations per chord: 0.5, 1.0, 1.5, 2.0, ..., up to N*2.0
    min_total = n_chords * MIN_CHORD_DURATION
    max_total = n_chords * MAX_CHORD_DURATION_FACTOR

    # Each chord can be 1, 2, 3, ... half-bar units
    # Max units per chord = max_total / GRID_UNIT (but practically limited)
    max_units_per_chord = int(max_total / GRID_UNIT)
    min_units_total = int(min_total / GRID_UNIT)
    max_units_total = int(max_total / GRID_UNIT)

    # Each chord gets at least 1 unit (0.5 bars)
    # Generate all combinations where each chord has 1..max_units units
    # and total is within bounds
    per_chord_options = list(range(1, max_units_per_chord + 1))

    # For small chord counts, enumerate exhaustively
    # For larger counts, sample directly
    distributions = []
    uniform = [1.0] * n_chords  # baseline: 1 bar each

    if n_chords <= 6:
        # Enumerate all combos (manageable for up to 6 chords)
        for combo in iterproduct(per_chord_options, repeat=n_chords):
            total = sum(combo)
            if min_units_total <= total <= max_units_total:
                dist = [u * GRID_UNIT for u in combo]
                distributions.append(dist)
    else:
        # Too many combos — sample directly
        rng = random.Random(seed)
        seen = set()
        attempts = 0
        max_attempts = MAX_CANDIDATES * 20
        while len(distributions) < MAX_CANDIDATES * 2 and attempts < max_attempts:
            combo = tuple(rng.randint(1, max_units_per_chord) for _ in range(n_chords))
            total = sum(combo)
            if min_units_total <= total <= max_units_total and combo not in seen:
                seen.add(combo)
                distributions.append([u * GRID_UNIT for u in combo])
            attempts += 1

    # Ensure uniform baseline is always included
    if uniform not in distributions:
        distributions.append(uniform)

    # Cap at MAX_CANDIDATES with seeded sampling
    if len(distributions) > MAX_CANDIDATES:
        rng = random.Random(seed)
        # Remove uniform, sample, then re-add
        distributions.remove(uniform)
        sampled = rng.sample(distributions, MAX_CANDIDATES - 1)
        sampled.append(uniform)
        distributions = sampled

    return distributions


# ---------------------------------------------------------------------------
# Drum accent extraction
# ---------------------------------------------------------------------------

ACCENT_VELOCITY_THRESHOLD = (
    80  # catches accent (120) and normal (90), excludes ghost (45)
)


def extract_drum_accents(
    midi_path: str,
    time_sig: tuple[int, int] = (4, 4),
    ticks_per_beat: int = 480,
) -> list[float]:
    """Parse an approved drum MIDI file and extract accented beat positions.

    Returns a list of beat positions (floats) within one bar where accents
    occur. Positions are in quarter-note beats relative to bar start.

    Accented = velocity >= ACCENT_VELOCITY_THRESHOLD.
    """
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat or ticks_per_beat

    num, den = time_sig
    bar_beats = num * (4.0 / den)
    bar_ticks = int(bar_beats * tpb)

    # Collect all note-on events with absolute ticks
    accents = set()
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity >= ACCENT_VELOCITY_THRESHOLD:
                # Position within bar (wrap around for multi-bar patterns)
                pos_in_bar = abs_tick % bar_ticks
                beat_pos = pos_in_bar / tpb
                accents.add(round(beat_pos, 4))

    # Always include bar start
    accents.add(0.0)

    return sorted(accents)


def accents_to_halfbar_mask(
    accent_positions: list[float],
    time_sig: tuple[int, int] = (4, 4),
) -> list[float]:
    """Convert accent beat positions to a half-bar grid mask.

    Returns list of half-bar boundary positions (in beats) that are "strong"
    — i.e., an accent falls within tolerance of the boundary.

    Tolerance: ± 1 eighth note (0.25 beats in quarter-note terms).
    """
    num, den = time_sig
    bar_beats = num * (4.0 / den)
    half_bar = bar_beats / 2.0
    tolerance = 0.25  # ± 1 eighth note

    # Half-bar boundaries within one bar
    boundaries = [i * half_bar for i in range(int(bar_beats / half_bar) + 1)]
    # Only keep boundaries strictly within the bar (0 up to but not including bar_beats)
    boundaries = [b for b in boundaries if b < bar_beats]

    strong = []
    for boundary in boundaries:
        for accent in accent_positions:
            if abs(accent - boundary) <= tolerance:
                strong.append(round(boundary, 4))
                break

    return strong


def default_accent_mask(time_sig: tuple[int, int] = (4, 4)) -> list[float]:
    """Fallback accent mask when no approved drums exist.

    Only bar starts (every 1.0 bar boundary) are marked as strong.
    Returns [0.0] — just the downbeat.
    """
    return [0.0]


# ---------------------------------------------------------------------------
# Drum alignment scoring
# ---------------------------------------------------------------------------


def drum_alignment_score(
    distribution: list[float],
    accent_mask: list[float],
    time_sig: tuple[int, int] = (4, 4),
) -> float:
    """Score a distribution by how well chord onsets align with drum accents.

    The accent mask tiles (repeats) across the full section length.
    Score = fraction of chord onsets landing on strong half-bar positions.

    Args:
        distribution: list of chord durations in bars
        accent_mask: strong beat positions within one bar (in beats)
        time_sig: time signature tuple

    Returns:
        Alignment score 0.0–1.0
    """
    if not distribution:
        return 0.0

    num, den = time_sig
    bar_beats = num * (4.0 / den)
    tolerance = 0.25  # ± 1 eighth note

    # Compute absolute onset positions in beats
    onsets = []
    current_beat = 0.0
    for dur in distribution:
        onsets.append(current_beat)
        current_beat += dur * bar_beats

    # Check each onset against the tiled accent mask
    aligned = 0
    for onset in onsets:
        # Position within a bar (modulo bar length)
        pos_in_bar = onset % bar_beats
        # Round to avoid floating point issues
        pos_in_bar = round(pos_in_bar, 4)

        for accent in accent_mask:
            if abs(pos_in_bar - accent) <= tolerance:
                aligned += 1
                break

    return aligned / len(onsets) if onsets else 0.0


# ---------------------------------------------------------------------------
# MIDI generation for scoring
# ---------------------------------------------------------------------------


def distribution_to_midi_bytes(
    chords: list[list[int]],
    distribution: list[float],
    bpm: int = 120,
    ticks_per_beat: int = 480,
    time_sig: tuple[int, int] = (4, 4),
) -> bytes:
    """Generate block-chord MIDI from a chord progression with variable durations.

    Each chord is sustained for its assigned duration (in bars). This produces
    simple whole-note-style MIDI suitable for ChromaticScorer evaluation.

    Args:
        chords: list of voicings, each a list of MIDI note numbers
        distribution: bars per chord (same length as chords)
        bpm: tempo
        ticks_per_beat: MIDI resolution
        time_sig: time signature tuple

    Returns:
        MIDI file as bytes
    """
    if len(chords) != len(distribution):
        raise ValueError(
            f"chords ({len(chords)}) and distribution ({len(distribution)}) must have same length"
        )

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    num, den = time_sig
    bar_beats = num * (4.0 / den)

    # Collect all events with absolute ticks
    events = []  # (abs_tick, note, velocity, is_on)

    current_tick = 0
    for voicing, dur_bars in zip(chords, distribution):
        dur_ticks = int(dur_bars * bar_beats * ticks_per_beat)
        for note in voicing:
            events.append((current_tick, note, 80, True))
            events.append((current_tick + dur_ticks, note, 0, False))
        current_tick += dur_ticks

    # Sort: by tick, note-offs before note-ons at same tick
    events.sort(key=lambda e: (e[0], not e[3], e[1]))

    # Convert to delta times
    prev_tick = 0
    for abs_tick, note, vel, is_on in events:
        delta = abs_tick - prev_tick
        msg_type = "note_on" if is_on else "note_off"
        track.append(mido.Message(msg_type, note=note, velocity=vel, time=delta))
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()
