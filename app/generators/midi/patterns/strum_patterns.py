#!/usr/bin/env python3
"""
Strum/rhythm pattern templates for the Music Production Pipeline.

Defines rhythm patterns that transform whole-note chord blocks into
rhythmic variations (half notes, quarters, eighths, syncopated, arpeggiated).
Each pattern specifies onset positions and durations relative to a single bar.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path

import mido
import yaml


@dataclass
class StrumPattern:
    """A single-bar strum rhythm template.

    Attributes:
        name: Unique pattern identifier.
        time_sig: Tuple of (numerator, denominator).
        description: Human-readable description.
        is_arpeggio: If True, chord tones are distributed across onsets
                     instead of played simultaneously.
        arp_direction: "up" (low→high) or "down" (high→low). Only used if is_arpeggio.
        onsets: List of beat positions (floats) where notes strike.
        durations: List of durations in beats for each onset. Must match len(onsets).
    """

    name: str
    time_sig: tuple[int, int]
    description: str
    is_arpeggio: bool = False
    arp_direction: str = "up"
    onsets: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)

    def bar_length_beats(self) -> float:
        num, den = self.time_sig
        return num * (4.0 / den)


# ---------------------------------------------------------------------------
# 4/4 Patterns (bar = 4 beats)
# ---------------------------------------------------------------------------

PATTERNS_4_4 = [
    StrumPattern(
        name="whole",
        time_sig=(4, 4),
        description="Whole note — one hit per bar, held full duration",
        onsets=[0],
        durations=[4.0],
    ),
    StrumPattern(
        name="half",
        time_sig=(4, 4),
        description="Half notes — two hits per bar",
        onsets=[0, 2],
        durations=[2.0, 2.0],
    ),
    StrumPattern(
        name="quarter",
        time_sig=(4, 4),
        description="Quarter notes — four hits per bar",
        onsets=[0, 1, 2, 3],
        durations=[1.0, 1.0, 1.0, 1.0],
    ),
    StrumPattern(
        name="eighth",
        time_sig=(4, 4),
        description="Eighth notes — eight hits per bar",
        onsets=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
        durations=[0.5] * 8,
    ),
    StrumPattern(
        name="push",
        time_sig=(4, 4),
        description="Syncopated push — emphasis on 1 and 3, pushes on 2-and and 4-and",
        onsets=[0, 1.5, 2, 3.5],
        durations=[1.5, 0.5, 1.5, 0.5],
    ),
    StrumPattern(
        name="arp_up",
        time_sig=(4, 4),
        description="Arpeggio up — chord tones played low to high across sixteenth subdivisions",
        is_arpeggio=True,
        arp_direction="up",
        onsets=[i * 0.25 for i in range(16)],
        durations=[0.25] * 16,
    ),
    StrumPattern(
        name="arp_down",
        time_sig=(4, 4),
        description="Arpeggio down — chord tones played high to low across sixteenth subdivisions",
        is_arpeggio=True,
        arp_direction="down",
        onsets=[i * 0.25 for i in range(16)],
        durations=[0.25] * 16,
    ),
]

# ---------------------------------------------------------------------------
# 7/8 Patterns (bar = 3.5 beats = 7 eighth notes)
# ---------------------------------------------------------------------------

PATTERNS_7_8 = [
    StrumPattern(
        name="whole",
        time_sig=(7, 8),
        description="Whole — one hit, held full bar",
        onsets=[0],
        durations=[3.5],
    ),
    StrumPattern(
        name="grouped_322",
        time_sig=(7, 8),
        description="3+2+2 grouping — hits on group boundaries",
        onsets=[0, 1.5, 2.5],
        durations=[1.5, 1.0, 1.0],
    ),
    StrumPattern(
        name="grouped_223",
        time_sig=(7, 8),
        description="2+2+3 grouping — hits on group boundaries",
        onsets=[0, 1.0, 2.0],
        durations=[1.0, 1.0, 1.5],
    ),
    StrumPattern(
        name="eighth",
        time_sig=(7, 8),
        description="All seventh eighth notes",
        onsets=[i * 0.5 for i in range(7)],
        durations=[0.5] * 7,
    ),
    StrumPattern(
        name="arp_up",
        time_sig=(7, 8),
        description="Arpeggio up across 7/8 bar in eighth-note subdivisions",
        is_arpeggio=True,
        arp_direction="up",
        onsets=[i * 0.5 for i in range(7)],
        durations=[0.5] * 7,
    ),
    StrumPattern(
        name="arp_down",
        time_sig=(7, 8),
        description="Arpeggio down across 7/8 bar in eighth-note subdivisions",
        is_arpeggio=True,
        arp_direction="down",
        onsets=[i * 0.5 for i in range(7)],
        durations=[0.5] * 7,
    ),
]

# ---------------------------------------------------------------------------
# Fallback patterns (any time signature)
# ---------------------------------------------------------------------------


def make_fallback_patterns(time_sig: tuple[int, int]) -> list[StrumPattern]:
    """Generate minimal patterns for unsupported time signatures."""
    num, den = time_sig
    bar_length = num * (4.0 / den)
    beat_duration = 4.0 / den

    return [
        StrumPattern(
            name="whole",
            time_sig=time_sig,
            description=f"Whole — one hit for {num}/{den}",
            onsets=[0],
            durations=[bar_length],
        ),
        StrumPattern(
            name="beat",
            time_sig=time_sig,
            description=f"Beat subdivision for {num}/{den}",
            onsets=[i * beat_duration for i in range(num)],
            durations=[beat_duration] * num,
        ),
    ]


# ---------------------------------------------------------------------------
# Pattern lookup
# ---------------------------------------------------------------------------

ALL_PATTERNS: list[StrumPattern] = [*PATTERNS_4_4, *PATTERNS_7_8]


def get_patterns_for_time_sig(
    time_sig: tuple[int, int], filter_names: list[str] | None = None
) -> list[StrumPattern]:
    """Get all patterns matching a time signature.

    If filter_names is provided, only return patterns with matching names.
    Falls back to generated patterns if no templates exist for the time signature.
    """
    matches = [p for p in ALL_PATTERNS if p.time_sig == time_sig]
    if not matches:
        matches = make_fallback_patterns(time_sig)

    if filter_names:
        matches = [p for p in matches if p.name in filter_names]

    return matches


# ---------------------------------------------------------------------------
# Pattern application and MIDI rendering (moved from strum_pipeline.py)
# ---------------------------------------------------------------------------


def apply_strum_pattern(
    voicing: list[int],
    pattern: StrumPattern,
    velocity: int = 80,
    ticks_per_beat: int = 480,
) -> list[dict]:
    """Apply a strum pattern to a chord voicing, producing MIDI events for one bar.

    Returns list of {notes: [int], onset_tick: int, duration_ticks: int, velocity: int}.
    For arpeggios, each onset has a single note from the chord.
    For block patterns, each onset has all notes from the chord.
    """
    events = []

    if pattern.is_arpeggio:
        if not voicing:
            return events
        sorted_notes = sorted(voicing)
        if pattern.arp_direction == "down":
            sorted_notes = list(reversed(sorted_notes))

        for i, (onset, duration) in enumerate(zip(pattern.onsets, pattern.durations)):
            note = sorted_notes[i % len(sorted_notes)]
            events.append(
                {
                    "notes": [note],
                    "onset_tick": int(onset * ticks_per_beat),
                    "duration_ticks": int(duration * ticks_per_beat),
                    "velocity": velocity,
                }
            )
    else:
        for onset, duration in zip(pattern.onsets, pattern.durations):
            events.append(
                {
                    "notes": list(voicing),
                    "onset_tick": int(onset * ticks_per_beat),
                    "duration_ticks": int(duration * ticks_per_beat),
                    "velocity": velocity,
                }
            )

    return events


def strum_to_midi_bytes(
    chords: list[list[int]],
    pattern: StrumPattern,
    bpm: int = 120,
    velocity: int = 80,
    ticks_per_beat: int = 480,
    durations: list[float] | None = None,
) -> bytes:
    """Apply a strum pattern to a sequence of chord voicings and produce MIDI bytes.

    Each chord gets one bar with the pattern applied, unless durations is provided.
    When durations is given, each chord gets its assigned duration in bars — the
    pattern repeats for longer chords and truncates for shorter ones.

    Args:
        durations: Optional list of bars per chord (from hr_distribution in review.yml).
                   If None, each chord gets exactly 1.0 bar.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    bar_ticks = int(pattern.bar_length_beats() * ticks_per_beat)

    all_events = []  # (abs_tick, note, velocity, is_on)

    current_offset = 0
    for chord_idx, voicing in enumerate(chords):
        if durations is not None:
            chord_dur_ticks = int(
                durations[chord_idx] * pattern.bar_length_beats() * ticks_per_beat
            )
        else:
            chord_dur_ticks = bar_ticks

        pattern_offset = 0
        while pattern_offset < chord_dur_ticks:
            bar_events = apply_strum_pattern(voicing, pattern, velocity, ticks_per_beat)
            for ev in bar_events:
                abs_on = current_offset + pattern_offset + ev["onset_tick"]
                if abs_on >= current_offset + chord_dur_ticks:
                    break
                abs_off = min(
                    abs_on + ev["duration_ticks"],
                    current_offset + chord_dur_ticks,
                )
                for note in ev["notes"]:
                    all_events.append((abs_on, note, ev["velocity"], True))
                    all_events.append((abs_off, note, 0, False))
            pattern_offset += bar_ticks

        current_offset += chord_dur_ticks

    # Sort: by tick, note-offs before note-ons at same tick
    all_events.sort(key=lambda e: (e[0], not e[3], e[1]))

    prev_tick = 0
    for abs_tick, note, vel, is_on in all_events:
        delta = abs_tick - prev_tick
        msg_type = "note_on" if is_on else "note_off"
        track.append(mido.Message(msg_type, note=note, velocity=vel, time=delta))
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Chord MIDI parsing utilities (moved from strum_pipeline.py)
# ---------------------------------------------------------------------------


def parse_chord_voicings(midi_path: Path) -> list[dict]:
    """Parse an approved chord MIDI file and extract voicings per bar.

    Returns list of dicts: [{notes: [int, ...], velocity: int, bar_ticks: int}, ...]
    """
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat

    events = []
    abs_tick = 0
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            events.append(
                {"note": msg.note, "velocity": msg.velocity, "tick": abs_tick}
            )

    if not events:
        return []

    chords = []
    current_tick = events[0]["tick"]
    current_notes = []
    current_vel = events[0]["velocity"]

    for ev in events:
        if ev["tick"] != current_tick:
            chords.append(
                {
                    "notes": sorted(current_notes),
                    "velocity": current_vel,
                    "tick": current_tick,
                }
            )
            current_tick = ev["tick"]
            current_notes = []
            current_vel = ev["velocity"]
        current_notes.append(ev["note"])

    if current_notes:
        chords.append(
            {
                "notes": sorted(current_notes),
                "velocity": current_vel,
                "tick": current_tick,
            }
        )

    if len(chords) >= 2:
        bar_ticks = chords[1]["tick"] - chords[0]["tick"]
    else:
        bar_ticks = tpb * 4

    for chord in chords:
        chord["bar_ticks"] = bar_ticks

    return chords


def read_approved_harmonic_rhythm(production_dir: Path) -> dict[str, list[float]]:
    """Read approved HR distributions for each section.

    Reads from chords/review.yml (hr_distribution field on each approved candidate),
    which is where HR is stored after the collapse-chord-primitive-phases refactor.
    Returns dict mapping section label → list of bar durations per chord.
    """
    review_path = production_dir / "chords" / "review.yml"
    if not review_path.exists():
        return {}

    with open(review_path) as f:
        review = yaml.safe_load(f)

    durations_by_section: dict[str, list[float]] = {}
    for candidate in review.get("candidates", []):
        status = str(candidate.get("status", "")).lower()
        if status not in ("approved", "accepted"):
            continue
        label = candidate.get("label")
        if not label:
            continue
        section_key = str(label).lower().replace("-", "_").replace(" ", "_")
        if section_key not in durations_by_section:
            dist = candidate.get("hr_distribution")
            if dist and isinstance(dist, list):
                durations_by_section[section_key] = [float(d) for d in dist]

    return durations_by_section
