#!/usr/bin/env python3
"""
Strum/rhythm pattern templates for the Music Production Pipeline.

Defines rhythm patterns that transform whole-note chord blocks into
rhythmic variations (half notes, quarters, eighths, syncopated, arpeggiated).
Each pattern specifies onset positions and durations relative to a single bar.
"""

from dataclasses import dataclass, field


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
