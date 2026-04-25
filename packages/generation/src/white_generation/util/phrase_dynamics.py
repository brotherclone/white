"""Phrase-level dynamic shaping for generated MIDI.

Applies a named velocity curve across a list of MIDI note-on events so that
sections feel performed rather than uniformly programmed.

Curves:
  FLAT          — no change (default; preserves existing accent/ghost tiers)
  LINEAR_CRESC  — multiply velocities from MIN_SCALE up to 1.0 across all notes
  LINEAR_DIM    — multiply velocities from 1.0 down to MIN_SCALE
  SWELL         — multiply velocities peaking at 1.0 at the midpoint (sine-shaped)

Velocities are scaled multiplicatively so that the relative accent/ghost/normal
tier structure is preserved.  A ghost note stays quieter than a normal note
throughout the curve.  ``min_vel`` and ``max_vel`` are applied as hard clamps
*after* scaling.

Usage::

    from white_generation.util.phrase_dynamics import DynamicCurve, apply_dynamics_curve, infer_curve

    events = [(tick, note, velocity, is_on), ...]
    curve = infer_curve(section_label)
    events = apply_dynamics_curve(events, curve, min_vel=60, max_vel=110)
"""

import math
from enum import Enum

# Floor multiplier applied at the quietest point of a non-FLAT curve.
# 0.6 means the softest moment is 60% of the original velocity — still audible.
_MIN_CURVE_SCALE = 0.6


class DynamicCurve(Enum):
    FLAT = "flat"
    LINEAR_CRESC = "linear_cresc"
    LINEAR_DIM = "linear_dim"
    SWELL = "swell"


_CURVE_ALIASES: dict[str, DynamicCurve] = {
    "flat": DynamicCurve.FLAT,
    "linear_cresc": DynamicCurve.LINEAR_CRESC,
    "crescendo": DynamicCurve.LINEAR_CRESC,
    "cresc": DynamicCurve.LINEAR_CRESC,
    "linear_dim": DynamicCurve.LINEAR_DIM,
    "diminuendo": DynamicCurve.LINEAR_DIM,
    "dim": DynamicCurve.LINEAR_DIM,
    "swell": DynamicCurve.SWELL,
}


def parse_curve(value: str) -> DynamicCurve:
    """Parse a string to a DynamicCurve, case-insensitive. Returns FLAT on unknown."""
    return _CURVE_ALIASES.get(str(value).lower().strip(), DynamicCurve.FLAT)


def infer_curve(section_label: str) -> DynamicCurve:
    """Infer a dynamic curve from a section label.

    Matches on the *label name* (e.g. "intro", "chorus"), not the energy tier
    ("low"/"medium"/"high") — pass the section label directly.

    Mapping:
      label contains "intro"  → SWELL   (build from near-silence)
      label contains "chorus" → LINEAR_CRESC  (push into the peak)
      label contains "outro"  → LINEAR_DIM    (wind down)
      everything else         → FLAT
    """
    key = str(section_label).lower().strip()
    if "intro" in key:
        return DynamicCurve.SWELL
    if "chorus" in key:
        return DynamicCurve.LINEAR_CRESC
    if "outro" in key:
        return DynamicCurve.LINEAR_DIM
    return DynamicCurve.FLAT


def apply_dynamics_curve(
    events: list[tuple],
    curve: DynamicCurve,
    min_vel: int,
    max_vel: int,
) -> list[tuple]:
    """Scale note-on velocities in *events* along *curve*.

    *events* is a list of ``(abs_tick, note, velocity, is_on)`` tuples as used
    internally by all three MIDI-generation pipelines.  Note-off events
    (``is_on=False`` or ``velocity==0``) are passed through unchanged.

    Velocities are scaled **multiplicatively** — a scale factor between
    ``_MIN_CURVE_SCALE`` (0.6) and 1.0 is applied to each original velocity,
    so accent/normal/ghost tiers remain in proportion throughout the curve.

    Note-on events are processed in **temporal order** (ascending abs_tick) so
    that the curve tracks the passage of time even when the events list is built
    voice-by-voice (as in the drum pipeline).

    Args:
        events:   List of (abs_tick, note, velocity, is_on) tuples.
        curve:    Which dynamic shape to apply.
        min_vel:  Lower velocity clamp (applied after scaling).
        max_vel:  Upper velocity clamp (applied after scaling).

    Returns:
        New list with note-on velocities rescaled; original list is not mutated.
    """
    if curve == DynamicCurve.FLAT:
        return events

    # Collect indices of note-on events sorted by absolute tick (temporal order)
    on_indices = sorted(
        [i for i, e in enumerate(events) if e[3] and e[2] > 0],
        key=lambda i: events[i][0],
    )
    n = len(on_indices)
    if n == 0:
        return events

    result = list(events)

    for rank, idx in enumerate(on_indices):
        t = rank / max(n - 1, 1)  # 0.0 → 1.0 across temporal note sequence

        if curve == DynamicCurve.LINEAR_CRESC:
            # Scale from MIN_CURVE_SCALE up to 1.0
            scale = _MIN_CURVE_SCALE + t * (1.0 - _MIN_CURVE_SCALE)
        elif curve == DynamicCurve.LINEAR_DIM:
            # Scale from 1.0 down to MIN_CURVE_SCALE
            scale = 1.0 - t * (1.0 - _MIN_CURVE_SCALE)
        elif curve == DynamicCurve.SWELL:
            # Peaks at 1.0 at midpoint, MIN_CURVE_SCALE at both ends (sine-shaped)
            scale = _MIN_CURVE_SCALE + math.sin(math.pi * t) * (1.0 - _MIN_CURVE_SCALE)
        else:
            scale = 1.0

        abs_tick, note, old_vel, is_on = result[idx]
        new_vel = int(round(old_vel * scale))
        new_vel = max(min_vel, min(max_vel, new_vel))
        result[idx] = (abs_tick, note, new_vel, is_on)

    return result
