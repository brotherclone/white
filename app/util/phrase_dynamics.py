"""Phrase-level dynamic shaping for generated MIDI.

Applies a named velocity curve across a list of MIDI note-on events so that
sections feel performed rather than uniformly programmed.

Curves:
  FLAT          — no change (default; preserves existing accent/ghost tiers)
  LINEAR_CRESC  — velocity ramps from min_vel to max_vel across all notes
  LINEAR_DIM    — velocity ramps from max_vel down to min_vel
  SWELL         — velocity rises to max_vel at the midpoint then falls back

Usage::

    from app.util.phrase_dynamics import DynamicCurve, apply_dynamics_curve, infer_curve

    events = [(tick, note, velocity, is_on), ...]
    curve = infer_curve(section_energy)
    events = apply_dynamics_curve(events, curve, min_vel=60, max_vel=110)
"""

from enum import Enum


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


def infer_curve(section_energy: str) -> DynamicCurve:
    """Infer a dynamic curve from a section energy label.

    Mapping:
      intro   → SWELL   (build from silence)
      chorus  → LINEAR_CRESC  (push into the peak)
      outro   → LINEAR_DIM    (wind down)
      everything else → FLAT  (verse, bridge, pre-chorus: let accent tiers speak)
    """
    key = str(section_energy).lower().strip()
    if key == "intro":
        return DynamicCurve.SWELL
    if key == "high" or key == "chorus":
        return DynamicCurve.LINEAR_CRESC
    if key == "outro":
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

    The curve is applied proportionally across the ordered sequence of note-on
    events — index 0 is the start of the curve, index N-1 is the end.

    Args:
        events:   List of (abs_tick, note, velocity, is_on) tuples.
        curve:    Which dynamic shape to apply.
        min_vel:  Lower velocity clamp for the instrument.
        max_vel:  Upper velocity clamp for the instrument.

    Returns:
        New list with note-on velocities rescaled; original list is not mutated.
    """
    if curve == DynamicCurve.FLAT:
        return events

    # Collect indices of note-on events
    on_indices = [i for i, e in enumerate(events) if e[3] and e[2] > 0]
    n = len(on_indices)
    if n == 0:
        return events

    result = list(events)

    for rank, idx in enumerate(on_indices):
        t = rank / max(n - 1, 1)  # 0.0 → 1.0 across the note sequence

        if curve == DynamicCurve.LINEAR_CRESC:
            scale = t
        elif curve == DynamicCurve.LINEAR_DIM:
            scale = 1.0 - t
        elif curve == DynamicCurve.SWELL:
            # Peaks at midpoint: 0 → 1 → 0 (sine-shaped)
            import math

            scale = math.sin(math.pi * t)
        else:
            scale = 1.0

        # Map scale [0,1] onto [min_vel, max_vel], then clamp
        new_vel = int(min_vel + scale * (max_vel - min_vel))
        new_vel = max(min_vel, min(max_vel, new_vel))

        abs_tick, note, _old_vel, is_on = result[idx]
        result[idx] = (abs_tick, note, new_vel, is_on)

    return result
