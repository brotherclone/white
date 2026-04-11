#!/usr/bin/env python3
"""
Aesthetic-hints tag weighting for pattern selection.

Reads `aesthetic_hints` from song_context.yml and applies a soft score
adjustment to candidates based on their pattern tags.

Weight rules (soft prior — Refractor chromatic score still dominates):
  - Tags matching the density hint: +0.10 bonus
  - Tags contradicting the density hint (e.g. dense when sparse desired): -0.05 penalty
  - No aesthetic_hints: 0.0 (no adjustment)

Density tag mapping:
  sparse  → prefers tags: sparse, ambient, ghost_only, minimal, pedal, drone, lamentful
  dense   → prefers tags: dense, electronic, rhythmic
  moderate → no adjustment

Texture tag mapping (bonus only, no penalty):
  hazy    → prefers tags: hazy, ambient, sparse
  clean   → prefers tags: clean, electronic
  rhythmic → prefers tags: motorik, rhythmic, dense
"""

from typing import Any

# Which pattern tags are preferred for each density level
_DENSITY_PREFER: dict[str, set[str]] = {
    "sparse": {
        "sparse",
        "ambient",
        "ghost_only",
        "minimal",
        "pedal",
        "drone",
        "lamentful",
        "half_time",
        "brushed",
    },
    "dense": {"dense", "electronic", "rhythmic"},
    "moderate": set(),
}

# Which pattern tags conflict with each density level
_DENSITY_CONFLICT: dict[str, set[str]] = {
    "sparse": {"dense"},
    "dense": {"sparse", "drone", "minimal"},
    "moderate": set(),
}

_TEXTURE_PREFER: dict[str, set[str]] = {
    "hazy": {"hazy", "ambient", "sparse"},
    "clean": {"clean", "electronic"},
    "rhythmic": {"motorik", "rhythmic", "dense"},
}


def aesthetic_tag_adjustment(
    pattern_tags: list[str],
    aesthetic_hints: dict[str, Any] | None,
) -> float:
    """Return a composite score adjustment for a pattern given aesthetic hints.

    Returns a float in range [-0.05, +0.10]. Returns 0.0 when hints are absent
    or the pattern has no relevant tags.
    """
    if not aesthetic_hints or not pattern_tags:
        return 0.0

    tags = set(pattern_tags)
    adjustment = 0.0

    density = aesthetic_hints.get("density", "")
    if density:
        prefer = _DENSITY_PREFER.get(density, set())
        conflict = _DENSITY_CONFLICT.get(density, set())
        if tags & prefer:
            adjustment += 0.10
        elif tags & conflict:
            adjustment -= 0.05

    texture = aesthetic_hints.get("texture", "")
    if texture:
        prefer_tex = _TEXTURE_PREFER.get(texture, set())
        if tags & prefer_tex:
            adjustment += 0.05  # smaller bonus for texture match

    return round(adjustment, 3)


# ---------------------------------------------------------------------------
# Arc helpers
# ---------------------------------------------------------------------------

_ARC_LOW = 0.3
_ARC_HIGH = 0.65


def arc_to_energy(arc: float) -> str:
    """Map an arc float (0.0–1.0) to a canonical energy band string.

    Returns:
        "low"    when arc < 0.30
        "medium" when 0.30 <= arc <= 0.65
        "high"   when arc > 0.65
    """
    if arc < _ARC_LOW:
        return "low"
    if arc > _ARC_HIGH:
        return "high"
    return "medium"


def arc_tag_adjustment(arc: float, pattern_tags: list[str]) -> float:
    """Return a score adjustment based on arc and pattern tags.

    Low arc  (< 0.30): +0.10 for drone/pedal tags; no penalty elsewhere
    High arc (> 0.65): -0.05 for root_drone tag; no adjustment elsewhere
    Mid arc           : 0.0

    These adjustments stack with aesthetic_tag_adjustment.
    """
    if not pattern_tags:
        return 0.0
    tags = set(pattern_tags)
    if arc < _ARC_LOW:
        if tags & {"drone", "pedal", "minimal", "sparse", "lamentful"}:
            return 0.10
    elif arc > _ARC_HIGH:
        if "root_drone" in tags:
            return -0.05
    return 0.0


# ---------------------------------------------------------------------------
# Style profile helpers
# ---------------------------------------------------------------------------

# Thresholds for style profile adjustments
_LOW_DENSITY = 2.0
_VERY_LOW_DENSITY = 1.5
_HIGH_REST = 0.5
_LONG_DURATION = 1.5
_LOW_HARMONIC_RHYTHM = 0.5
_HIGH_VEL_VARIANCE = 20.0


def style_profile_tag_adjustment(
    profile: dict | None,
    pattern_tags: list[str],
    instrument: str,
) -> float:
    """Return a score adjustment based on style reference profile.

    profile: dict from song_context.yml `style_reference_profile` block.
    instrument: "drums", "bass", or "melody"
    Returns float adjustment (typically ±0.05–0.10).
    """
    if not profile or not pattern_tags:
        return 0.0

    tags = set(pattern_tags)
    adj = 0.0
    note_density = float(profile.get("note_density", 3.0))
    rest_ratio = float(profile.get("rest_ratio", 0.0))
    mean_duration = float(profile.get("mean_duration_beats", 0.5))
    vel_var = float(profile.get("velocity_variance", 10.0))
    harmonic_rhythm = float(profile.get("harmonic_rhythm", 1.0))
    style_weight = float(profile.get("style_weight", 0.4))

    if instrument == "drums":
        if note_density < _LOW_DENSITY and tags & {
            "sparse",
            "ambient",
            "ghost_only",
            "ambient_pulse",
        }:
            adj += 0.10
        if note_density < _VERY_LOW_DENSITY and tags & {
            "dense",
            "electronic",
            "motorik",
        }:
            adj -= 0.05
        if vel_var > _HIGH_VEL_VARIANCE and tags & {"ghost_only"}:
            adj += 0.05

    elif instrument == "bass":
        if mean_duration > _LONG_DURATION and tags & {"pedal", "drone", "minimal"}:
            adj += 0.10
        if rest_ratio > _HIGH_REST and tags & {"minimal", "drone"}:
            adj += 0.05
        if rest_ratio > _HIGH_REST and tags & {"walking", "arpeggiated"}:
            adj -= 0.05
        if harmonic_rhythm < _LOW_HARMONIC_RHYTHM and tags & {"pedal", "drone"}:
            adj += 0.05

    elif instrument == "melody":
        if rest_ratio > _HIGH_REST and tags & {"sparse", "stepwise", "descent"}:
            adj += 0.10
        if note_density < _LOW_DENSITY and tags & {"dense"}:
            adj -= 0.05
        if mean_duration > _LONG_DURATION and tags & {"stepwise", "descent"}:
            adj += 0.05

    # Scale by style_weight so human can tune influence per-song
    return round(adj * style_weight / 0.4, 3)  # normalise to weight=0.4 baseline
