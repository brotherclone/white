#!/usr/bin/env python3
"""
Drum pattern template library for the Music Production Pipeline.

Provides multi-voice drum templates organized by genre family and energy level,
with GM percussion MIDI mapping and velocity dynamics (accent/normal/ghost).

Templates are structured data — no procedural generation. Each template defines
per-voice onset positions and velocity levels relative to a single bar.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# GM Percussion MIDI mapping (channel 10)
# ---------------------------------------------------------------------------

GM_PERCUSSION = {
    "kick": 36,
    "snare": 38,
    "rimshot": 37,
    "hh_closed": 42,
    "hh_open": 46,
    "hh_pedal": 44,
    "crash": 49,
    "ride": 51,
    "tom_high": 50,
    "tom_mid": 47,
    "tom_low": 45,
    "clap": 39,
    "tambourine": 54,
    "cowbell": 56,
    "conga_high": 63,
    "conga_low": 64,
}

# ---------------------------------------------------------------------------
# Velocity levels
# ---------------------------------------------------------------------------

VELOCITY = {
    "accent": 120,
    "normal": 90,
    "ghost": 45,
}

# ---------------------------------------------------------------------------
# Template data structure
# ---------------------------------------------------------------------------


# ToDo: Convert to Pydantic model if we want validation and better IDE support. For now, dataclass is sufficient for static templates.
@dataclass
class DrumPattern:
    """A single-bar drum pattern template.

    Attributes:
        name: Unique pattern identifier (e.g., "motorik").
        genre_family: Genre family this pattern belongs to.
        energy: Energy level — "low", "medium", or "high".
        time_sig: Tuple of (numerator, denominator).
        description: Human-readable description of the pattern feel.
        voices: Dict mapping voice names to lists of (beat_position, velocity_level).
                Beat positions are floats relative to bar start (0 = beat 1).
                Velocity levels are keys into VELOCITY dict.
    """

    name: str
    genre_family: str
    energy: str
    time_sig: tuple[int, int]
    description: str
    voices: dict[str, list[tuple[float, str]]] = field(default_factory=dict)

    def bar_length_beats(self) -> float:
        """Bar length in quarter-note beats."""
        num, den = self.time_sig
        return num * (4.0 / den)


# ---------------------------------------------------------------------------
# Genre family mapping
# ---------------------------------------------------------------------------

GENRE_FAMILY_KEYWORDS = {
    "ambient": ["ambient", "drone", "atmospheric", "soundscape"],
    "electronic": ["electronic", "synth", "idm", "glitch", "microsound"],
    "krautrock": ["krautrock", "motorik", "kosmische", "neu!", "can"],
    "rock": ["rock", "post-punk", "punk", "garage", "alternative"],
    "classical": ["classical", "post-classical", "orchestral", "chamber"],
    "experimental": ["experimental", "noise", "industrial", "avant-garde"],
    "folk": ["folk", "acoustic", "singer-songwriter"],
    "americana": ["americana", "country", "bluegrass", "appalachian", "roots", "waltz"],
    "jazz": ["jazz", "swing", "bebop", "fusion"],
    "breakbeat": [
        "breakbeat",
        "hip-hop",
        "boom bap",
        "break",
        "funk",
        "soul",
        "r&b",
        "groove",
    ],
}

DEFAULT_GENRE_FAMILY = "electronic"


def map_genres_to_families(genre_tags: list[str]) -> list[str]:
    """Map song proposal genre tags to genre families via keyword scan.

    Returns a list of matched genre families (may be empty, caller handles fallback).
    """
    matched = set()
    for tag in genre_tags:
        tag_lower = tag.lower()
        for family, keywords in GENRE_FAMILY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in tag_lower:
                    matched.add(family)
    return sorted(matched)


# ---------------------------------------------------------------------------
# Section energy mapping
# ---------------------------------------------------------------------------

DEFAULT_SECTION_ENERGY = {
    "intro": "low",
    "verse": "medium",
    "chorus": "high",
    "bridge": "low",
    "outro": "medium",
}

ENERGY_ORDER = ["low", "medium", "high"]


def energy_distance(a: str, b: str) -> int:
    """Distance between two energy levels (0, 1, or 2)."""
    try:
        return abs(ENERGY_ORDER.index(a) - ENERGY_ORDER.index(b))
    except ValueError:
        return 2


def energy_appropriateness(template_energy: str, target_energy: str) -> float:
    """Score how appropriate a template's energy is for the target section.

    Returns 1.0 for exact match, 0.5 for one level away, 0.0 for two levels away.
    """
    dist = energy_distance(template_energy, target_energy)
    if dist == 0:
        return 1.0
    elif dist == 1:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------


def select_templates(
    all_templates: list[DrumPattern],
    time_sig: tuple[int, int],
    genre_families: list[str],
    target_energy: str,
) -> list[DrumPattern]:
    """Select templates matching time signature, genre families, and energy.

    Includes exact energy matches and one-level-adjacent templates.
    Returns templates sorted by energy appropriateness (exact first).
    """
    results = []
    for t in all_templates:
        if t.time_sig != time_sig:
            continue
        if t.genre_family not in genre_families:
            continue
        dist = energy_distance(t.energy, target_energy)
        if dist <= 1:
            results.append((dist, t))
    results.sort(key=lambda x: x[0])
    return [t for _, t in results]


# ===========================================================================
# PATTERN TEMPLATES
# ===========================================================================

# ---------------------------------------------------------------------------
# 4/4 Ambient
# ---------------------------------------------------------------------------

TEMPLATES_4_4_AMBIENT = [
    DrumPattern(
        name="ambient_breath",
        genre_family="ambient",
        energy="low",
        time_sig=(4, 4),
        description="Sparse breath — kick on 1, soft hat on 3, lots of space",
        voices={
            "kick": [(0, "normal")],
            "hh_closed": [(2, "ghost")],
        },
    ),
    DrumPattern(
        name="ambient_pulse",
        genre_family="ambient",
        energy="medium",
        time_sig=(4, 4),
        description="Gentle pulse — kick on 1 and 3, ghost hats on eighths",
        voices={
            "kick": [(0, "normal"), (2, "ghost")],
            "hh_closed": [
                (0, "ghost"),
                (0.5, "ghost"),
                (1, "ghost"),
                (1.5, "ghost"),
                (2, "ghost"),
                (2.5, "ghost"),
                (3, "ghost"),
                (3.5, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="ambient_build",
        genre_family="ambient",
        energy="high",
        time_sig=(4, 4),
        description="Ambient build — kick on 1 and 3, rimshot on 2 and 4, hats open on offbeats",
        voices={
            "kick": [(0, "accent"), (2, "normal")],
            "rimshot": [(1, "normal"), (3, "normal")],
            "hh_closed": [(0, "ghost"), (1, "ghost"), (2, "ghost"), (3, "ghost")],
            "hh_open": [(0.5, "ghost"), (1.5, "ghost"), (2.5, "ghost"), (3.5, "ghost")],
        },
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Electronic
# ---------------------------------------------------------------------------

TEMPLATES_4_4_ELECTRONIC = [
    DrumPattern(
        name="electronic_minimal",
        genre_family="electronic",
        energy="low",
        time_sig=(4, 4),
        description="Minimal electronic — kick on 1, soft clap on 3, pedal hat on quarters",
        voices={
            "kick": [(0, "normal")],
            "clap": [(2, "ghost")],
            "hh_pedal": [(0, "ghost"), (1, "ghost"), (2, "ghost"), (3, "ghost")],
        },
    ),
    DrumPattern(
        name="electronic_pulse",
        genre_family="electronic",
        energy="medium",
        time_sig=(4, 4),
        description="Steady pulse — kick on 1 and 3, clap on 2 and 4, closed hats on eighths",
        voices={
            "kick": [(0, "accent"), (2, "normal")],
            "clap": [(1, "normal"), (3, "normal")],
            "hh_closed": [
                (0, "normal"),
                (0.5, "ghost"),
                (1, "normal"),
                (1.5, "ghost"),
                (2, "normal"),
                (2.5, "ghost"),
                (3, "normal"),
                (3.5, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="electronic_four_on_floor",
        genre_family="electronic",
        energy="high",
        time_sig=(4, 4),
        description="Four-on-the-floor — kick every beat, clap on 2 and 4, open hat on offbeats",
        voices={
            "kick": [(0, "accent"), (1, "normal"), (2, "accent"), (3, "normal")],
            "clap": [(1, "accent"), (3, "accent")],
            "hh_closed": [(0, "normal"), (1, "normal"), (2, "normal"), (3, "normal")],
            "hh_open": [
                (0.5, "normal"),
                (1.5, "normal"),
                (2.5, "normal"),
                (3.5, "normal"),
            ],
        },
    ),
    DrumPattern(
        name="electronic_syncopated",
        genre_family="electronic",
        energy="medium",
        time_sig=(4, 4),
        description="Syncopated electronic — kick on 1 and 2.5, snare on 2 and 4, ghost hats",
        voices={
            "kick": [(0, "accent"), (2.5, "normal")],
            "snare": [(1, "normal"), (3, "accent")],
            "hh_closed": [
                (0, "ghost"),
                (0.5, "ghost"),
                (1, "ghost"),
                (1.5, "ghost"),
                (2, "ghost"),
                (2.5, "ghost"),
                (3, "ghost"),
                (3.5, "ghost"),
            ],
        },
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Krautrock / Motorik
# ---------------------------------------------------------------------------

TEMPLATES_4_4_KRAUTROCK = [
    DrumPattern(
        name="motorik",
        genre_family="krautrock",
        energy="medium",
        time_sig=(4, 4),
        description="Classic motorik — steady kick every beat, relentless eighth-note hats, snare on 3",
        voices={
            "kick": [(0, "accent"), (1, "normal"), (2, "normal"), (3, "normal")],
            "snare": [(2, "accent")],
            "hh_closed": [
                (0, "accent"),
                (0.5, "normal"),
                (1, "normal"),
                (1.5, "normal"),
                (2, "normal"),
                (2.5, "normal"),
                (3, "normal"),
                (3.5, "normal"),
            ],
        },
    ),
    DrumPattern(
        name="motorik_stripped",
        genre_family="krautrock",
        energy="low",
        time_sig=(4, 4),
        description="Stripped motorik — kick on 1 and 3, ghost hats on eighths, no snare",
        voices={
            "kick": [(0, "normal"), (2, "normal")],
            "hh_closed": [
                (0, "ghost"),
                (0.5, "ghost"),
                (1, "ghost"),
                (1.5, "ghost"),
                (2, "ghost"),
                (2.5, "ghost"),
                (3, "ghost"),
                (3.5, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="motorik_driving",
        genre_family="krautrock",
        energy="high",
        time_sig=(4, 4),
        description="Driving motorik — kick every beat accented, snare on 3, open hat on offbeats, crash on 1",
        voices={
            "kick": [(0, "accent"), (1, "accent"), (2, "accent"), (3, "accent")],
            "snare": [(2, "accent")],
            "hh_closed": [
                (0, "accent"),
                (0.5, "normal"),
                (1, "accent"),
                (1.5, "normal"),
                (2, "accent"),
                (2.5, "normal"),
                (3, "accent"),
                (3.5, "normal"),
            ],
            "hh_open": [(1.5, "ghost"), (3.5, "ghost")],
            "crash": [(0, "normal")],
        },
    ),
    DrumPattern(
        name="kosmische_pulse",
        genre_family="krautrock",
        energy="medium",
        time_sig=(4, 4),
        description="Kosmische pulse — hypnotic kick on 1 and 3, tom pattern on offbeats, ride instead of hat",
        voices={
            "kick": [(0, "accent"), (2, "normal")],
            "ride": [
                (0, "normal"),
                (0.5, "ghost"),
                (1, "normal"),
                (1.5, "ghost"),
                (2, "normal"),
                (2.5, "ghost"),
                (3, "normal"),
                (3.5, "ghost"),
            ],
            "tom_mid": [(1, "ghost"), (3, "ghost")],
        },
    ),
    DrumPattern(
        name="kosmische_build",
        genre_family="krautrock",
        energy="high",
        time_sig=(4, 4),
        description="Kosmische build — kick every beat, toms cascading, ride driving, snare on 3",
        voices={
            "kick": [(0, "accent"), (1, "normal"), (2, "accent"), (3, "normal")],
            "snare": [(2, "accent")],
            "ride": [
                (0, "accent"),
                (0.5, "normal"),
                (1, "normal"),
                (1.5, "normal"),
                (2, "accent"),
                (2.5, "normal"),
                (3, "normal"),
                (3.5, "normal"),
            ],
            "tom_high": [(0.5, "ghost"), (2.5, "ghost")],
            "tom_mid": [(1.5, "ghost"), (3.5, "ghost")],
        },
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Rock
# ---------------------------------------------------------------------------

TEMPLATES_4_4_ROCK = [
    DrumPattern(
        name="rock_halftime",
        genre_family="rock",
        energy="low",
        time_sig=(4, 4),
        description="Half-time rock — kick on 1, snare on 3, hats on quarters",
        voices={
            "kick": [(0, "accent")],
            "snare": [(2, "accent")],
            "hh_closed": [(0, "normal"), (1, "normal"), (2, "normal"), (3, "normal")],
        },
    ),
    DrumPattern(
        name="rock_basic",
        genre_family="rock",
        energy="medium",
        time_sig=(4, 4),
        description="Basic rock — kick on 1 and 3, snare on 2 and 4, hats on eighths",
        voices={
            "kick": [(0, "accent"), (2, "normal")],
            "snare": [(1, "accent"), (3, "accent")],
            "hh_closed": [
                (0, "normal"),
                (0.5, "ghost"),
                (1, "normal"),
                (1.5, "ghost"),
                (2, "normal"),
                (2.5, "ghost"),
                (3, "normal"),
                (3.5, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="rock_driving",
        genre_family="rock",
        energy="high",
        time_sig=(4, 4),
        description="Driving rock — kick on every beat, snare on 2 and 4, accented hats, crash on 1",
        voices={
            "kick": [(0, "accent"), (1, "normal"), (2, "accent"), (3, "normal")],
            "snare": [(1, "accent"), (3, "accent")],
            "hh_closed": [
                (0, "accent"),
                (0.5, "normal"),
                (1, "accent"),
                (1.5, "normal"),
                (2, "accent"),
                (2.5, "normal"),
                (3, "accent"),
                (3.5, "normal"),
            ],
            "crash": [(0, "accent")],
        },
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Classical
# ---------------------------------------------------------------------------

TEMPLATES_4_4_CLASSICAL = [
    DrumPattern(
        name="classical_minimal",
        genre_family="classical",
        energy="low",
        time_sig=(4, 4),
        description="Minimal classical — single timpani hit on 1",
        voices={
            "tom_low": [(0, "normal")],
        },
    ),
    DrumPattern(
        name="classical_march",
        genre_family="classical",
        energy="medium",
        time_sig=(4, 4),
        description="March feel — timpani on 1 and 3, snare roll on 2 and 4",
        voices={
            "tom_low": [(0, "accent"), (2, "normal")],
            "snare": [(1, "normal"), (3, "normal")],
        },
    ),
    DrumPattern(
        name="classical_full",
        genre_family="classical",
        energy="high",
        time_sig=(4, 4),
        description="Full orchestral pulse — timpani on every beat, crash on 1",
        voices={
            "tom_low": [(0, "accent"), (1, "normal"), (2, "accent"), (3, "normal")],
            "crash": [(0, "accent")],
            "snare": [(1, "accent"), (3, "accent")],
        },
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Experimental
# ---------------------------------------------------------------------------

TEMPLATES_4_4_EXPERIMENTAL = [
    DrumPattern(
        name="experimental_rimshot",
        genre_family="experimental",
        energy="low",
        time_sig=(4, 4),
        description="Rimshot texture — ghost rimshots on offbeats, no kick",
        voices={
            "rimshot": [(0.5, "ghost"), (1.5, "ghost"), (2.5, "ghost"), (3.5, "ghost")],
        },
    ),
    DrumPattern(
        name="experimental_stutter",
        genre_family="experimental",
        energy="medium",
        time_sig=(4, 4),
        description="Stuttered pattern — irregular kick and snare placement, ghost hats",
        voices={
            "kick": [(0, "accent"), (1.5, "normal"), (3, "ghost")],
            "snare": [(0.5, "ghost"), (2, "normal")],
            "hh_closed": [(0, "ghost"), (1, "ghost"), (2, "ghost"), (3, "ghost")],
        },
    ),
    DrumPattern(
        name="experimental_industrial",
        genre_family="experimental",
        energy="high",
        time_sig=(4, 4),
        description="Industrial pound — accented kick every beat, clap on 2 and 4, tom fills",
        voices={
            "kick": [(0, "accent"), (1, "accent"), (2, "accent"), (3, "accent")],
            "clap": [(1, "accent"), (3, "accent")],
            "tom_high": [(0.5, "normal"), (2.5, "normal")],
            "tom_low": [(1.5, "ghost"), (3.5, "ghost")],
        },
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Folk
# ---------------------------------------------------------------------------

TEMPLATES_4_4_FOLK = [
    DrumPattern(
        name="folk_brushes",
        genre_family="folk",
        energy="low",
        time_sig=(4, 4),
        description="Brush feel — ghost snare swishes on quarters, kick on 1",
        voices={
            "kick": [(0, "ghost")],
            "snare": [(0, "ghost"), (1, "ghost"), (2, "ghost"), (3, "ghost")],
        },
    ),
    DrumPattern(
        name="folk_simple",
        genre_family="folk",
        energy="medium",
        time_sig=(4, 4),
        description="Simple folk — kick on 1 and 3, snare on 2 and 4, no hats",
        voices={
            "kick": [(0, "normal"), (2, "normal")],
            "snare": [(1, "normal"), (3, "normal")],
        },
    ),
    DrumPattern(
        name="folk_driving",
        genre_family="folk",
        energy="high",
        time_sig=(4, 4),
        description="Driving folk — kick on 1 and 3, snare on 2 and 4, tambourine-style hats on eighths",
        voices={
            "kick": [(0, "accent"), (2, "normal")],
            "snare": [(1, "accent"), (3, "accent")],
            "hh_closed": [
                (0, "ghost"),
                (0.5, "ghost"),
                (1, "ghost"),
                (1.5, "ghost"),
                (2, "ghost"),
                (2.5, "ghost"),
                (3, "ghost"),
                (3.5, "ghost"),
            ],
        },
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Jazz
# ---------------------------------------------------------------------------

TEMPLATES_4_4_JAZZ = [
    DrumPattern(
        name="jazz_brushes",
        genre_family="jazz",
        energy="low",
        time_sig=(4, 4),
        description="Jazz brushes — ride on quarters, ghost kick on 1, cross-stick on 4",
        voices={
            "kick": [(0, "ghost")],
            "rimshot": [(3, "ghost")],
            "ride": [(0, "normal"), (1, "ghost"), (2, "normal"), (3, "ghost")],
        },
    ),
    DrumPattern(
        name="jazz_swing",
        genre_family="jazz",
        energy="medium",
        time_sig=(4, 4),
        description="Swing ride — ride on quarters with ghost triplet skip, kick on 1 and 3, hi-hat pedal on 2 and 4",
        voices={
            "kick": [(0, "normal"), (2, "ghost")],
            "hh_pedal": [(1, "normal"), (3, "normal")],
            "ride": [
                (0, "accent"),
                (1, "normal"),
                (2, "accent"),
                (3, "normal"),
            ],
        },
    ),
    DrumPattern(
        name="jazz_driving",
        genre_family="jazz",
        energy="high",
        time_sig=(4, 4),
        description="Driving jazz — ride on eighths, snare comping, kick on 1 and 3",
        voices={
            "kick": [(0, "accent"), (2, "normal")],
            "snare": [(1, "ghost"), (2.5, "ghost"), (3.5, "normal")],
            "ride": [
                (0, "accent"),
                (0.5, "normal"),
                (1, "normal"),
                (1.5, "normal"),
                (2, "accent"),
                (2.5, "normal"),
                (3, "normal"),
                (3.5, "normal"),
            ],
            "hh_pedal": [(1, "normal"), (3, "normal")],
        },
    ),
]

# ---------------------------------------------------------------------------
# 7/8 Templates
# ---------------------------------------------------------------------------
# In 7/8, the bar has 7 eighth notes = 3.5 quarter-note beats.
# Positions are in quarter-note beats: 0, 0.5, 1, 1.5, 2, 2.5, 3

TEMPLATES_7_8_AMBIENT = [
    DrumPattern(
        name="ambient_7_sparse",
        genre_family="ambient",
        energy="low",
        time_sig=(7, 8),
        description="Sparse 7/8 — kick on 1, ghost hat on beat 2.5",
        voices={
            "kick": [(0, "normal")],
            "hh_closed": [(2, "ghost")],
        },
    ),
    DrumPattern(
        name="ambient_7_pulse",
        genre_family="ambient",
        energy="medium",
        time_sig=(7, 8),
        description="7/8 ambient pulse — kick on 1 and 2.5, ghost hats on all eighths",
        voices={
            "kick": [(0, "normal"), (2, "ghost")],
            "hh_closed": [
                (0, "ghost"),
                (0.5, "ghost"),
                (1, "ghost"),
                (1.5, "ghost"),
                (2, "ghost"),
                (2.5, "ghost"),
                (3, "ghost"),
            ],
        },
    ),
]

TEMPLATES_7_8_ELECTRONIC = [
    DrumPattern(
        name="electronic_7_322",
        genre_family="electronic",
        energy="medium",
        time_sig=(7, 8),
        description="7/8 electronic 3+2+2 — kick on group starts, clap on 2nd and 3rd groups",
        voices={
            "kick": [(0, "accent"), (1.5, "normal"), (2.5, "normal")],
            "clap": [(1.5, "normal"), (2.5, "normal")],
            "hh_closed": [
                (0, "normal"),
                (0.5, "ghost"),
                (1, "ghost"),
                (1.5, "normal"),
                (2, "ghost"),
                (2.5, "normal"),
                (3, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="electronic_7_223",
        genre_family="electronic",
        energy="medium",
        time_sig=(7, 8),
        description="7/8 electronic 2+2+3 — kick on group starts, clap on group 3",
        voices={
            "kick": [(0, "accent"), (1, "normal"), (2, "normal")],
            "clap": [(2, "normal")],
            "hh_closed": [
                (0, "normal"),
                (0.5, "ghost"),
                (1, "normal"),
                (1.5, "ghost"),
                (2, "normal"),
                (2.5, "ghost"),
                (3, "ghost"),
            ],
        },
    ),
]

TEMPLATES_7_8_KRAUTROCK = [
    DrumPattern(
        name="motorik_7_322",
        genre_family="krautrock",
        energy="medium",
        time_sig=(7, 8),
        description="Motorik 7/8 3+2+2 — relentless eighth hats, kick on group starts, snare on group 2",
        voices={
            "kick": [(0, "accent"), (1.5, "normal"), (2.5, "normal")],
            "snare": [(1.5, "accent")],
            "hh_closed": [
                (0, "accent"),
                (0.5, "normal"),
                (1, "normal"),
                (1.5, "normal"),
                (2, "normal"),
                (2.5, "normal"),
                (3, "normal"),
            ],
        },
    ),
    DrumPattern(
        name="motorik_7_driving",
        genre_family="krautrock",
        energy="high",
        time_sig=(7, 8),
        description="Driving motorik 7/8 — kick on every eighth-note group start, snare on 2nd group, accented hats",
        voices={
            "kick": [(0, "accent"), (1.5, "accent"), (2.5, "accent")],
            "snare": [(1.5, "accent")],
            "hh_closed": [
                (0, "accent"),
                (0.5, "normal"),
                (1, "normal"),
                (1.5, "accent"),
                (2, "normal"),
                (2.5, "accent"),
                (3, "normal"),
            ],
            "crash": [(0, "normal")],
        },
    ),
]

TEMPLATES_7_8_ROCK = [
    DrumPattern(
        name="rock_7_322",
        genre_family="rock",
        energy="medium",
        time_sig=(7, 8),
        description="Rock 7/8 3+2+2 — kick on 1 and group 3, snare on group 2, hats on eighths",
        voices={
            "kick": [(0, "accent"), (2.5, "normal")],
            "snare": [(1.5, "accent")],
            "hh_closed": [
                (0, "normal"),
                (0.5, "ghost"),
                (1, "ghost"),
                (1.5, "normal"),
                (2, "ghost"),
                (2.5, "normal"),
                (3, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="rock_7_driving",
        genre_family="rock",
        energy="high",
        time_sig=(7, 8),
        description="Driving rock 7/8 — kick accented on group starts, snare on 2, crash on 1",
        voices={
            "kick": [(0, "accent"), (1.5, "normal"), (2.5, "accent")],
            "snare": [(1.5, "accent")],
            "hh_closed": [
                (0, "accent"),
                (0.5, "normal"),
                (1, "normal"),
                (1.5, "accent"),
                (2, "normal"),
                (2.5, "accent"),
                (3, "normal"),
            ],
            "crash": [(0, "accent")],
        },
    ),
]

# ---------------------------------------------------------------------------
# Fallback template (any time signature)
# ---------------------------------------------------------------------------


def make_fallback_pattern(time_sig: tuple[int, int]) -> DrumPattern:
    """Generate a minimal fallback pattern for unsupported time signatures.

    Kick on beat 1, that's it.
    """
    num, den = time_sig
    return DrumPattern(
        name="fallback_minimal",
        genre_family="fallback",
        energy="medium",
        time_sig=time_sig,
        description=f"Fallback minimal — kick on 1 for {num}/{den}",
        voices={
            "kick": [(0, "normal")],
        },
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 3/4 Americana
# ---------------------------------------------------------------------------

TEMPLATES_3_4_AMERICANA = [
    DrumPattern(
        name="americana_waltz_brushes",
        genre_family="americana",
        energy="low",
        time_sig=(3, 4),
        description="Waltz brushes — ghost snare swishes on all three beats, kick on 1",
        voices={
            "kick": [(0, "normal")],
            "snare": [(0, "ghost"), (1, "ghost"), (2, "ghost")],
        },
    ),
    DrumPattern(
        name="americana_waltz",
        genre_family="americana",
        energy="medium",
        time_sig=(3, 4),
        description="Country waltz — kick on 1, snare on 2 and 3, closed hat on eighths",
        voices={
            "kick": [(0, "accent")],
            "snare": [(1, "normal"), (2, "normal")],
            "hh_closed": [
                (0, "ghost"),
                (0.5, "ghost"),
                (1, "ghost"),
                (1.5, "ghost"),
                (2, "ghost"),
                (2.5, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="americana_waltz_driving",
        genre_family="americana",
        energy="high",
        time_sig=(3, 4),
        description="Driving waltz — accented kick on 1, snare accent on 3, hi-hat eighths with open on 2",
        voices={
            "kick": [(0, "accent"), (1.5, "ghost")],
            "snare": [(1, "normal"), (2, "accent")],
            "hh_closed": [
                (0, "normal"),
                (0.5, "ghost"),
                (1.5, "ghost"),
                (2, "normal"),
                (2.5, "ghost"),
            ],
            "hh_open": [(1, "normal")],
        },
    ),
]

# ---------------------------------------------------------------------------
# 3/4 Experimental
# ---------------------------------------------------------------------------

TEMPLATES_3_4_EXPERIMENTAL = [
    DrumPattern(
        name="experimental_3_4_sparse",
        genre_family="experimental",
        energy="low",
        time_sig=(3, 4),
        description="Sparse 3/4 — kick on 1, rimshot ghost on beat 2.5, silence elsewhere",
        voices={
            "kick": [(0, "normal")],
            "rimshot": [(2.5, "ghost")],
        },
    ),
    DrumPattern(
        name="experimental_3_4_industrial",
        genre_family="experimental",
        energy="medium",
        time_sig=(3, 4),
        description="Industrial 3/4 — accented kick on 1, clap on 2, tom low ghost on 2.5 and 3",
        voices={
            "kick": [(0, "accent")],
            "clap": [(1, "accent")],
            "tom_low": [(1.5, "ghost"), (2, "ghost")],
            "hh_closed": [(0.5, "ghost"), (2.5, "ghost")],
        },
    ),
    DrumPattern(
        name="experimental_3_4_polyrhythm",
        genre_family="experimental",
        energy="high",
        time_sig=(3, 4),
        description="Polyrhythmic 3/4 — kick on 1 and 1.5 (hemiola feel), snare ghost on 2, clap on 3",
        voices={
            "kick": [(0, "accent"), (1.5, "normal")],
            "snare": [(1, "ghost")],
            "clap": [(2, "accent")],
            "tom_high": [(0.5, "ghost"), (2.5, "ghost")],
        },
    ),
]

# ---------------------------------------------------------------------------
# 4/4 Breakbeat (classic breaks + energy variants)
# ---------------------------------------------------------------------------
# Grid mapping: 16th-note positions in quarter-beat floats
#   1=0.0  1e=0.25  1+=0.5  1a=0.75
#   2=1.0  2e=1.25  2+=1.5  2a=1.75
#   3=2.0  3e=2.25  3+=2.5  3a=2.75
#   4=3.0  4e=3.25  4+=3.5  4a=3.75

TEMPLATES_4_4_BREAKBEAT = [
    # -------------------------------------------------------------------
    # 9 classic medium-energy breaks (transcribed from reference patterns)
    # -------------------------------------------------------------------
    DrumPattern(
        name="billie_jean",
        genre_family="breakbeat",
        energy="medium",
        time_sig=(4, 4),
        description="Billie Jean — syncopated kick on 1/1a/2+/3/3a/4+, snare on 2 and 4, 8th-note hats",
        voices={
            "kick": [
                (0.0, "accent"),
                (0.75, "normal"),
                (1.5, "normal"),
                (2.0, "accent"),
                (2.75, "normal"),
                (3.5, "normal"),
            ],
            "snare": [(1.0, "accent"), (3.0, "accent")],
            "hh_closed": [
                (0.0, "normal"),
                (0.5, "ghost"),
                (1.0, "normal"),
                (1.5, "ghost"),
                (2.0, "normal"),
                (2.5, "ghost"),
                (3.0, "normal"),
                (3.5, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="funky_drummer",
        genre_family="breakbeat",
        energy="medium",
        time_sig=(4, 4),
        description="The Funky Drummer — Clyde Stubblefield ghost-note snare complexity, 16th hats",
        voices={
            "kick": [
                (0.0, "accent"),
                (0.75, "normal"),
                (2.0, "normal"),
                (2.75, "normal"),
            ],
            "snare": [
                (1.0, "accent"),
                (1.75, "ghost"),
                (2.5, "ghost"),
                (3.0, "accent"),
                (3.25, "ghost"),
            ],
            "hh_closed": [
                (0.0, "normal"),
                (0.25, "ghost"),
                (0.5, "normal"),
                (0.75, "ghost"),
                (1.0, "normal"),
                (1.25, "ghost"),
                (1.5, "normal"),
                (1.75, "ghost"),
                (2.0, "normal"),
                (2.25, "ghost"),
                (2.5, "normal"),
                (2.75, "ghost"),
                (3.0, "normal"),
                (3.25, "ghost"),
                (3.5, "normal"),
                (3.75, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="impeach_the_president",
        genre_family="breakbeat",
        energy="medium",
        time_sig=(4, 4),
        description="Impeach The President — syncopated kick on 1/1+/3/4+, open hat on offbeats",
        voices={
            "kick": [
                (0.0, "accent"),
                (0.5, "normal"),
                (2.0, "normal"),
                (3.5, "normal"),
            ],
            "snare": [(1.0, "accent"), (3.0, "accent")],
            "hh_closed": [
                (0.0, "normal"),
                (0.5, "ghost"),
                (1.0, "normal"),
                (1.5, "ghost"),
                (2.0, "normal"),
                (2.5, "ghost"),
                (3.0, "normal"),
                (3.5, "ghost"),
            ],
            "hh_open": [(1.5, "normal"), (3.5, "normal")],
        },
    ),
    DrumPattern(
        name="when_the_levee_breaks",
        genre_family="breakbeat",
        energy="medium",
        time_sig=(4, 4),
        description="When The Levee Breaks — half-time thunder, dense kick, snare on beat 3, ghost hats",
        voices={
            "kick": [
                (0.0, "accent"),
                (0.5, "normal"),
                (0.75, "normal"),
                (2.75, "normal"),
            ],
            "snare": [(2.0, "accent")],
            "hh_closed": [
                (0.0, "ghost"),
                (0.5, "ghost"),
                (1.0, "ghost"),
                (1.5, "ghost"),
                (2.0, "ghost"),
                (2.5, "ghost"),
                (3.0, "ghost"),
                (3.5, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="walk_this_way",
        genre_family="breakbeat",
        energy="medium",
        time_sig=(4, 4),
        description="Walk This Way — busy syncopated kick, snare on 2 and 4, open hat on offbeats",
        voices={
            "kick": [
                (0.0, "accent"),
                (0.75, "normal"),
                (1.75, "normal"),
                (2.0, "accent"),
                (2.75, "normal"),
                (3.75, "normal"),
            ],
            "snare": [(1.0, "accent"), (3.0, "accent")],
            "hh_closed": [
                (0.0, "normal"),
                (0.5, "ghost"),
                (1.0, "normal"),
                (1.5, "ghost"),
                (2.0, "normal"),
                (2.5, "ghost"),
                (3.0, "normal"),
                (3.5, "ghost"),
            ],
            "hh_open": [(1.5, "ghost"), (3.5, "ghost")],
        },
    ),
    DrumPattern(
        name="its_a_new_day",
        genre_family="breakbeat",
        energy="medium",
        time_sig=(4, 4),
        description="It's A New Day — James Brown kick on 1/1a/3/3a, snare on 2 and 4",
        voices={
            "kick": [
                (0.0, "accent"),
                (0.75, "normal"),
                (2.0, "accent"),
                (2.75, "normal"),
            ],
            "snare": [(1.0, "accent"), (3.0, "accent")],
            "hh_closed": [
                (0.0, "normal"),
                (0.5, "ghost"),
                (1.0, "normal"),
                (1.5, "ghost"),
                (2.0, "normal"),
                (2.5, "ghost"),
                (3.0, "normal"),
                (3.5, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="papa_was_too",
        genre_family="breakbeat",
        energy="medium",
        time_sig=(4, 4),
        description="Papa Was Too — kick 1/2+/3/4+, snare with ghost on 2 and 4, offbeat tambourine",
        voices={
            "kick": [
                (0.0, "accent"),
                (1.5, "normal"),
                (2.0, "normal"),
                (3.5, "normal"),
            ],
            "snare": [(1.0, "accent"), (2.75, "ghost"), (3.0, "accent")],
            "hh_closed": [
                (0.0, "normal"),
                (0.5, "ghost"),
                (1.0, "normal"),
                (1.5, "ghost"),
                (2.0, "normal"),
                (2.5, "ghost"),
                (3.0, "normal"),
                (3.5, "ghost"),
            ],
            "tambourine": [
                (0.25, "ghost"),
                (0.75, "ghost"),
                (1.25, "ghost"),
                (1.75, "ghost"),
                (2.25, "ghost"),
                (2.75, "ghost"),
                (3.25, "ghost"),
                (3.75, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="the_big_beat",
        genre_family="breakbeat",
        energy="medium",
        time_sig=(4, 4),
        description="The Big Beat — kick on 1/2+/3/4+, snare and clap on 2 and 4",
        voices={
            "kick": [
                (0.0, "accent"),
                (1.5, "normal"),
                (2.0, "normal"),
                (3.5, "normal"),
            ],
            "snare": [(1.0, "accent"), (3.0, "accent")],
            "clap": [(1.0, "normal"), (3.0, "accent")],
        },
    ),
    DrumPattern(
        name="ashleys_roachclip",
        genre_family="breakbeat",
        energy="medium",
        time_sig=(4, 4),
        description="Ashley's Roachclip — kick/snare backbone with open hat and conga syncopation",
        voices={
            "kick": [(0.0, "accent"), (1.5, "normal"), (2.5, "normal")],
            "snare": [(1.0, "accent"), (3.0, "accent")],
            "hh_closed": [
                (0.0, "normal"),
                (0.5, "ghost"),
                (1.0, "ghost"),
                (1.5, "ghost"),
                (2.0, "ghost"),
                (2.5, "ghost"),
                (3.0, "ghost"),
                (3.5, "ghost"),
            ],
            "hh_open": [(1.5, "normal"), (2.5, "normal")],
            "conga_high": [(0.25, "normal"), (1.75, "normal"), (2.75, "normal")],
            "conga_low": [(0.75, "ghost"), (1.25, "ghost"), (3.25, "ghost")],
        },
    ),
    # -------------------------------------------------------------------
    # Low-energy variants
    # -------------------------------------------------------------------
    DrumPattern(
        name="breakbeat_sparse",
        genre_family="breakbeat",
        energy="low",
        time_sig=(4, 4),
        description="Sparse boom-bap skeleton — kick on 1, snare on 3, ghost hat pulse",
        voices={
            "kick": [(0.0, "accent")],
            "snare": [(2.0, "accent")],
            "hh_closed": [
                (0.0, "ghost"),
                (0.5, "ghost"),
                (1.0, "ghost"),
                (1.5, "ghost"),
                (2.0, "ghost"),
                (2.5, "ghost"),
                (3.0, "ghost"),
                (3.5, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="funky_drummer_stripped",
        genre_family="breakbeat",
        energy="low",
        time_sig=(4, 4),
        description="Funky Drummer stripped — kick/snare ghost skeleton, no hats",
        voices={
            "kick": [(0.0, "accent"), (0.75, "normal"), (2.0, "normal")],
            "snare": [
                (1.0, "accent"),
                (1.75, "ghost"),
                (2.5, "ghost"),
                (3.0, "accent"),
                (3.25, "ghost"),
            ],
        },
    ),
    DrumPattern(
        name="when_the_levee_breaks_ghost",
        genre_family="breakbeat",
        energy="low",
        time_sig=(4, 4),
        description="Levee feel at low volume — ghosted kick cluster, quiet snare on 3",
        voices={
            "kick": [(0.0, "ghost"), (0.5, "ghost"), (0.75, "ghost"), (2.75, "ghost")],
            "snare": [(2.0, "normal")],
            "hh_closed": [
                (0.0, "ghost"),
                (0.5, "ghost"),
                (1.0, "ghost"),
                (1.5, "ghost"),
                (2.0, "ghost"),
                (2.5, "ghost"),
                (3.0, "ghost"),
                (3.5, "ghost"),
            ],
        },
    ),
    # -------------------------------------------------------------------
    # High-energy variants
    # -------------------------------------------------------------------
    DrumPattern(
        name="billie_jean_full",
        genre_family="breakbeat",
        energy="high",
        time_sig=(4, 4),
        description="Billie Jean full — all kicks accented, snare accented, crash on 1",
        voices={
            "kick": [
                (0.0, "accent"),
                (0.75, "accent"),
                (1.5, "accent"),
                (2.0, "accent"),
                (2.75, "accent"),
                (3.5, "accent"),
            ],
            "snare": [(1.0, "accent"), (3.0, "accent")],
            "hh_closed": [
                (0.0, "accent"),
                (0.5, "normal"),
                (1.0, "accent"),
                (1.5, "normal"),
                (2.0, "accent"),
                (2.5, "normal"),
                (3.0, "accent"),
                (3.5, "normal"),
            ],
            "crash": [(0.0, "normal")],
        },
    ),
    DrumPattern(
        name="funky_drummer_heavy",
        genre_family="breakbeat",
        energy="high",
        time_sig=(4, 4),
        description="Funky Drummer heavy — cowbell on offbeats, accented snares, driving 16th hats",
        voices={
            "kick": [
                (0.0, "accent"),
                (0.75, "accent"),
                (2.0, "accent"),
                (2.75, "accent"),
            ],
            "snare": [
                (1.0, "accent"),
                (1.75, "ghost"),
                (2.5, "ghost"),
                (3.0, "accent"),
                (3.25, "ghost"),
            ],
            "hh_closed": [
                (0.0, "accent"),
                (0.25, "ghost"),
                (0.5, "normal"),
                (0.75, "ghost"),
                (1.0, "accent"),
                (1.25, "ghost"),
                (1.5, "normal"),
                (1.75, "ghost"),
                (2.0, "accent"),
                (2.25, "ghost"),
                (2.5, "normal"),
                (2.75, "ghost"),
                (3.0, "accent"),
                (3.25, "ghost"),
                (3.5, "normal"),
                (3.75, "ghost"),
            ],
            "cowbell": [
                (0.5, "normal"),
                (1.5, "normal"),
                (2.5, "normal"),
                (3.5, "normal"),
            ],
        },
    ),
    DrumPattern(
        name="breakbeat_driving",
        genre_family="breakbeat",
        energy="high",
        time_sig=(4, 4),
        description="Driving breakbeat — four-on-floor kick with breakbeat snare and open hat offbeats",
        voices={
            "kick": [
                (0.0, "accent"),
                (1.0, "accent"),
                (2.0, "accent"),
                (3.0, "accent"),
            ],
            "snare": [(1.0, "accent"), (3.0, "accent")],
            "hh_closed": [
                (0.0, "accent"),
                (0.25, "ghost"),
                (0.5, "normal"),
                (0.75, "ghost"),
                (1.0, "accent"),
                (1.25, "ghost"),
                (1.5, "normal"),
                (1.75, "ghost"),
                (2.0, "accent"),
                (2.25, "ghost"),
                (2.5, "normal"),
                (2.75, "ghost"),
                (3.0, "accent"),
                (3.25, "ghost"),
                (3.5, "normal"),
                (3.75, "ghost"),
            ],
            "hh_open": [
                (0.5, "normal"),
                (1.5, "normal"),
                (2.5, "normal"),
                (3.5, "normal"),
            ],
        },
    ),
]


# ---------------------------------------------------------------------------
# All templates registry
# ---------------------------------------------------------------------------

ALL_TEMPLATES: list[DrumPattern] = [
    # 4/4
    *TEMPLATES_4_4_AMBIENT,
    *TEMPLATES_4_4_ELECTRONIC,
    *TEMPLATES_4_4_KRAUTROCK,
    *TEMPLATES_4_4_ROCK,
    *TEMPLATES_4_4_CLASSICAL,
    *TEMPLATES_4_4_EXPERIMENTAL,
    *TEMPLATES_4_4_FOLK,
    *TEMPLATES_4_4_JAZZ,
    # 7/8
    *TEMPLATES_7_8_AMBIENT,
    *TEMPLATES_7_8_ELECTRONIC,
    *TEMPLATES_7_8_KRAUTROCK,
    *TEMPLATES_7_8_ROCK,
    # 3/4
    *TEMPLATES_3_4_AMERICANA,
    *TEMPLATES_3_4_EXPERIMENTAL,
    # 4/4 Breakbeat
    *TEMPLATES_4_4_BREAKBEAT,
]
