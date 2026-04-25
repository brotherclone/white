"""Pydantic models for composition_narrative.yml.

The composition narrative is a multi-dimensional description of a song's
intended shape, written by Claude as a composer before any loop generation.
Each section carries four controlled-vocabulary dimensions plus a free-text
narrative paragraph.

Controlled vocabularies:
  register:             low | low_mid | mid | mid_high | high
  texture:              absent | near_absent | sparse | moderate | full
  harmonic_complexity:  simple | moderate | tense | rich
  rhythm_character:     absent | minimal | present | busy | open

These are intentionally small so the pipeline can map them to concrete
generation choices without ambiguity.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Controlled vocabulary types
# ---------------------------------------------------------------------------

Register = Literal["low", "low_mid", "mid", "mid_high", "high"]
Texture = Literal["absent", "near_absent", "sparse", "moderate", "full"]
HarmonicComplexity = Literal["simple", "moderate", "tense", "rich"]
RhythmCharacter = Literal["absent", "minimal", "present", "busy", "open"]
LeadVoice = Literal["bass", "melody", "none", "drums", "chords"]


# ---------------------------------------------------------------------------
# Section model
# ---------------------------------------------------------------------------


class NarrativeSection(BaseModel):
    """Per-section narrative constraints.

    All dimension fields are optional so that partially-written narratives
    can be loaded without error. Missing fields produce no constraint.
    """

    arc: Optional[float] = None
    register: Optional[Register] = None
    texture: Optional[Texture] = None
    harmonic_complexity: Optional[HarmonicComplexity] = None
    rhythm_character: Optional[RhythmCharacter] = None
    lead_voice: Optional[LeadVoice] = None
    contrast_with: Optional[str] = None
    is_the_moment: bool = False
    narrative: str = ""

    model_config = {"extra": "allow"}  # allow future fields without errors


# ---------------------------------------------------------------------------
# Top-level document
# ---------------------------------------------------------------------------


class TheMoment(BaseModel):
    section: str
    description: str = ""


class CompositionNarrative(BaseModel):
    """Full composition_narrative.yml document."""

    schema_version: str = "1"
    generated_by: str = "claude"
    rationale: str = ""
    the_moment: Optional[TheMoment] = None
    sections: dict[str, NarrativeSection] = {}

    model_config = {"extra": "allow"}

    @classmethod
    def from_dict(cls, data: dict) -> "CompositionNarrative":
        """Parse a raw YAML dict into a CompositionNarrative.

        Section entries may use arbitrary section names as keys.
        """
        sections_raw = data.get("sections", {})
        sections = {
            k: NarrativeSection(**v) if isinstance(v, dict) else NarrativeSection()
            for k, v in sections_raw.items()
        }
        moment_raw = data.get("the_moment")
        the_moment = TheMoment(**moment_raw) if isinstance(moment_raw, dict) else None
        return cls(
            schema_version=data.get("schema_version", "1"),
            generated_by=data.get("generated_by", "claude"),
            rationale=data.get("rationale", ""),
            the_moment=the_moment,
            sections=sections,
        )


# ---------------------------------------------------------------------------
# Constraint extraction
# ---------------------------------------------------------------------------

# Tag lists for each rhythm_character value (drums)
_RHYTHM_PREFER: dict[str, list[str]] = {
    "absent": ["ambient", "ambient_pulse", "sparse"],
    "minimal": ["ghost_only", "sparse", "half_time", "brushed"],
    "present": [],  # no bonus/penalty
    "busy": ["dense", "electronic", "motorik"],
    "open": ["half_time"],  # open hi-hat proxy
}

_RHYTHM_PENALISE: dict[str, list[str]] = {
    "absent": ["dense", "electronic", "motorik"],
    "minimal": ["dense", "electronic"],
    "present": [],
    "busy": ["sparse", "ghost_only", "ambient"],
    "open": [],
}

# Tag lists for texture (bass)
_TEXTURE_BASS_PREFER: dict[str, list[str]] = {
    "absent": ["minimal", "drone", "pedal"],
    "near_absent": ["minimal", "drone"],
    "sparse": ["pedal", "drone"],
    "moderate": [],
    "full": ["walking", "arpeggiated"],
}

_TEXTURE_BASS_PENALISE: dict[str, list[str]] = {
    "absent": ["walking", "arpeggiated"],
    "near_absent": ["walking"],
    "sparse": [],
    "moderate": [],
    "full": [],
}

# Lead voice → bass tag preference
_LEAD_VOICE_BASS_PREFER: dict[str, list[str]] = {
    "bass": ["walking", "arpeggiated"],
    "melody": ["pedal", "drone", "minimal"],
    "none": ["drone", "minimal"],
    "drums": [],
    "chords": [],
}

_LEAD_VOICE_BASS_PENALISE: dict[str, list[str]] = {
    "bass": [],
    "melody": ["walking", "arpeggiated"],
    "none": ["walking", "arpeggiated"],
    "drums": [],
    "chords": [],
}

# Register → melody tag preference
_REGISTER_MELODY_PREFER: dict[str, list[str]] = {
    "low": ["descent", "sparse", "stepwise"],
    "low_mid": ["descent", "stepwise"],
    "mid": [],
    "mid_high": ["wide_interval", "arpeggiated"],
    "high": ["wide_interval", "arpeggiated", "dense"],
}


def extract_constraints(section_name: str, narrative: "CompositionNarrative") -> dict:
    """Return a constraint dict for a named section from the narrative.

    Keys in the returned dict:
        rhythm_prefer        list[str] — drum tags to boost
        rhythm_penalise      list[str] — drum tags to penalise
        bass_prefer          list[str] — bass tags to boost
        bass_penalise        list[str] — bass tags to penalise
        melody_prefer        list[str] — melody tags to boost
        skip_melody          bool      — True when lead_voice == "none"
    """
    key = section_name.lower().replace("-", "_").replace(" ", "_")
    sec = narrative.sections.get(key)
    if sec is None:
        return {}

    result: dict = {}

    # Drums
    rc = sec.rhythm_character
    if rc:
        result["rhythm_prefer"] = _RHYTHM_PREFER.get(rc, [])
        result["rhythm_penalise"] = _RHYTHM_PENALISE.get(rc, [])

    # Bass
    tex = sec.texture
    lv = sec.lead_voice
    bp = list(_TEXTURE_BASS_PREFER.get(tex, [])) if tex else []
    bn = list(_TEXTURE_BASS_PENALISE.get(tex, [])) if tex else []
    if lv:
        bp.extend(_LEAD_VOICE_BASS_PREFER.get(lv, []))
        bn.extend(_LEAD_VOICE_BASS_PENALISE.get(lv, []))
    if bp:
        result["bass_prefer"] = bp
    if bn:
        result["bass_penalise"] = bn

    # Melody
    if lv == "none":
        result["skip_melody"] = True
    reg = sec.register
    if reg:
        prefer = _REGISTER_MELODY_PREFER.get(reg, [])
        if prefer:
            result["melody_prefer"] = prefer

    return result


def narrative_tag_adjustment(
    constraints: dict,
    pattern_tags: list[str],
    instrument: str,
) -> float:
    """Return a score adjustment for a pattern given narrative constraints.

    instrument: "drums", "bass", or "melody"
    Returns a float (typically ±0.05–0.10).
    """
    if not constraints or not pattern_tags:
        return 0.0

    tags = set(pattern_tags)
    adj = 0.0

    if instrument == "drums":
        prefer = set(constraints.get("rhythm_prefer", []))
        penalise = set(constraints.get("rhythm_penalise", []))
        if tags & prefer:
            adj += 0.10
        if tags & penalise:
            adj -= 0.05

    elif instrument == "bass":
        prefer = set(constraints.get("bass_prefer", []))
        penalise = set(constraints.get("bass_penalise", []))
        if tags & prefer:
            adj += 0.10
        if tags & penalise:
            adj -= 0.05

    elif instrument == "melody":
        prefer = set(constraints.get("melody_prefer", []))
        if tags & prefer:
            adj += 0.10

    return round(adj, 3)
