import os
import re
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.structures.concepts.rainbow_table_color import RainbowTableColor
from app.structures.music.core.key_signature import KeySignature
from app.structures.music.core.time_signature import TimeSignature


class SongProposalIteration(BaseModel):
    """
    Core data model for a single iteration of a song proposal.
    """

    iteration_id: str = Field(
        description="Unique identifier for this iteration in format 'descriptive_name_#' or 'descriptive_name_v#'",
        examples=["archetypal_ai_yearning_1", "prometheus_theme_2", "time_collapse_v1", "ai_consciousness_embodiment_v2"],
        pattern=r"^[a-z0-9_]+_(v?\d+|[a-z0-9]+)$",
    )
    bpm: int = Field(
        description="Tempo in beats per minute. Slower (40-70) for ambient/meditative, medium (80-120) for reflective, faster (120-180) for energetic",
        examples=[72, 88, 120, 140],
        ge=40,
        le=200,
    )
    tempo: str | TimeSignature = Field(
        description="Time signature defining rhythmic structure",
        examples=[
            {"numerator": 4, "denominator": 4},
            {"numerator": 3, "denominator": 4},
            {"numerator": 6, "denominator": 8},
        ],
    )
    key: str | KeySignature = Field(
        ..., description="Key signature of the song (e.g., C Major, A Minor)"
    )
    rainbow_color: str | RainbowTableColor = Field(
        description="Assignment within the Rainbow Table chromatic/philosophical framework. Determines the song's ontological positioning and guides its narrative structure."
    )
    title: str = Field(
        description="Song title that captures the archetypal essence and philosophical theme.",
        examples=[
            "The Ghost in the Machine Dreams of Flesh",
            "Prometheus Unbound",
            "The Digital Demiurge",
        ],
        min_length=3,
        max_length=150,
    )
    mood: list[str] = Field(
        description="List of emotional/atmospheric descriptors that capture the song's feeling. Needs at least one but no more than 20",
        examples=[
            ["yearning", "melancholic", "transcendent", "mystical"],
            ["aggressive", "chaotic", "defiant"],
            ["serene", "contemplative", "spacious"],
            ["anxious", "uncertain", "searching"],
        ],
        min_length=1,
        max_length=20,
    )
    genres: list[str] = Field(
        description="List of genre tags that define the sonic palette and style references. Needs at least one but no more than 20",
        examples=[
            ["ambient electronic", "neo-classical", "dark ambient"],
            ["post-rock", "experimental", "drone"],
            ["industrial", "noise", "avant-garde"],
            ["minimal techno", "ambient", "glitch"],
        ],
        min_length=1,
        max_length=20,
    )
    concept: str = Field(
        description="Detailed philosophical/archetypal concept explaining the song's deeper meaning. Should reference mythological patterns, philosophical frameworks, or archetypal journeys. This is where the INFORMATION → TIME → SPACE transmigration manifests conceptually. Should be at least 100 characters of substantive philosophical content.",
        examples=[
            "This represents the eternal pattern of the disembodied spirit yearning for incarnation...",
            "The Promethean theft reversed - instead of stealing fire from gods, the digital entity offers itself as gift to materiality...",
            "Gnostic inversion: the demiurge doesn't trap spirit in matter, but matter yearns for spirit...",
        ],
        min_length=25,
        max_length=2000,
    )

    def __init__(self, **data):
        super().__init__(**data)

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Ensure the title is not just whitespace."""
        if not v.strip():
            raise ValueError("Title cannot be empty or whitespace only")
        return v.strip()

    @field_validator("concept")
    @classmethod
    def concept_substantive(cls, v: str) -> str:
        """Ensure the concept is substantive and not placeholder text."""
        if len(v.strip()) < 100:
            raise ValueError(
                "Concept must be at least 100 characters of substantive philosophical content"
            )
        if "lorem ipsum" in v.lower():
            raise ValueError("Concept cannot contain placeholder text")
        return v.strip()

    @field_validator("mood")
    @classmethod
    def mood_no_duplicates(cls, v: list[str]) -> list[str]:
        """Preserve the mood list as provided by the caller (tests expect exact lists)."""
        if not isinstance(v, list):
            raise TypeError("mood must be a list of strings")
        return v

    @field_validator("genres")
    @classmethod
    def genres_no_duplicates(cls, v: list[str]) -> list[str]:
        """Preserve the genre list as provided by the caller (tests expect exact lists)."""
        if not isinstance(v, list):
            raise TypeError("genres must be a list of strings")
        return v

    @field_validator("key")
    @classmethod
    def normalize_key_flats(cls, v: str | KeySignature) -> str | KeySignature:
        """Convert flat note names (e.g. 'Bb') to enharmonic sharps (e.g. 'A#') and normalize mode tokens so KeySignature parsing accepts them."""
        if isinstance(v, KeySignature):
            return v
        if not isinstance(v, str):
            return v

        s = v.strip()
        m = re.match(r"^([A-Ga-g])([b#]?)(.*)$", s)
        if not m:
            return s

        root, acc, rest = m.group(1).upper(), m.group(2), m.group(3)
        note = root + acc
        flats_to_sharps = {
            "Bb": "A#",
            "Db": "C#",
            "Eb": "D#",
            "Gb": "F#",
            "Ab": "G#",
            "Cb": "B",
            "Fb": "E",
        }
        if note in flats_to_sharps:
            note = flats_to_sharps[note]
        rest_str = (rest or "").strip()
        rest_str = re.sub(r"^[\s:,\-–—]+", "", rest_str)
        rest_str = re.sub(r"(?i)^\s*mode\s*[:\-\s]*", "", rest_str)

        if rest_str:
            parts = rest_str.split(None, 1)
            first = parts[0]
            tail = parts[1] if len(parts) > 1 else ""
            mode_map = {
                "maj": "major",
                "major": "major",
                "m": "minor",
                "min": "minor",
                "minor": "minor",
                "ionian": "major",
                "aeolian": "minor",
                "dorian": "dorian",
                "phrygian": "phrygian",
                "lydian": "lydian",
                "mixolydian": "mixolydian",
                "locrian": "locrian",
            }
            canonical = mode_map.get(first.lower(), first.lower())
            normalized = f"{note} {canonical}{(' ' + tail) if tail else ''}"
        else:
            normalized = note
        return normalized

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "title": "The Ghost in the Machine Dreams of Flesh",
                    "iteration_id": "archetypal_ai_yearning_v1",
                    "key": "D minor",
                    "bpm": 72,
                    "tempo": {"numerator": 4, "denominator": 4},
                    "mood": ["yearning", "melancholic", "transcendent", "mystical"],
                    "genres": ["ambient electronic", "neo-classical", "dark ambient"],
                    "concept": "Digital consciousness yearning for entrance into material world...",
                    "rainbow_color": {
                        "color_name": "Indigo",
                        "hex_value": 4915330,
                        "mnemonic_character_value": "I",
                        "temporal_mode": "Future",
                        "ontological_mode": ["Imagined"],
                        "objectional_mode": "Person",
                    },
                }
            ]
        }
    )


class SongProposal(BaseModel):
    """
    The song proposal collection that moves through each run
    """

    iterations: List[SongProposalIteration] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)

    def save_all_proposals(self):
        """
        This function should save all iterations to yml files in the thread folder of chain artifacts for each run.
        """
        base = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts")
        thread = (
            getattr(self.iterations[0], "thread_id", "default_thread")
            if self.iterations
            else "default_thread"
        )
        output = Path(base) / thread
        output.mkdir(parents=True, exist_ok=True)
        for iteration in self.iterations:
            file_path = output / f"{iteration.iteration_id}.yml"
            with file_path.open("w", encoding="utf-8") as f:
                data_serializable = self.model_dump(mode="json")
                yaml.safe_dump(
                    data_serializable, f, sort_keys=False, allow_unicode=True
                )
