"""Pydantic model for MIDI style reference profiles.

A StyleProfile captures statistical features extracted from MIDI files
associated with a `sounds_like` artist. Profiles are averaged to form an
aggregate reference that biases pipeline generation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StyleProfile(BaseModel):
    """Statistical features extracted from a set of MIDI files for one artist."""

    artist: str = ""
    note_density: float = Field(
        default=0.0,
        description="Mean notes per bar",
    )
    note_density_variance: float = Field(
        default=0.0,
        description="Variance of note density across bars",
    )
    mean_duration_beats: float = Field(
        default=0.5,
        description="Mean note duration in beats",
    )
    duration_variance: float = Field(
        default=0.0,
        description="Variance of note duration",
    )
    velocity_mean: float = Field(
        default=80.0,
        description="Mean MIDI velocity (0–127)",
    )
    velocity_variance: float = Field(
        default=10.0,
        description="Variance of MIDI velocity",
    )
    interval_histogram: dict[int, float] = Field(
        default_factory=dict,
        description="Semitone interval → relative frequency (sums to ~1.0)",
    )
    harmonic_rhythm: float = Field(
        default=1.0,
        description="Mean distinct pitch classes introduced per bar (proxy for chord change rate)",
    )
    rest_ratio: float = Field(
        default=0.0,
        description="Fraction of bars with >50% silence",
    )
    phrase_length_mean: float = Field(
        default=4.0,
        description="Mean notes per phrase (silence-delimited)",
    )
    style_weight: float = Field(
        default=0.4,
        description="How strongly this profile influences generation (0.0–1.0)",
    )

    model_config = {"extra": "allow"}
