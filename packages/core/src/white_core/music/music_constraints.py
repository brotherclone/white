"""Pydantic model for optional musical constraints in a song proposal.

When a color agent specifies an explicit harmonic intent (e.g. 'I→IV→I' or
'a single chord for two minutes'), it writes a ``musical_constraints`` block
in the proposal YAML.  The chord pipeline reads this block and, when
``harmonic_sequence`` is present, generates a constrained candidate set
alongside the standard Markov candidates.
"""

from __future__ import annotations

from pydantic import BaseModel


class MusicConstraints(BaseModel):
    """Optional harmonic constraints extracted from a song proposal."""

    harmonic_sequence: str | None = None
    """Space-separated Roman numeral tokens in order (e.g. 'i iv i').
    A single token ('i') expresses a one-chord song.  When present the chord
    pipeline builds candidates whose progressions follow this exact sequence."""

    performance_notes: str | None = None
    """Free-prose performance direction that the pipeline cannot encode
    (e.g. 'phrase resolutions anticipate the lyric by a half-beat').
    Surfaced as a top-level field in chords/review.yml for the human reviewer;
    has no pipeline effect."""

    model_config = {"extra": "ignore"}
