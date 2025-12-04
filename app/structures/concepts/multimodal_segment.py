from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MultimodalSegment(BaseModel):
    """
    Represents a multimodal segment extracted from a manifest.
    Combines lyrical, audio, MIDI, and conceptual data for a temporal segment.
    """

    # Core temporal identity
    manifest_id: str = Field(description="The manifest ID this segment belongs to")
    segment_type: str = Field(description="Type of segment (e.g., 'section')")
    segment_id: str = Field(description="Unique identifier for this segment")
    canonical_start: float = Field(description="Start time in seconds", ge=0)
    canonical_end: float = Field(description="End time in seconds", ge=0)
    duration: float = Field(description="Duration in seconds", ge=0)

    # Section metadata
    section_name: str = Field(description="Name of the section")
    section_description: Optional[str] = Field(
        default=None, description="Description of what happens in this section"
    )

    # Lyrical content
    lyrical_content: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of lyrics with temporal relationships and metadata",
    )

    # Musical context
    bpm: Optional[int] = Field(default=None, description="Beats per minute", ge=0)
    time_signature: Optional[str] = Field(
        default=None, description="Time signature (e.g., '4/4')"
    )
    key: Optional[str] = Field(default=None, description="Musical key")
    rainbow_color: Optional[str] = Field(
        default=None, description="Rainbow album color (R/O/Y/G/B/V/I)"
    )

    # Track metadata
    title: Optional[str] = Field(default=None, description="Track title")
    mood_tags: List[str] = Field(default_factory=list, description="Mood descriptors")
    concept: Optional[str] = Field(default=None, description="Conceptual description")

    # Audio features (optional - single track or aggregate)
    audio_features: Optional[Dict[str, Any]] = Field(
        default=None, description="Audio features for a single track (deprecated)"
    )

    # Multi-track audio features
    audio_tracks_features: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Audio features for each track with player info",
    )

    # MIDI features (optional)
    midi_features: Optional[Dict[str, Any]] = Field(
        default=None, description="MIDI features if available"
    )

    # Rebracketing analysis (basic)
    rebracketing_score: Optional[float] = Field(
        default=None,
        description="Boundary fluidity score across modalities",
        ge=0,
        le=1,
    )

    # Enhanced rebracketing analysis (optional)
    rebracketing_features: Optional[Dict[str, Any]] = Field(
        default=None, description="Enhanced rebracketing training features"
    )
    concept_analysis: Optional[Dict[str, Any]] = Field(
        default=None, description="Concept analysis including ontological categories"
    )
    ontological_category: Optional[str] = Field(
        default=None, description="Ontological category of the segment"
    )
    memory_discrepancy_severity: Optional[float] = Field(
        default=None, description="Memory discrepancy severity score", ge=0, le=1
    )
    temporal_complexity: Optional[float] = Field(
        default=None, description="Temporal rebracketing complexity", ge=0, le=1
    )
    section_rebracketing_score: Optional[float] = Field(
        default=None, description="Section-specific rebracketing score", ge=0, le=1
    )
    boundary_crossing_indicators: List[str] = Field(
        default_factory=list,
        description="Indicators of temporal/modal boundary crossing",
    )
    audio_rebracketing_metrics: Optional[Dict[str, Any]] = Field(
        default=None, description="Aggregate audio rebracketing metrics"
    )
    midi_rebracketing_metrics: Optional[Dict[str, Any]] = Field(
        default=None, description="MIDI rebracketing metrics"
    )
    comprehensive_rebracketing_score: Optional[float] = Field(
        default=None,
        description="Comprehensive rebracketing score across all modalities",
        ge=0,
        le=1,
    )

    # Computed fields for analysis
    lyric_count: Optional[int] = Field(
        default=None, description="Number of lyrics in segment"
    )
    has_temporal_bleeding: Optional[bool] = Field(
        default=None,
        description="Whether lyrics bleed across segment boundaries",
    )
    players: Optional[List[str]] = Field(
        default_factory=list, description="List of players in audio tracks"
    )
    player_count: Optional[int] = Field(
        default=None, description="Number of unique players"
    )

    def __init__(self, **data):
        super().__init__(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary for backwards compatibility"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalSegment":
        """Create segment from dictionary"""
        return cls(**data)
