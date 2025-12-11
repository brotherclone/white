from pydantic import BaseModel, Field, computed_field

from app.structures.concepts.rainbow_table_color import RainbowTableColor
from app.structures.enums.rebracketing_analysis_type import RebracketingAnalysisType


class RebracketingAnalysis(BaseModel):
    track_id: str = Field(
        description="Track identifier: {album_sequence}_{track_position}"
    )
    rebracketing_type: RebracketingAnalysisType = Field(
        description="Domain: CAUSAL, SPATIAL, PERCEPTUAL, EXPERIENTIAL, Bor TEMPORAL"
    )
    rebracketing_intensity: float = Field(
        description="Density of methodology markers (matches per 100 words)"
    )
    rebracketing_coverage: float = Field(
        description="Breadth of marker types (0.0 to 1.0)"
    )
    ontological_uncertainty: float = Field(
        description="IMAGINED/REAL boundary instability (0.0 to 1.0)"
    )
    memory_discrepancy_severity: float = Field(
        description="Memory revision intensity (0.0 to 1.0)"
    )
    temporal_complexity_score: float = Field(
        description="Temporal boundary complexity (0.0 to 1.0)"
    )
    boundary_fluidity_score: float = Field(
        description="Categorical fuzziness (0.0 to 1.0)"
    )
    original_memory: str = Field(description="Original concept text")
    concept_text_analyzed: str = Field(description="Normalized text used for analysis")

    rainbow_color: RainbowTableColor = Field(
        default=None, description="Color mnemonic (RED, ORANGE, etc.)"
    )

    @computed_field
    @property
    def album_sequence(self) -> int:
        """Derive an album sequence from track_id"""
        return int(self.track_id.split("_")[0])

    @computed_field
    @property
    def track_position(self) -> int:
        """Derive track position from track_id"""
        return int(self.track_id.split("_")[1])
