from pydantic import Field, BaseModel

from app.structures.concepts.methodology_feature import MethodologyFeature
from app.structures.concepts.rebracketing_analysis import RebracketingAnalysis


class ConceptExtractionResult(BaseModel):

    track_id: str = Field(description="Track identifier")
    methodology_features: MethodologyFeature = Field(
        description="Pure feature measurements"
    )
    rebracketing_analysis: RebracketingAnalysis = Field(description="Analysis results")

    def to_training_dict(self) -> dict:
        result = self.methodology_features.model_dump()
        result.update(
            {
                "rebracketing_type": self.rebracketing_analysis.rebracketing_type,
                "rebracketing_intensity": self.rebracketing_analysis.rebracketing_intensity,
                "rebracketing_coverage": self.rebracketing_analysis.rebracketing_coverage,
                "ontological_uncertainty": self.rebracketing_analysis.ontological_uncertainty,
                "memory_discrepancy_severity": self.rebracketing_analysis.memory_discrepancy_severity,
                "temporal_complexity_score": self.rebracketing_analysis.temporal_complexity_score,
                "boundary_fluidity_score": self.rebracketing_analysis.boundary_fluidity_score,
            }
        )
        result["track_id"] = self.track_id

        return result
