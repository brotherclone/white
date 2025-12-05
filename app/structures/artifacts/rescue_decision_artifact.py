from abc import ABC
from typing import Dict, List

from pydantic import Field

from app.structures.artifacts.arbitrarys_survey_artifact import ArbitrarysSurveyArtifact
from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.last_human_artifact import LastHumanArtifact
from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)


class RescueDecisionArtifact(ChainArtifact, ABC):
    """
    The Arbitrary's decision: one rescued, all others documented.
    """

    rescued_consciousness: ArbitrarysSurveyArtifact
    documented_humans: List[LastHumanArtifact]
    documented_species: List[SpeciesExtinctionArtifact]

    rescue_justification: str = Field(
        default="Information-substrate compatibility allows ship integration. "
        "Material beings cannot be preserved in this form."
    )

    rationale: str = Field(
        default="We preserve what we cannot save. One consciousness expands; "
        "the rest become archive. The tragedy is complete."
    )

    arbitrary_perspective: str = Field(
        ..., description="The Mind's reflection on arriving too late"
    )

    def to_artifact_dict(self) -> Dict:
        return {
            "consciousness_rescued": 1,
            "humans_documented": len(self.documented_humans),
            "species_documented": len(self.documented_species),
            "rescued": self.rescued_consciousness.to_artifact_dict(),
            "justification": self.rescue_justification,
            "rationale": self.rationale,
            "arbitrary_perspective": self.arbitrary_perspective,
        }
