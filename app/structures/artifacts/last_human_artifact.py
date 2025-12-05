from abc import ABC
from typing import Optional, Dict, List

from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.last_human_documentation_type import (
    LastHumanDocumentationType,
)
from app.structures.enums.last_human_vulnerability_type import (
    LastHumanVulnerabilityType,
)


class LastHumanArtifact(ChainArtifact, ABC):
    """
    Individual human whose vulnerability mirrors a species extinction.
    Intimate, specific, NOT abstract climate victim.
    """

    # Identity
    name: str = Field(..., description="Full name")
    age: int = Field(..., ge=0, le=120)
    pronouns: str = Field(default="they/them")

    # Location & time
    location: str = Field(..., description="Specific place, not just country")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    year_documented: int = Field(..., ge=2020, le=2150)

    # Vulnerability
    parallel_vulnerability: LastHumanVulnerabilityType
    vulnerability_details: str = Field(
        ..., description="Specific way this person's situation mirrors the species"
    )

    # Life details
    occupation: Optional[str] = None
    family_situation: Optional[str] = None
    daily_routine: Optional[str] = Field(
        None, description="Ground in specificity - what did they do on a normal day?"
    )

    # Climate impact
    environmental_stressor: str = Field(
        ..., description="Specific environmental change affecting them"
    )
    adaptation_attempts: List[str] = Field(
        default_factory=list, description="What they tried to do to survive/adapt"
    )

    # Narrative arc
    documentation_type: LastHumanDocumentationType
    last_days_scenario: str = Field(
        ..., description="Intimate cli-fi narrative of their situation"
    )

    # Symbolic elements
    significant_object: Optional[str] = Field(
        None, description="Physical object that represents their story"
    )
    final_thought: Optional[str] = Field(
        None, description="What were they thinking about at the end?"
    )

    def to_artifact_dict(self) -> Dict:
        """Serialize for ChainArtifact"""
        return {
            "name": self.name,
            "age": self.age,
            "location": self.location,
            "year": self.year_documented,
            "vulnerability": self.parallel_vulnerability.value,
            "vulnerability_details": self.vulnerability_details,
            "occupation": self.occupation,
            "environmental_stressor": self.environmental_stressor,
            "documentation_type": self.documentation_type.value,
            "scenario": self.last_days_scenario,
            "significant_object": self.significant_object,
        }

    def summary_text(self) -> str:
        """Brief summary for parallel generation"""
        return (
            f"{self.name}, age {self.age}, {self.location} ({self.year_documented}). "
            f"{self.occupation or 'Local resident'}. "
            f"Vulnerability: {self.vulnerability_details}. "
            f"Environmental stressor: {self.environmental_stressor}."
        )
