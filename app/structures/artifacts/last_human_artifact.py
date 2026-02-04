import os
import yaml

from abc import ABC
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv
from pydantic import Field, field_validator

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.last_human_documentation_type import (
    LastHumanDocumentationType,
)
from app.structures.enums.last_human_vulnerability_type import (
    LastHumanVulnerabilityType,
)
from app.util.string_utils import sanitize_for_filename

load_dotenv()


class LastHumanArtifact(ChainArtifact, ABC):
    """
    Individual human whose vulnerability mirrors a species extinction.
    Intimate, specific, NOT abstract climate victim.
    """

    chain_artifact_type: ChainArtifactType = ChainArtifactType.LAST_HUMAN
    chain_artifact_file_type: ChainArtifactFileType = ChainArtifactFileType.YML
    rainbow_color_mnemonic_character_value: str = "G"
    name: str = Field(..., description="Full name")
    age: int = Field(..., ge=0, le=120)
    pronouns: str = Field(default="they/them")
    location: str = Field(..., description="Specific place, not just country")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    year_documented: int = Field(
        ...,
        description="MUST be 1975 OR between 2028-2350. The Arbitrary visited Earth briefly in 1975, then returns in 2028+. No other years are valid.",
    )
    parallel_vulnerability: LastHumanVulnerabilityType
    vulnerability_details: str = Field(
        ..., description="Specific way this person's situation mirrors the species"
    )
    occupation: Optional[str] = None
    family_situation: Optional[str] = None
    daily_routine: Optional[str] = Field(
        None, description="Ground in specificity - what did they do on a normal day?"
    )
    environmental_stressor: str = Field(
        ..., description="Specific environmental change affecting them"
    )
    adaptation_attempts: List[str] = Field(
        default_factory=list, description="What they tried to do to survive/adapt"
    )
    documentation_type: LastHumanDocumentationType
    last_days_scenario: str = Field(
        ..., description="Intimate cli-fi narrative of their situation"
    )
    significant_object: Optional[str] = Field(
        None, description="Physical object that represents their story"
    )
    final_thought: Optional[str] = Field(
        None, description="What were they thinking about at the end?"
    )

    def __init__(self, **data):
        if "artifact_name" not in data and "name" in data:
            data["artifact_name"] = sanitize_for_filename(data["name"])
        super().__init__(**data)

    @field_validator("year_documented")
    @classmethod
    def _validate_year_documented(cls, v: int) -> int:
        if v == 1975 or (2028 <= v <= 2350):
            return v
        raise ValueError(
            "year_documented must be 1975 or between 2028 and 2350 (inclusive)"
        )

    def flatten(self) -> Dict:
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
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

    def save_file(self):
        file = Path(self.file_path, self.file_name)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as f:
            yaml.dump(
                self.model_dump(mode="python"),
                f,
                default_flow_style=False,
                allow_unicode=True,
            )

    def for_prompt(self):
        """Format for prompt - emphasizes the parallel structure."""
        parts = [
            f"{self.name}, age {self.age}",
            f"Location: {self.location}, {self.year_documented}",
        ]
        if self.occupation:
            parts.append(f"Occupation: {self.occupation}")
        parts.append(f"\nParallel Vulnerability: {self.parallel_vulnerability.value}")
        parts.append(f"{self.vulnerability_details}")
        parts.append(f"\nEnvironmental Stressor: {self.environmental_stressor}")
        if self.adaptation_attempts:
            if len(self.adaptation_attempts) <= 3:
                parts.append(
                    f"Adaptation attempts: {', '.join(self.adaptation_attempts)}"
                )
            else:
                parts.append(
                    f"Adaptation attempts: {', '.join(self.adaptation_attempts[:3])} (and {len(self.adaptation_attempts) - 3} others)"
                )
        parts.append(f"\n{self.last_days_scenario}")
        if self.significant_object:
            parts.append(f"\nSignificant object: {self.significant_object}")
        if self.final_thought:
            parts.append(f'Final thought: "{self.final_thought}"')
        return "\n".join(parts)


if __name__ == "__main__":
    with open(
        os.path.join(
            os.getenv("AGENT_MOCK_DATA_PATH"),
            "last_human_artifact_mock.yml",
        ),
        "r",
    ) as f:
        data = yaml.safe_load(f)
        data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        last_human_artifact = LastHumanArtifact(**data)
        print(last_human_artifact)
        last_human_artifact.save_file()
        print(last_human_artifact.flatten())
        p = last_human_artifact.for_prompt()
        print(p)
