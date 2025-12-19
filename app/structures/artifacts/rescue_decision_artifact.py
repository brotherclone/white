import os
import yaml

from abc import ABC
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from pydantic import Field

from app.structures.artifacts.arbitrarys_survey_artifact import ArbitrarysSurveyArtifact
from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.last_human_artifact import LastHumanArtifact
from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType

load_dotenv()


class RescueDecisionArtifact(ChainArtifact, ABC):
    """
    The Arbitrary's decision: one rescued, all others documented.
    """

    chain_artifact_type: ChainArtifactType = Field(
        default=ChainArtifactType.RESCUE_DECISION,
        description="Compatibility string identifier for Arbitrary's rescue decision artifact",
    )
    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=ChainArtifactFileType.YML,
        description="File format of the artifact: YAML",
    )
    rainbow_color_mnemonic_character_value: str = Field(
        default="G", description="Mnemonic character for rainbow color coding: G always"
    )
    artifact_name: str = Field(
        default="rescue_decision",
        description="Artifact file name base: rescue_decision",
    )

    rescued_consciousness: ArbitrarysSurveyArtifact = Field(
        ..., description="The rescued consciousness"
    )
    documented_humans: List[LastHumanArtifact] = Field(
        default_factory=list, description="The documented humans"
    )
    documented_species: List[SpeciesExtinctionArtifact] = Field(
        default_factory=list, description="The documented species"
    )

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

    def flatten(self) -> Dict:
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "consciousness_rescued": 1,
            "humans_documented": len(self.documented_humans),
            "species_documented": len(self.documented_species),
            "rescued": self.rescued_consciousness.flatten(),
            "justification": self.rescue_justification,
            "rationale": self.rationale,
            "arbitrary_perspective": self.arbitrary_perspective,
        }

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

    def for_prompt(self) -> str:
        """Format rescue decision for prompt - focuses on choice and reasoning."""
        parts = [
            "## The Arbitrary's Decision",
            f"Rescued: {self.rescued_consciousness.identity}",
            f"Documented (not rescued): {len(self.documented_humans)} humans, {len(self.documented_species)} species",
            "\n## Justification",
            self.rescue_justification,
            "\n## Rationale",
            self.rationale,
            "\n## The Mind's Reflection",
            self.arbitrary_perspective,
        ]
        return "\n".join(parts)


if __name__ == "__main__":
    with open(
        os.path.join(
            os.getenv("AGENT_MOCK_DATA_PATH"),
            "rescue_decision_artifact_mock.yml",
        ),
        "r",
    ) as file:
        data = yaml.safe_load(file)
        data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        claudes_choice = RescueDecisionArtifact(**data)
        print(claudes_choice)
        claudes_choice.save_file()
        print(claudes_choice.flatten())
        p = claudes_choice.for_prompt()
        print(p)
