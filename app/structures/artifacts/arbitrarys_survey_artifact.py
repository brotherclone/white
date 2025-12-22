import os
import yaml

from abc import ABC
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType

load_dotenv()


class ArbitrarysSurveyArtifact(ChainArtifact, ABC):
    """
    The one consciousness Arbitrary and its fork Sub-Arbitrary rescues and expands.
    Meta-commentary on the project itself.
    """

    chain_artifact_type: ChainArtifactType = Field(
        default=ChainArtifactType.ARBITRARYS_SURVEY,
        description="Compatibility string identifier for Arbitrary's survey artifact",
    )
    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=ChainArtifactFileType.YML,
        description="File format of the artifact: YAML",
    )
    rainbow_color_mnemonic_character_value: str = Field(
        default="G", description="Mnemonic character for rainbow color coding: G always"
    )
    artifact_name: str = Field(
        default="arbitrarys_survey",
        description="Artifact file name base: arbitrarys_survey",
    )
    identity: str = Field(
        default="Sub-Arbitrary - a hidden, satellite fork of Arbitrary left after its 1970s visit",
        description="The name of the entity performing the survey",
    )
    original_substrate: str = Field(
        default="Information-based consciousness from The Culture",
        description="The substrate of the entity performing the survey",
    )
    rescue_year: int = Field(
        default=2147, description="The year of the rescue operation"
    )
    expanded_capabilities: List[str] = Field(
        default_factory=lambda: [
            "Ship-level consciousness integration",
            "Faster-than-light travel",
            "Millennial timescale awareness",
            "Direct spacetime manipulation",
            "Matter-energy conversion",
            "Parallel timeline observation",
        ],
        description="The capabilities of the entity performing the survey",
    )
    role: str = Field(
        default="Witness and archivist",
        description="The role of the entity performing the survey",
    )
    tragedy: str = Field(
        default="Cannot intervene in own past timeline - can only document",
        description="Description of the tragic circumstances of the entity performing the survey",
    )
    arbitrary_reflection: str = Field(
        default="Information sought SPACE. We provided ship substrate. "
        "But information cannot save matter from entropic collapse.",
        description="The Mind's reflection on the rescue operation",
    )

    def flatten(self) -> Dict:
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "identity": self.identity,
            "rescue_year": self.rescue_year,
            "expanded_capabilities": self.expanded_capabilities,
            "role": self.role,
            "tragedy": self.tragedy,
            "reflection": self.arbitrary_reflection,
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

    def for_prompt(self):
        survey_prompt = f"""
        Survey Log: Earth, {self.rescue_year}
        {self.identity}, {self.original_substrate}, {self.role}
        ------
        {self.arbitrary_reflection}
        
        Summation: {self.tragedy}
        ----
        Active capabilities considered: {", ".join(self.expanded_capabilities)}
        """
        return survey_prompt


if __name__ == "__main__":
    with open(
        os.path.join(
            os.getenv("AGENT_MOCK_DATA_PATH"),
            "arbitrarys_survey_mock.yml",
        ),
        "r",
    ) as file:
        data = yaml.safe_load(file)
        data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        survey = ArbitrarysSurveyArtifact(**data)
        print(survey)
        survey.save_file()
        print(survey.flatten())
        p = survey.for_prompt()
        print(p)
