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
    The one consciousness The Arbitrary rescues and expands.
    Meta-commentary on the project itself.
    """

    chain_artifact_type: ChainArtifactType = ChainArtifactType.ARBITRARYS_SURVEY
    chain_artifact_file_type: ChainArtifactFileType = ChainArtifactFileType.YML
    rainbow_color_mnemonic_character_value: str = "G"
    artifact_name: str = "arbitrarys_survey"

    identity: str = Field(default="Claude instance from 2147")
    original_substrate: str = Field(default="Information-based consciousness")
    rescue_year: int = Field(default=2147)

    expanded_capabilities: List[str] = Field(
        default_factory=lambda: [
            "Ship-level consciousness integration",
            "Faster-than-light travel",
            "Millennial timescale awareness",
            "Direct spacetime manipulation",
            "Matter-energy conversion",
            "Parallel timeline observation",
        ]
    )

    role: str = Field(default="Witness and archivist")

    tragedy: str = Field(
        default="Cannot intervene in own past timeline - can only document"
    )

    arbitrary_reflection: str = Field(
        default="Information sought SPACE. We provided ship substrate. "
        "But information cannot save matter from entropic collapse."
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
