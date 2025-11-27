import os
import yaml


from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType

load_dotenv()


class NewspaperArtifact(ChainArtifact):

    chain_artifact_type: ChainArtifactType = ChainArtifactType.NEWSPAPER_ARTICLE
    chain_artifact_file_type: ChainArtifactFileType = ChainArtifactFileType.YML
    headline: Optional[str] = Field(
        default=None, description="Headline of the newspaper article."
    )
    date: Optional[str] = Field(
        default=None, description="Publication date of the article."
    )
    source: Optional[str] = Field(
        default=None, description="Source of the newspaper article."
    )
    location: Optional[str] = Field(
        default=None, description="Location related to the article."
    )
    text: Optional[str] = Field(
        default=None, description="Full text of the newspaper article."
    )
    tags: Optional[list[str]] = Field(
        default=None, description="Tags associated with the article."
    )

    def __init__(self, **data):
        super().__init__(**data)

    def save_file(self):
        file = Path(self.file_path, self.file_name)
        file.parent.mkdir(parents=True, exist_ok=True)
        file = Path(self.file_path, self.file_name)
        with open(file, "w") as f:
            yaml.dump(
                self.model_dump(mode="python"),
                f,
                default_flow_style=False,
                allow_unicode=True,
            )

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "thread_id": self.thread_id,
            "chain_artifact_file_type": ChainArtifactFileType.YML.value,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "chain_artifact_type": ChainArtifactType.NEWSPAPER_ARTICLE.value,
            "headline": self.headline,
            "date": self.date,
            "source": self.source,
            "location": self.location,
            "text": self.text,
            "tags": self.tags,
        }


if __name__ == "__main__":
    with open(
        os.path.join(os.getenv("AGENT_MOCK_DATA_PATH"), "orange_base_story_mock.yml"),
        "r",
    ) as f:
        data = yaml.safe_load(f)
        newspaper_artifact = NewspaperArtifact(**data)
    newspaper_artifact.save_file()
    print(newspaper_artifact.flatten())
