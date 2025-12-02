from abc import ABC
from pathlib import Path

from pydantic import ConfigDict

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType


class PulsarPalaceCharacterSheet(ChainArtifact, ABC):

    chain_artifact_type: ChainArtifactType = ChainArtifactType.CHARACTER_SHEET
    chain_artifact_file_type: ChainArtifactFileType = ChainArtifactFileType.MARKDOWN
    artifact_name: str = "character_sheet"
    model_config = ConfigDict(extra="allow")
    sheet_content: str = ""

    def __init__(self, **data):
        super().__init__(**data)

    def save_file(self):
        file = Path(self.file_path, self.file_name)
        file.parent.mkdir(parents=True, exist_ok=True)

        content = self.sheet_content
        if hasattr(content, "to_markdown") and callable(
            getattr(content, "to_markdown")
        ):
            try:
                md = content.to_markdown()
            except ValueError:
                md = str(content)
        else:
            md = content if isinstance(content, str) else str(content)

        with open(file, "w", encoding="utf-8") as f:
            f.write(md)

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
        }
