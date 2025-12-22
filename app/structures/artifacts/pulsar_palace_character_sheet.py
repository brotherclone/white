import os
from typing import Optional, TYPE_CHECKING

import yaml

from abc import ABC
from pathlib import Path
from pydantic import Field, ConfigDict
from dotenv import load_dotenv

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType

if TYPE_CHECKING:
    from app.structures.concepts.pulsar_palace_character import PulsarPalaceCharacter

load_dotenv()


class PulsarPalaceCharacterSheet(ChainArtifact, ABC):

    chain_artifact_type: ChainArtifactType = Field(
        default=ChainArtifactType.CHARACTER_SHEET,
        description="Type of the chain artifact should always be CHARACTER_SHEET",
    )
    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=ChainArtifactFileType.MARKDOWN,
        description="File format of the artifact: Markdown for text and images",
    )
    artifact_name: str = Field(
        default="character_sheet", description="Name of the artifact"
    )
    rainbow_color_mnemonic_character_value: str = Field(
        default="Y", description="Mnemonic character for rainbow color coding: Y always"
    )
    sheet_content: Optional["PulsarPalaceCharacter"] = Field(
        default=None,
        description="The PulsarPalaceCharacter object representing the character data",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(self, **data):
        # Rebuild model to resolve forward references if needed
        if not hasattr(self.__class__, "_rebuilt"):
            # Import at runtime to make PulsarPalaceCharacter available for model_rebuild
            from app.structures.concepts.pulsar_palace_character import (
                PulsarPalaceCharacter as _PulsarPalaceCharacter,
            )

            self.__class__.model_rebuild(
                _types_namespace={"PulsarPalaceCharacter": _PulsarPalaceCharacter}
            )
            self.__class__._rebuilt = True
        super().__init__(**data)

    def build_sheet_content(
        self,
        name: str,
        background: str,
        profession: str,
        on_current: int,
        on_max: int,
        off_current: int,
        off_max: int,
    ):
        pass

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
        """Flatten the character sheet for easier processing."""
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}

        # Serialize PulsarPalaceCharacter if present
        sheet_data = None
        if self.sheet_content is not None:
            if hasattr(self.sheet_content, "model_dump"):
                sheet_data = self.sheet_content.model_dump()
            elif hasattr(self.sheet_content, "dict"):
                sheet_data = self.sheet_content.dict()
            else:
                sheet_data = str(self.sheet_content)

        return {
            **parent_data,
            "sheet_content": sheet_data,
        }

    def for_prompt(self) -> str:
        """Plain text character sheet for prompt."""
        if self.sheet_content is None:
            return "No character sheet available."

        # Convert PulsarPalaceCharacter to formatted string
        char = self.sheet_content
        return f"""Character Sheet:
Name: {char.disposition.disposition if char.disposition else 'Unknown'} {char.profession.profession if char.profession else 'Unknown'}
Background: {char.background.place if char.background else 'Unknown'} ({char.background.time if char.background else 'Unknown'})
ON: {char.on_current}/{char.on_max}
OFF: {char.off_current}/{char.off_max}"""


if __name__ == "__main__":
    with open(
        os.path.join(
            os.getenv("AGENT_MOCK_DATA_PATH"),
            "yellow_character_sheet_one_mock.yml",
        ),
        "r",
    ) as f:
        data = yaml.safe_load(f)
    sheet = PulsarPalaceCharacterSheet(**data)
    print(sheet)
    sheet.save_file()
    print(sheet.flatten())
    p = sheet.for_prompt()
    print(p)
