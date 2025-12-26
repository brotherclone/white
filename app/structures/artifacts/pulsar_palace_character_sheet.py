import os
from typing import Optional, TYPE_CHECKING

import yaml

from abc import ABC
from pathlib import Path
from pydantic import Field, ConfigDict
from dotenv import load_dotenv

from app.structures.artifacts.html_artifact_file import HtmlChainArtifactFile
from app.structures.artifacts.template_renderer import (
    get_template_path,
    HTMLTemplateRenderer,
)
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType

if TYPE_CHECKING:
    from app.structures.concepts.pulsar_palace_character import PulsarPalaceCharacter

load_dotenv()


class PulsarPalaceCharacterSheet(HtmlChainArtifactFile, ABC):

    chain_artifact_type: ChainArtifactType = Field(
        default=ChainArtifactType.CHARACTER_SHEET,
        description="Type of the chain artifact should always be CHARACTER_SHEET",
    )
    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=ChainArtifactFileType.HTML,
        description="File format of the artifact: HTML for text and images",
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
        template_path = get_template_path("character_sheet")
        renderer = HTMLTemplateRenderer(template_path)

        # Calculate percentages
        # Exclude circular reference: sheet_content.character_sheet points back to self
        data = self.model_dump(exclude={"sheet_content": {"character_sheet"}})

        # Get character stats from sheet_content
        char = self.sheet_content
        if char:
            data["on_percentage"] = (
                int((char.on_current / char.on_max) * 100)
                if char.on_max and char.on_max > 0
                else 0
            )
            data["off_percentage"] = (
                int((char.off_current / char.off_max) * 100)
                if char.off_max and char.off_max > 0
                else 0
            )
        else:
            data["on_percentage"] = 0
            data["off_percentage"] = 0

        # Generate inventory slots HTML
        inventory_html = []
        inventory = getattr(self, "inventory", [])
        for i in range(9):  # 3x3 grid
            if i < len(inventory):
                inventory_html.append(
                    f'<div class="inventory-slot">{inventory[i]}</div>'
                )
            else:
                inventory_html.append('<div class="inventory-slot empty">EMPTY</div>')
        data["inventory_slots"] = "\n        ".join(inventory_html)

        html_content = renderer.render(data)

        file_path = Path(self.file_path)
        file_path.mkdir(parents=True, exist_ok=True)

        output_file = file_path / self.file_name
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

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
