import os
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml
from dotenv import load_dotenv
from pydantic import ConfigDict, Field

from white_core.artifacts.base_artifact import ChainArtifact
from white_core.enums.chain_artifact_file_type import ChainArtifactFileType
from white_core.enums.chain_artifact_type import ChainArtifactType

if TYPE_CHECKING:
    from white_core.concepts.pulsar_palace_character import PulsarPalaceCharacter

load_dotenv()


class PulsarPalaceCharacterSheet(ChainArtifact, ABC):

    chain_artifact_type: ChainArtifactType = Field(
        default=ChainArtifactType.CHARACTER_SHEET,
        description="Type of the chain artifact should always be CHARACTER_SHEET",
    )
    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=ChainArtifactFileType.MARKDOWN,
        description="File format of the artifact",
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
        exclude=True,
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def __init__(self, **data):
        if not hasattr(self.__class__, "_rebuilt"):
            from white_core.concepts.pulsar_palace_character import (
                PulsarPalaceCharacter as _PulsarPalaceCharacter,
            )

            self.__class__.model_rebuild(
                _types_namespace={"PulsarPalaceCharacter": _PulsarPalaceCharacter}
            )
            self.__class__._rebuilt = True
        super().__init__(**data)

    def to_markdown(self) -> str:
        char = self.sheet_content
        if char is None:
            return "# Character Sheet\n\nNo character data available.\n"

        disposition = char.disposition.disposition if char.disposition else "Unknown"
        profession = char.profession.profession if char.profession else "Unknown"
        place = char.background.place if char.background else "Unknown"
        time = char.background.time if char.background else "Unknown"

        lines = [
            f"# {disposition} {profession}",
            "",
            f"**Background:** {place} ({time})",
            f"**Thread:** {self.thread_id}",
            "",
            "## Stats",
            "",
            f"- **ON:** {char.on_current}/{char.on_max}",
            f"- **OFF:** {char.off_current}/{char.off_max}",
        ]

        arrival = getattr(char, "arrival_circumstances", None)
        if arrival:
            lines += ["", "## Arrival", "", arrival]

        location = getattr(char, "current_location", None)
        if location:
            lines += ["", f"**Current Location:** {location}"]

        anchor = getattr(char, "reality_anchor", None)
        if anchor:
            lines += [f"**Reality Anchor:** {anchor}"]

        attunement = getattr(char, "frequency_attunement", None)
        if attunement is not None:
            lines += [f"**Frequency Attunement:** {attunement}"]

        inventory = getattr(self, "inventory", [])
        if inventory:
            lines += ["", "## Inventory", ""]
            for item in inventory:
                lines.append(f"- {item}")

        lines.append("")
        return "\n".join(lines)

    def save_file(self):
        file_path = Path(self.file_path)
        file_path.mkdir(parents=True, exist_ok=True)
        output_file = file_path / self.file_name
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

    def flatten(self):
        sheet_data = None
        if self.sheet_content is not None:
            if hasattr(self.sheet_content, "model_dump"):
                sheet_data = self.sheet_content.model_dump()
            elif hasattr(self.sheet_content, "dict"):
                sheet_data = self.sheet_content.dict()
            else:
                sheet_data = str(self.sheet_content)

        return {
            **self.model_dump(exclude={"sheet_content"}),
            "sheet_content": sheet_data,
        }

    def for_prompt(self) -> str:
        if self.sheet_content is None:
            return "No character sheet available."

        char = self.sheet_content
        return f"""Character Sheet:
Name: {char.disposition.disposition if char.disposition else 'Unknown'} {char.profession.profession if char.profession else 'Unknown'}
Background: {char.background.place if char.background else 'Unknown'} ({char.background.time if char.background else 'Unknown'})
ON: {char.on_current}/{char.on_max}
OFF: {char.off_current}/{char.off_max}"""


if __name__ == "__main__":
    with open(
        os.path.join(
            os.getenv("AGENT_MOCK_DATA_PATH", ""),
            "yellow_character_sheet_one_mock.yml",
        ),
        "r",
    ) as a_file:
        example_data = yaml.safe_load(a_file)

    base_path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
    example_data["base_path"] = base_path

    sheet = PulsarPalaceCharacterSheet(**example_data)
    print(sheet)
    sheet.save_file()
    print(sheet.flatten())
    p = sheet.for_prompt()
    print(p)
