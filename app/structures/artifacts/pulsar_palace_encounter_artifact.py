import logging
import os
import yaml

from abc import ABC
from typing import List, Optional
from pathlib import Path
from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.character_portrait_artifact import (
    CharacterPortraitArtifact,
)
from app.structures.concepts.pulsar_palace_character import PulsarPalaceCharacter
from app.structures.concepts.pulsar_palace_room import PulsarPalaceRoom
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType

logger = logging.getLogger(__name__)


class PulsarPalaceEncounterArtifact(ChainArtifact, ABC):
    chain_artifact_type: ChainArtifactType = Field(
        default=ChainArtifactType.GAME_RUN,
        description="Type of the chain artifact should always be GAME_RUN",
    )
    chain_artifact_file_type: ChainArtifactFileType = Field(
        ChainArtifactFileType.MARKDOWN,
        description="File format of the artifact: Markdown for text and images",
    )
    rainbow_color_mnemonic_character_value: str = Field(
        default="Y",
        description="Mnemonic character for rainbow color coding: Y always",
    )
    artifact_name: str = "pulsar_palace_game_run"
    encounter_id: Optional[str] = Field(
        default=None,
        description="The ID of the encounter.",
    )
    characters: List[PulsarPalaceCharacter] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="The Pulsar Palace RPG-style characters for a given game run.",
    )
    character_images: List[CharacterPortraitArtifact] = Field(
        default_factory=list,
        description="Images representing the characters in the game run.",
    )
    rooms: List[PulsarPalaceRoom] = Field(
        default_factory=list,
        description="The rooms visited during this game run.",
    )
    story: List[str] = Field(
        default_factory=list,
        description="Narrative segments for each room encountered.",
    )

    def __init__(self, **data):
        # Rebuild model to resolve forward references if needed
        if not hasattr(self.__class__, "_rebuilt"):
            # Import at runtime to make both PulsarPalaceCharacter and PulsarPalaceCharacterSheet available for model_rebuild
            from app.structures.concepts.pulsar_palace_character import (
                PulsarPalaceCharacter as _PulsarPalaceCharacter,
            )
            from app.structures.artifacts.pulsar_palace_character_sheet import (
                PulsarPalaceCharacterSheet as _PulsarPalaceCharacterSheet,
            )

            # Rebuild all three models with the complete namespace
            _PulsarPalaceCharacter.model_rebuild(
                _types_namespace={
                    "PulsarPalaceCharacterSheet": _PulsarPalaceCharacterSheet
                }
            )
            _PulsarPalaceCharacter._rebuilt = True

            _PulsarPalaceCharacterSheet.model_rebuild(
                _types_namespace={"PulsarPalaceCharacter": _PulsarPalaceCharacter}
            )
            _PulsarPalaceCharacterSheet._rebuilt = True

            self.__class__.model_rebuild(
                _types_namespace={
                    "PulsarPalaceCharacter": _PulsarPalaceCharacter,
                    "PulsarPalaceCharacterSheet": _PulsarPalaceCharacterSheet,
                }
            )
            self.__class__._rebuilt = True
        super().__init__(**data)

    def to_markdown(self) -> str:
        """Convert encounter artifact to formatted markdown."""
        md_lines = ["# Pulsar Palace Game Run", "", "## Characters", ""]
        for i, char in enumerate(self.characters, 1):
            md_lines.append(
                f"### Character {i}: {char.disposition.disposition} {char.profession.profession}"
            )
            md_lines.append(
                f"- **Background:** {char.background.place} ({char.background.time})"
            )
            md_lines.append(f"- **ON:** {char.on_current}/{char.on_max}")
            md_lines.append(f"- **OFF:** {char.off_current}/{char.off_max}")
            if hasattr(char, "inventory") and char.inventory:
                md_lines.append(f"- **Inventory:** {', '.join(char.inventory)}")
            if getattr(char, "portrait_artifact", None):
                try:
                    md_lines.append(
                        f"- **Portrait:** {char.portrait_artifact.for_prompt()}"
                    )
                except ValueError as e:
                    logger.error(f"Error getting portrait path: {e}")
                    img = getattr(
                        getattr(char, "portrait_artifact", None), "image", None
                    )
                    if img:
                        md_lines.append(
                            f"- **Portrait:** {getattr(img, 'file_path', '<no-path>')}"
                        )
            elif getattr(char, "portrait", None):
                md_lines.append(
                    f"- **Portrait:** {getattr(char.portrait, 'file_path', '<no-path>')}"
                )
            md_lines.append("")
        md_lines.append("## Encounters")
        md_lines.append("")
        for i, (room, narrative) in enumerate(zip(self.rooms, self.story), 1):
            md_lines.append(f"### Room {i}: {room.name}")
            md_lines.append(f"**Type:** {room.room_type}")
            md_lines.append(f"**Atmosphere:** {room.atmosphere}")
            md_lines.append("")
            md_lines.append("**Description:**")
            md_lines.append(room.description)
            md_lines.append("")
            if room.features:
                md_lines.append("**Features:**")
                for feature in room.features:
                    md_lines.append(f"- {feature}")
                md_lines.append("")
            if room.inhabitants:
                md_lines.append("**Inhabitants:**")
                for inhabitant in room.inhabitants:
                    md_lines.append(f"- {inhabitant}")
                md_lines.append("")
            if room.exits:
                md_lines.append(f"**Exits:** {', '.join(room.exits)}")
                md_lines.append("")
            md_lines.append("**What Happened:**")
            md_lines.append(narrative)
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
        md_lines.append("## Metadata")
        md_lines.append(f"- **Artifact ID:** {self.artifact_id}")
        md_lines.append(f"- **Thread ID:** {self.thread_id}")
        md_lines.append(
            f"- **Rainbow Color:** {self.rainbow_color_mnemonic_character_value}"
        )
        md_lines.append(f"- **Rooms Visited:** {len(self.rooms)}")
        md_lines.append(f"- **Characters:** {len(self.characters)}")
        return "\n".join(md_lines)

    def for_prompt(self) -> str:
        """Plain text game run summary for prompt inclusion."""
        parts = [f"## Characters ({len(self.characters)})"]
        for char in self.characters:
            parts.append(
                f"\n{char.disposition.disposition} {char.profession.profession}"
            )
            parts.append(f"From: {char.background.place} ({char.background.time})")
            if hasattr(char, "inventory") and char.inventory:
                if len(char.inventory) <= 3:
                    parts.append(f"Carries: {', '.join(char.inventory)}")
                else:
                    parts.append(
                        f"Carries: {', '.join(char.inventory[:3])} (+{len(char.inventory) - 3} more)"
                    )
        parts.append(f"\n## Encounters ({len(self.rooms)} rooms)")
        if len(self.rooms) <= 3:
            for i, (room, narrative) in enumerate(zip(self.rooms, self.story), 1):
                parts.append(f"\n### {room.name}")
                parts.append(f"{room.room_type} - {room.atmosphere}")
                parts.append(f"\n{narrative}")
        else:
            room, narrative = self.rooms[0], self.story[0]
            parts.append(f"\n### First: {room.name}")
            parts.append(f"{room.room_type} - {room.atmosphere}")
            parts.append(f"\n{narrative}")
            parts.append(f"\n[Traversed {len(self.rooms) - 2} additional rooms...]")
            room, narrative = self.rooms[-1], self.story[-1]
            parts.append(f"\n### Final: {room.name}")
            parts.append(f"{room.room_type} - {room.atmosphere}")
            parts.append(f"\n{narrative}")
        return "\n".join(parts)

    def save_file(self):
        file = Path(self.file_path, self.file_name)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as f:
            f.write(self.to_markdown())

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "encounter_id": self.encounter_id,
            "characters": [char.model_dump() for char in self.characters],
            "character_images": self.character_images,
            "portrait_artifacts": [
                (
                    char.portrait_artifact.model_dump()
                    if getattr(char, "portrait_artifact", None)
                    else None
                )
                for char in self.characters
            ],
            "rooms": [room.model_dump() for room in self.rooms],
            "story": self.story,
        }


if __name__ == "__main__":
    with open(
        os.path.join(
            os.getenv("AGENT_MOCK_DATA_PATH"),
            "yellow_encounter_narrative_artifact_mock.yml",
        ),
        "r",
    ) as f:
        data = yaml.safe_load(f)
        data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        encounter_artifact = PulsarPalaceEncounterArtifact(**data)
        print(encounter_artifact)
        encounter_artifact.save_file()
        print(encounter_artifact.flatten())
        p = encounter_artifact.for_prompt()
        print(p)
