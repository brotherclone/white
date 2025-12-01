from abc import ABC
from typing import List, Optional

from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.image_artifact_file import ImageChainArtifactFile
from app.structures.concepts.pulsar_palace_character import PulsarPalaceCharacter
from app.structures.concepts.pulsar_palace_room import PulsarPalaceRoom
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType


class PulsarPalaceEncounterArtifact(ChainArtifact, ABC):

    chain_artifact_type: ChainArtifactType = ChainArtifactType.GAME_RUN
    chain_artifact_file_type: ChainArtifactFileType = ChainArtifactFileType.MARKDOWN
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
    character_images: List[ImageChainArtifactFile] = Field(
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
        super().__init__(**data)

    def to_markdown(self) -> str:
        """Convert encounter artifact to formatted markdown."""
        md_lines = []

        # Title
        md_lines.append("# Pulsar Palace Game Run")
        md_lines.append("")

        # Characters section
        md_lines.append("## Characters")
        md_lines.append("")
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
            md_lines.append("")

        # Encounters section
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

        # Metadata footer
        md_lines.append("## Metadata")
        md_lines.append(f"- **Artifact ID:** {self.artifact_id}")
        md_lines.append(f"- **Thread ID:** {self.thread_id}")
        md_lines.append(
            f"- **Rainbow Color:** {self.rainbow_color_mnemonic_character_value}"
        )
        md_lines.append(f"- **Rooms Visited:** {len(self.rooms)}")
        md_lines.append(f"- **Characters:** {len(self.characters)}")

        return "\n".join(md_lines)

    def save_file(self):
        from pathlib import Path

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
            "rooms": [room.model_dump() for room in self.rooms],
            "story": self.story,
        }
