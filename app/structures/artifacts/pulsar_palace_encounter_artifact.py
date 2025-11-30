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
    room: PulsarPalaceRoom = Field(
        default=None, description="The current room in the Pulsar Palace."
    )
    encounter_narrative_artifact: Optional[str] = Field(
        default=None, description="Narrative description of the current encounter."
    )

    def __init__(self, **data):
        super().__init__(**data)

    def save_file(self):
        # Add the characters to the top of the file
        # for each room generate a log of the actions taken
        # save as markdown
        pass
