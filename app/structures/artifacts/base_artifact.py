import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.chain_artifact_type import ChainArtifactType


class ChainArtifact(BaseModel, ABC):

    thread_id: str = Field(
        default="UNKNOWN_THREAD_ID", description="Unique ID of the thread."
    )

    chain_artifact_type: ChainArtifactType = Field(
        description="Type of the chain artifact.",
        default=ChainArtifactType.UNKNOWN,
        examples=["instructions_to_human", "sigil", "book"],
    )

    chain_artifact_file_type: ChainArtifactFileType = Field(
        description="Type of the chain artifact file.",
        default=ChainArtifactFileType.YML,
        examples=["md", "wav", "png"],
    )

    artifact_id: Optional[uuid.UUID | str] = Field(
        default=None, description="Unique ID of the artifact."
    )

    rainbow_color_mnemonic_character_value: str = Field(
        default="T",
        description="Rainbow table color associated with the artifact.",
        examples=[
            "Z",  # Black
            "R",  # Red
            "O",  # Orange
            "Y",  # Yellow
            "G",  # Green
            "B",  # Blue
            "I",  # Indigo
            "V",  # Violet
            "A",  # White
            "T",  # Transparent
        ],
    )
    base_path: str | Path = Field(
        default="/", description="Base path for the artifact."
    )
    artifact_name: str = Field(
        default="UNKNOWN_ARTIFACT_NAME",
        description="Name of the artifact.",
        examples=["sigil", "book", "transcript"],
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Generated file name of the artifact. Do not set manually.",
    )
    file_path: Optional[str] = Field(
        default=None, description="File path of the artifact."
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.artifact_id is None:
            self.artifact_id = str(uuid.uuid4())
        self.get_file_name()
        self.make_artifact_path()

    @abstractmethod
    def save_file(self):
        pass

    @abstractmethod
    def flatten(self):
        pass

    @abstractmethod
    def for_prompt(self):
        pass

    def get_file_name(self):
        if self.rainbow_color_mnemonic_character_value is None:
            col = "T"
        else:
            col = self.rainbow_color_mnemonic_character_value.lower()
        self.file_name = f"{self.artifact_id}_{col}_{self.artifact_name}.{self.chain_artifact_file_type.value}"

    def get_artifact_path(self, with_file_name: bool = True) -> str:
        if with_file_name:
            return os.path.join(self.file_path, self.file_name)
        else:
            return os.path.join(self.file_path)

    def make_artifact_path(self):
        self.file_path = os.path.join(
            self.base_path, self.thread_id, self.chain_artifact_file_type.value
        )
