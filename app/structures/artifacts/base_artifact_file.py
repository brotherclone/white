import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.structures.concepts.rainbow_table_color import RainbowTableColor
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType

load_dotenv()


class BaseChainArtifactFile(BaseModel):
    artifact_id: Optional[uuid.UUID | str] = Field(
        default=None, description="Unique ID of the artifact."
    )
    thread_id: Optional[str] = Field(
        default=None, description="Unique ID of the thread."
    )
    rainbow_color: Optional[RainbowTableColor] = Field(
        default=None,
        description="Rainbow table color associated with the artifact.",
        examples=[
            {
                "color_name": "Black",
                "hex_value": "000000",
                "mnemonic_character_value": "Z",
                "transmigrational_mode": None,
                "objectional_mode": None,
                "ontological_mode": None,
                "temporal_mode": None,
                "file_prefix": "01",
            },
            {
                "color_name": "Red",
                "hex_value": "4915330",
                "mnemonic_character_value": "R",
                "temporal_mode": "Past",
                "objectional_mode": "Thing",
                "ontological_mode": ["Known"],
                "transmigrational_mode": None,
                "file_prefix": "02",
            },
        ],
    )
    base_path: str | Path = Field(
        default="/", description="Base path for the artifact."
    )
    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=None,
        description="Type of the chain artifact file.",
        examples=["MARKDOWN", "WAV"],
    )
    artifact_name: Optional[str] = Field(
        default=None,
        description="Name of the artifact.",
        examples=["sigil", "book", "transcript"],
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Generated file name of the artifact. Do not set manually.",
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.artifact_id is None:
            self.artifact_id = str(uuid.uuid4())
        self.file_name = self.get_file_name()

    def get_file_name(self) -> str:
        if self.rainbow_color is None:
            col = "transparent"
        else:
            col = self.rainbow_color.color_name.lower()
        return f"{self.artifact_id}_{col}_{self.artifact_name}.{self.chain_artifact_file_type.value}"

    def get_artifact_path(self, with_file_name: bool = True) -> str:
        # Normalize base_path to a string path
        base = (
            str(self.base_path)
            if self.base_path
            else os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
        )
        if base is None:
            raise ValueError("Base path is required to build an artifact path.")
        base_path = Path(base)
        if self.thread_id is None:
            raise ValueError("Thread ID is required to build an artifact path.")
        if self.chain_artifact_file_type is None:
            raise ValueError(
                "Chain artifact file type is required to build an artifact path."
            )
        if with_file_name:
            filename = self.file_name if self.file_name else "unknown.txt"
            return str(
                base_path
                / self.thread_id
                / self.chain_artifact_file_type.value
                / filename
            )
        else:
            return str(base_path / self.thread_id / self.chain_artifact_file_type.value)
