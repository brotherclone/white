import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.concepts.rainbow_table_color import RainbowTableColor

load_dotenv()


class BaseChainArtifactFile(BaseModel):

    artifact_id: uuid.UUID | str = None
    thread_id: str | None = None
    rainbow_color: RainbowTableColor | None = None
    base_path: str | Path
    chain_artifact_file_type: ChainArtifactFileType
    artifact_name: str | None = None
    file_name: str | None = None

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
        base = str(self.base_path) if self.base_path else os.getenv('AGENT_WORK_PRODUCT_BASE_PATH')
        if base is None:
            raise ValueError("Base path is required to build an artifact path.")
        base_path = Path(base)
        if self.thread_id is None:
            raise ValueError("Thread ID is required to build an artifact path.")
        if self.chain_artifact_file_type is None:
            raise ValueError("Chain artifact file type is required to build an artifact path.")
        if with_file_name:
            filename = self.file_name if self.file_name else "unknown.txt"
            return str(base_path / self.thread_id / self.chain_artifact_file_type.value / filename)
        else:
            return str(base_path / self.thread_id / self.chain_artifact_file_type.value)
