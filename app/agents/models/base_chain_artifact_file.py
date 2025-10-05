import os

from dotenv import load_dotenv
from pydantic import BaseModel

from app.agents.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.concepts.rainbow_table_color import RainbowTableColor

load_dotenv()


class BaseChainArtifactFile(BaseModel):

    rainbow_color: RainbowTableColor | None = None
    base_path: str
    chain_artifact_file_type: ChainArtifactFileType
    artifact_path: str | None = None
    file_name: str | None = None

    def __init__(self, **data):
        super().__init__(**data)

    def get_artifact_path(self, with_file_name: bool = True) -> str:
        base = self.base_path or os.getenv('AGENT_WORK_PRODUCT_BASE_PATH')
        col_dir = "unknown"
        type_dir = "unknown"
        if self.rainbow_color:
            col_dir = self.rainbow_color.color_name.lower()
        if self.chain_artifact_file_type:
            type_dir = self.chain_artifact_file_type.value if hasattr(self.chain_artifact_file_type, "value") else str(
                self.chain_artifact_file_type)
        if with_file_name:
            filename = self.file_name if self.file_name else ".unknown"
            return os.path.join(base, col_dir, type_dir, filename)
        else:
            return os.path.join(base, col_dir, type_dir)