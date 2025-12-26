from abc import ABC

from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType


class HtmlChainArtifactFile(ChainArtifact, ABC):

    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=ChainArtifactFileType.HTML,
        description="Type of the chain artifact file should always be HTML in this case.",
    )
    image_path: str = Field(description="Path to the image file.")
