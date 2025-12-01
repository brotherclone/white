from abc import ABC
from pathlib import Path
from typing import Optional

from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact


class ImageChainArtifactFile(ChainArtifact, ABC):

    file_path: Path = Field(description="Path to the image file")
    height: int = Field(description="Height of the image in pixels", ge=0, le=1000)
    width: int = Field(description="Width of the image in pixels", ge=0, le=1000)
    aspect_ratio: Optional[float] = Field(
        description="Aspect ratio of the image (width / height)",
        ge=0.0,
        le=1000.0,
        default=1.0,
    )

    def __init__(self, **data):
        super().__init__(**data)

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "thread_id": self.thread_id,
            "chain_artifact_type": self.chain_artifact_type.value,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "height": self.height,
            "width": self.width,
            "aspect_ratio": self.aspect_ratio,
        }

    def save_file(self):
        if self.base_path:
            file_path_obj = Path(self.base_path) / "png"
            file_path_obj.mkdir(parents=True, exist_ok=True)
            file_path = file_path_obj / self.file_name
        else:
            file_path = Path(self.file_name)
        with open(file_path, "wb") as f:
            f.write(self.image_bytes)
