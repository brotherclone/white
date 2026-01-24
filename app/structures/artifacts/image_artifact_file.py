from abc import ABC
from pathlib import Path
from typing import Optional

from pydantic import Field, PrivateAttr

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType


class ImageChainArtifactFile(ChainArtifact, ABC):

    # If JPEG is added, remember to update the chain_artifact_file_type default value
    chain_artifact_file_type: ChainArtifactFileType = Field(
        default=ChainArtifactFileType.PNG,
        description="Type of the chain artifact file should always be PNG in this case.",
    )
    artifact_name: str = "img"
    # Note: file_path is inherited as a @property from ChainArtifact
    # For explicit file paths (e.g., existing images), store them separately
    _explicit_file_path: Optional[Path] = PrivateAttr(default=None)
    height: int = Field(description="Height of the image in pixels", ge=0, le=10000)
    width: int = Field(description="Width of the image in pixels", ge=0, le=10000)
    aspect_ratio: Optional[float] = Field(
        description="Aspect ratio of the image (width / height)",
        ge=0.0,
        le=1000.0,
        default=1.0,
    )

    def __init__(self, **data):
        # Extract explicit_file_path before calling super().__init__
        explicit_file_path = data.pop("file_path", None)
        super().__init__(**data)
        if explicit_file_path is not None:
            self._explicit_file_path = (
                Path(explicit_file_path)
                if not isinstance(explicit_file_path, Path)
                else explicit_file_path
            )

    @property
    def file_path(self) -> str | Path:
        """Return explicit file path if set, otherwise use computed path from base class"""
        if self._explicit_file_path is not None:
            return self._explicit_file_path
        # Use parent's computed property
        return super().file_path

    def flatten(self):
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}
        return {
            **parent_data,
            "thread_id": self.thread_id,
            "chain_artifact_type": self.chain_artifact_type.value,
            "file_name": self.file_name,
            "file_path": str(self.file_path),  # Convert to string for serialization
            "height": self.height,
            "width": self.width,
            "aspect_ratio": self.aspect_ratio,
        }

    def save_file(self):
        # Check if image_bytes exists (it's dynamically added, not a declared field)
        if not hasattr(self, "image_bytes") or self.image_bytes is None:
            raise ValueError(
                "Cannot save image file: image_bytes not set. "
                "Set the image_bytes attribute before calling save_file()."
            )

        if self.base_path:
            file_path_obj = Path(self.base_path) / "png"
            file_path_obj.mkdir(parents=True, exist_ok=True)
            file_path = file_path_obj / self.file_name
        else:
            file_path = Path(self.file_name)
        with open(file_path, "wb") as f:
            f.write(self.image_bytes)

    def for_prompt(self):
        return f"Image of {self.artifact_name}"
