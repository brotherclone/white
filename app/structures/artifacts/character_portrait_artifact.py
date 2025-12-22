import logging
from typing import Optional
from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.image_artifact_file import ImageChainArtifactFile


class CharacterPortraitArtifact(ChainArtifact):
    """High-level artifact representing a character portrait and its metadata.

    This separates character metadata (name, role, pose, description) from the
    low-level image file represented by `ImageChainArtifactFile`.
    """

    character_name: str = Field(..., description="Name of the character")
    role: Optional[str] = Field(None, description="Character role or archetype")
    pose: Optional[str] = Field(None, description="Pose or expression for the portrait")
    description: Optional[str] = Field(None, description="Optional description")
    image: Optional[ImageChainArtifactFile] = Field(
        default=None, description="The actual image file artifact (PNG)"
    )

    def save_file(self):
        """If an image artifact is attached, delegate saving to it; otherwise do nothing."""
        if self.image:
            if getattr(self.image, "thread_id", None) is None:
                self.image.thread_id = self.thread_id
            if getattr(self.image, "base_path", None) is None:
                self.image.base_path = self.base_path
            self.image.save_file()

    def flatten(self) -> dict:
        parent = super().flatten() or {}
        image_info = None
        if self.image:
            image_info = {
                "file_name": getattr(self.image, "file_name", None),
                "file_path": getattr(self.image, "file_path", None),
            }
        return {
            **parent,
            "character_name": self.character_name,
            "role": self.role,
            "pose": self.pose,
            "description": self.description,
            "image": image_info,
        }

    def for_prompt(self) -> str:
        parts = [f"Portrait of {self.character_name}"]
        if self.role:
            parts.append(f"({self.role})")
        if self.pose:
            parts.append(f"— {self.pose}")
        if self.description:
            parts.append(f"| {self.description}")
        if self.image:
            try:
                img_path = self.image.get_artifact_path(with_file_name=True)
            except EnvironmentError as e:
                logging.error(f"Error getting image path: {e}")
                img_path = getattr(self.image, "file_path", None) or "<no-path>"
            parts.append(f"Image: {img_path}")
        return " ".join(parts)
