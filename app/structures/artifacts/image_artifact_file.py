from typing import Optional

from pydantic import Field

from app.structures.artifacts.base_artifact_file import BaseChainArtifactFile


class ImageChainArtifactFile(BaseChainArtifactFile):

    thread_id: Optional[str] = Field(
        default=None, description="Unique ID of the thread."
    )

    def __init__(self, **data):
        super().__init__(**data)
