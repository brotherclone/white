from pydantic import Field

from app.structures.artifacts.base_artifact_file import BaseChainArtifactFile


class AudioChainArtifactFile(BaseChainArtifactFile):
    """Audio-specific artifact file model kept in its own module for clarity.

    Recreated here to restore separation after a temporary consolidation.
    """

    sample_rate: int = Field(
        description="Sample rate of the audio file", ge=24000, le=96000, default=44100
    )
    duration: float = Field(
        description="Duration of the audio file in seconds",
        ge=0.0,
        le=1000000.0,
        default=1.0,
    )
    channels: int = Field(description="Number of audio channels", ge=1, le=2, default=2)

    def __init__(self, **data):
        super().__init__(**data)
