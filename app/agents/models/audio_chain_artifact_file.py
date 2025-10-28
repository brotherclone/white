from app.agents.models.base_chain_artifact_file import BaseChainArtifactFile


class AudioChainArtifactFile(BaseChainArtifactFile):
    """Audio-specific artifact file model kept in its own module for clarity.

    Recreated here to restore separation after a temporary consolidation.
    """

    sample_rate: int = 44100
    duration: float = 1.0
    channels: int = 2

    def __init__(self, **data):
        super().__init__(**data)
