from app.agents.models.base_chain_artifact_file import BaseChainArtifactFile


class AudioChainArtifactFile(BaseChainArtifactFile):

    sample_rate: int = 44100
    duration: float = 1.0
    channels: int = 1 | 2

    def __init__(self, **data):
        super().__init__(**data)