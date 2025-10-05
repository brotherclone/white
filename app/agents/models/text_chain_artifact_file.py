from app.agents.models.base_chain_artifact_file import BaseChainArtifactFile


class TextChainArtifactFile(BaseChainArtifactFile):


    def __init__(self, **data):
        super().__init__(**data)