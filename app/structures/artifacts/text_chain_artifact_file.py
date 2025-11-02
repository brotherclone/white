from app.structures.artifacts.base_chain_artifact_file import BaseChainArtifactFile


class TextChainArtifactFile(BaseChainArtifactFile):

    text_content: str | None = None

    def __init__(self, **data):
        super().__init__(**data)