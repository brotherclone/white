from typing import List

from pydantic import BaseModel

from app.agents.models.base_chain_artifact_file import BaseChainArtifactFile


class ChainArtifact(BaseModel):

    chain_artifact_type: str
    files: List[BaseChainArtifactFile]

    def __init__(self, **data):
        super().__init__(**data)