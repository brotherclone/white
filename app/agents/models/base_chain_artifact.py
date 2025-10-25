from pydantic import BaseModel


class ChainArtifact(BaseModel):

    chain_artifact_type: str | None = None

    def __init__(self, **data):
        super().__init__(**data)