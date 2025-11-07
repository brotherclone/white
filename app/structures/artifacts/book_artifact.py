from typing import List, Optional

from pydantic import Field

from app.structures.artifacts.base_chain_artifact import ChainArtifact
from app.structures.artifacts.book_data import BookData
from app.structures.artifacts.text_chain_artifact_file import TextChainArtifactFile


class BookArtifact(ChainArtifact):

    book_data: Optional[BookData] = Field(
        default=None,
        description="Book data associated with the artifact.",
        examples=[{}, {}],
    )
    excerpts: Optional[List[TextChainArtifactFile]] = Field(
        default=None,
        description="Excerpts associated with the artifact that will be generated and saved.",
        examples=[{}, {}],
    )
    thread_id: Optional[str] = Field(
        default=None, description="Unique ID of the thread."
    )

    def __init__(self, **data):
        super().__init__(**data)


class ReactionBookArtifact(ChainArtifact):

    subject: BookArtifact | None = None
    author: str | None = None
    title: str | None = None
    pages: List[TextChainArtifactFile] | None = None
    thread_id: str

    def __init__(self, **data):
        super().__init__(**data)
