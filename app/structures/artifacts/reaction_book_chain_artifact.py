from typing import List, Optional

from pydantic import Field

from app.structures.artifacts.base_chain_artifact import ChainArtifact
from app.structures.artifacts.book_data import BookData
from app.structures.artifacts.text_chain_artifact_file import TextChainArtifactFile


class ReactionBookChainArtifact(ChainArtifact):

    thread_id: Optional[str] = Field(
        default=None, description="Unique ID of the thread."
    )
    chain_artifact_type: str = "book"
    pages: List[TextChainArtifactFile] | None = []
    book_data: BookData
    original_book_title: str
    original_book_author: str
