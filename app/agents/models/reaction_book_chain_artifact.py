from typing import List

from app.agents.models.base_chain_artifact import ChainArtifact
from app.agents.models.book_data import BookData
from app.agents.models.text_chain_artifact_file import TextChainArtifactFile


class ReactionBookChainArtifact(ChainArtifact):

    thread_id: str
    chain_artifact_type: str = "book"
    pages: List[TextChainArtifactFile] | None = []
    book_data: BookData
    original_book_title: str
    original_book_author: str