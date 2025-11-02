from typing import List

from app.structures.artifacts.base_chain_artifact import ChainArtifact
from app.structures.artifacts.book_data import BookData
from app.structures.artifacts.text_chain_artifact_file import TextChainArtifactFile


class BookChainArtifact(ChainArtifact):

    thread_id: str
    book_data: BookData
    chain_artifact_type: str = "book"
    pages: List[TextChainArtifactFile] | None = []