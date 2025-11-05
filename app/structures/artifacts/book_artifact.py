from typing import List

from app.structures.artifacts.base_chain_artifact import ChainArtifact
from app.structures.artifacts.book_data import BookData
from app.structures.artifacts.text_chain_artifact_file import \
    TextChainArtifactFile


class BookArtifact(ChainArtifact):

    book_data: BookData | None = None
    excerpts: List[TextChainArtifactFile] | None = None
    thread_id: str

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
