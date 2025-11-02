from typing import List

from app.structures.artifacts.base_chain_artifact import ChainArtifact
from app.structures.artifacts.book_data import BookData
from app.structures.artifacts.text_chain_artifact_file import TextChainArtifactFile


class BookArtifact(ChainArtifact):

    book: BookData | None = None
    book_data: BookData | None = None  # Alias-like attribute to sync with `book`
    excerpts: List[TextChainArtifactFile] | None = None
    thread_id: str

    def __init__(self, **data):
        super().__init__(**data)
        # Sync `book` and `book_data` to ensure consistency
        if self.book is None and self.book_data is not None:
            self.book = self.book_data
        elif self.book_data is None and self.book is not None:
            self.book_data = self.book


class ReactionBookArtifact(ChainArtifact):

    subject: BookArtifact | None = None
    author: str | None = None
    title: str | None = None
    pages:  List[TextChainArtifactFile] | None = None

    def __init__(self, **data):
        super().__init__(**data)
