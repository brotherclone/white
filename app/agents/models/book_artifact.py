from typing import List

from app.agents.models.base_chain_artifact import ChainArtifact
from app.agents.models.book_data import BookData
from app.agents.models.text_chain_artifact_file import TextChainArtifactFile


class BookArtifact(ChainArtifact):

    book: BookData | None = None
    excerpts: List[TextChainArtifactFile] | None = None
    thread_id: str

    def __init__(self, **data):
        super().__init__(**data)


class ReactionBookArtifact(ChainArtifact):

    subject: BookArtifact | None = None
    author: str | None = None
    title: str | None = None
    pages:  List[TextChainArtifactFile] | None = None

    def __init__(self, **data):
        super().__init__(**data)
