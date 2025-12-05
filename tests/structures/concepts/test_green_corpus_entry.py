from pydantic import BaseModel

from app.structures.concepts.green_corpus_entry import GreenCorpusEntry


def test_inheritance():
    assert issubclass(GreenCorpusEntry, BaseModel)


# ToDo: Add additional tests for GreenCorpusEntry fields and validation logic
