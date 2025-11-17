from app.structures.artifacts.reaction_book_artifact import (
    ReactionBookChainArtifact,
)
from app.structures.enums.book_condition import BookCondition
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.publisher_type import PublisherType


def test_reaction_book_chain_artifact():
    """Test basic ReactionBookChainArtifact creation"""
    book = ReactionBookChainArtifact(
        thread_id="m_1",
        title="Test Book",
        author="Test Author",
        year=2020,
        publisher="Test Publisher",
        publisher_type=list(PublisherType)[0],
        pages=100,
        catalog_number="RA-0001",
        condition=list(BookCondition)[0],
        danger_level=1,
        original_book_title="Original Title",
        original_book_author="Original Author",
        excerpts=["Page 1 content", "Page 2 content"],
    )
    assert book.thread_id == "m_1"
    assert book.chain_artifact_type == ChainArtifactType.BOOK
    assert book.title == "Test Book"
    assert book.original_book_title == "Original Title"

    assert len(book.excerpts) == 2
    assert book.excerpts[0] == "Page 1 content"
    assert book.excerpts[1] == "Page 2 content"
