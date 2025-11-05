import pytest
from pydantic import ValidationError

from app.structures.artifacts.book_data import BookData
from app.structures.artifacts.reaction_book_chain_artifact import (
    ReactionBookChainArtifact,
)
from app.structures.artifacts.text_chain_artifact_file import TextChainArtifactFile
from app.structures.enums.book_condition import BookCondition
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.publisher_type import PublisherType


def minimal_book_data():
    """Helper to create minimal BookData"""
    return BookData(
        title="Test Book",
        author="Test Author",
        year=2020,
        publisher="Test Publisher",
        publisher_type=list(PublisherType)[0],
        pages=100,
        catalog_number="RA-0001",
        condition=list(BookCondition)[0],
        danger_level=1,
    )


def test_reaction_book_chain_artifact():
    """Test basic ReactionBookChainArtifact creation"""
    book_data = minimal_book_data()

    artifact = ReactionBookChainArtifact(
        thread_id="test-thread-book",
        book_data=book_data,
        original_book_title="Original Title",
        original_book_author="Original Author",
    )
    assert artifact.thread_id == "test-thread-book"
    assert artifact.chain_artifact_type == "book"
    assert artifact.book_data.title == "Test Book"
    assert artifact.original_book_title == "Original Title"
    assert artifact.original_book_author == "Original Author"


def test_reaction_book_chain_artifact_with_pages():
    """Test ReactionBookChainArtifact with page content"""
    book_data = minimal_book_data()

    page1 = TextChainArtifactFile(
        base_path="/tmp",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
        text_content="Page 1 content",
    )
    page2 = TextChainArtifactFile(
        base_path="/tmp",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
        text_content="Page 2 content",
    )

    artifact = ReactionBookChainArtifact(
        thread_id="test-thread",
        book_data=book_data,
        original_book_title="Original",
        original_book_author="Author",
        pages=[page1, page2],
    )
    assert len(artifact.pages) == 2
    assert artifact.pages[0].text_content == "Page 1 content"
    assert artifact.pages[1].text_content == "Page 2 content"


def test_reaction_book_chain_artifact_requires_book_data():
    """Test that BookData is required"""
    with pytest.raises(ValidationError):
        ReactionBookChainArtifact(
            thread_id="test-thread",
            original_book_title="Title",
            original_book_author="Author",
        )
