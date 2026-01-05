import pytest
from pydantic import ValidationError

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


def test_minimal_required_fields():
    """Test creating book with only required fields."""
    book = ReactionBookChainArtifact(
        thread_id="minimal",
        title="Minimal Book",
        author="Jane Doe",
        year=2023,
        publisher="Test Publisher",
        publisher_type=PublisherType.UNIVERSITY,
        pages=200,
        catalog_number="MIN-001",
        condition=BookCondition.GOOD,
        danger_level=2,
        original_book_title="Original",
        original_book_author="Original Author",
    )

    # Check required fields
    assert book.title == "Minimal Book"
    assert book.author == "Jane Doe"
    assert book.year == 2023

    # Check defaults
    assert book.edition == "1st"
    assert book.language == "English"
    assert book.tags == []
    assert book.related_works == []

    # Check optional fields are None
    assert book.subtitle is None
    assert book.author_credentials is None
    assert book.isbn is None
    assert book.acquisition_date is None
    assert book.translated_from is None
    assert book.translator is None
    assert book.abstract is None
    assert book.notable_quote is None
    assert book.suppression_history is None


def test_for_prompt_basic():
    """Test for_prompt() method with basic fields."""
    book = ReactionBookChainArtifact(
        thread_id="test",
        title="The Test Book",
        author="Test Author",
        year=2020,
        publisher="Test Press",
        publisher_type=PublisherType.UNIVERSITY,
        pages=300,
        catalog_number="TB-001",
        condition=BookCondition.PRISTINE,
        danger_level=3,
        original_book_title="Original",
        original_book_author="Original Author",
    )

    prompt = book.for_prompt()

    assert "Book: The Test Book" in prompt
    assert "Author: Test Author" in prompt
    assert "Year: 2020" in prompt
    assert "Publisher: Test Press" in prompt
    assert "Condition:" in prompt
    assert "Danger level: 3" in prompt
    assert "File:" in prompt


def test_for_prompt_with_subtitle():
    """Test for_prompt() includes subtitle when present."""
    book = ReactionBookChainArtifact(
        thread_id="test",
        title="Main Title",
        subtitle="A Compelling Subtitle",
        author="Test Author",
        year=2020,
        publisher="Test Press",
        publisher_type=PublisherType.OCCULT,
        pages=250,
        catalog_number="ST-001",
        condition=BookCondition.WORN,
        danger_level=1,
        original_book_title="Original",
        original_book_author="Original Author",
    )

    prompt = book.for_prompt()

    assert "Main Title — A Compelling Subtitle" in prompt


def test_for_prompt_with_abstract_and_quote():
    """Test for_prompt() includes abstract and notable_quote when present."""
    book = ReactionBookChainArtifact(
        thread_id="test",
        title="Quoted Book",
        author="Quotable Author",
        year=2021,
        publisher="Quote Press",
        publisher_type=PublisherType.SAMIZDAT,
        pages=400,
        catalog_number="QB-001",
        condition=BookCondition.DAMAGED,
        danger_level=5,
        original_book_title="Original",
        original_book_author="Original Author",
        abstract="A fascinating exploration of forbidden knowledge",
        notable_quote="Knowledge is the doorway to transformation",
    )

    prompt = book.for_prompt()

    assert "Abstract: A fascinating exploration of forbidden knowledge" in prompt
    assert "Notable quote: Knowledge is the doorway to transformation" in prompt


def test_for_prompt_without_optional_fields():
    """Test for_prompt() without abstract or quote."""
    book = ReactionBookChainArtifact(
        thread_id="test",
        title="Plain Book",
        author="Plain Author",
        year=2019,
        publisher="Plain Press",
        publisher_type=PublisherType.VANITY,
        pages=150,
        catalog_number="PB-001",
        condition=BookCondition.GOOD,
        danger_level=2,
        original_book_title="Original",
        original_book_author="Original Author",
    )

    prompt = book.for_prompt()

    assert "Abstract:" not in prompt
    assert "Notable quote:" not in prompt


def test_all_optional_fields():
    """Test creating book with all optional fields populated."""
    book = ReactionBookChainArtifact(
        thread_id="complete",
        title="Complete Book",
        subtitle="Every Field Filled",
        author="Complete Author",
        author_credentials="PhD in Occult Studies",
        year=2022,
        publisher="Complete Press",
        publisher_type=PublisherType.UNIVERSITY,
        edition="2nd Revised",
        pages=500,
        isbn="978-1-234-56789-0",
        catalog_number="CB-001",
        condition=BookCondition.PRISTINE,
        acquisition_date="2023-01-15",
        acquisition_notes="Found in rare bookstore",
        language="French",
        translated_from="Latin",
        translator="Jean Translator",
        tags=["occult", "forbidden", "rare"],
        danger_level=4,
        abstract="A complete exploration",
        notable_quote="Everything has its place",
        suppression_history="Banned in 1925, rediscovered 2020",
        related_works=["Related Book 1", "Related Book 2"],
        excerpts=["Excerpt one", "Excerpt two", "Excerpt three"],
        original_book_title="Liber Completum",
        original_book_author="Auctor Originalis",
    )

    assert book.subtitle == "Every Field Filled"
    assert book.author_credentials == "PhD in Occult Studies"
    assert book.edition == "2nd Revised"
    assert book.isbn == "978-1-234-56789-0"
    assert book.acquisition_date == "2023-01-15"
    assert book.acquisition_notes == "Found in rare bookstore"
    assert book.language == "French"
    assert book.translated_from == "Latin"
    assert book.translator == "Jean Translator"
    assert len(book.tags) == 3
    assert len(book.related_works) == 2
    assert len(book.excerpts) == 3
    assert book.suppression_history == "Banned in 1925, rediscovered 2020"


def test_translated_book():
    """Test a book with translation metadata."""
    book = ReactionBookChainArtifact(
        thread_id="trans",
        title="Das Verbotene Buch",
        author="Klaus Schmidt",
        year=2018,
        publisher="German Press",
        publisher_type=PublisherType.UNIVERSITY,
        pages=320,
        catalog_number="TR-001",
        condition=BookCondition.GOOD,
        danger_level=3,
        language="German",
        translated_from="Ancient Greek",
        translator="Maria Translator",
        original_book_title="Το Απαγορευμένο Βιβλίο",
        original_book_author="Αρχαίος Συγγραφέας",
    )

    assert book.language == "German"
    assert book.translated_from == "Ancient Greek"
    assert book.translator == "Maria Translator"
    assert book.original_book_title == "Το Απαγορευμένο Βιβλίο"


def test_danger_levels():
    """Test different danger levels."""
    for level in range(1, 6):
        book = ReactionBookChainArtifact(
            thread_id=f"danger_{level}",
            title=f"Level {level} Book",
            author="Dangerous Author",
            year=2020,
            publisher="Dangerous Press",
            publisher_type=PublisherType.OCCULT,
            pages=200,
            catalog_number=f"DL-{level:03d}",
            condition=BookCondition.GOOD,
            danger_level=level,
            original_book_title="Original",
            original_book_author="Original Author",
        )
        assert book.danger_level == level


def test_all_publisher_types():
    """Test creating books with each publisher type."""
    for pub_type in PublisherType:
        book = ReactionBookChainArtifact(
            thread_id="pub_test",
            title="Publisher Test",
            author="Test Author",
            year=2020,
            publisher="Test Publisher",
            publisher_type=pub_type,
            pages=100,
            catalog_number="PT-001",
            condition=BookCondition.GOOD,
            danger_level=1,
            original_book_title="Original",
            original_book_author="Original Author",
        )
        assert book.publisher_type == pub_type


def test_all_book_conditions():
    """Test creating books with each condition."""
    for condition in BookCondition:
        book = ReactionBookChainArtifact(
            thread_id="cond_test",
            title="Condition Test",
            author="Test Author",
            year=2020,
            publisher="Test Publisher",
            publisher_type=PublisherType.UNIVERSITY,
            pages=100,
            catalog_number="CT-001",
            condition=condition,
            danger_level=1,
            original_book_title="Original",
            original_book_author="Original Author",
        )
        assert book.condition == condition


def test_flatten_includes_all_fields():
    """Test that flatten() includes all fields."""
    book = ReactionBookChainArtifact(
        thread_id="flatten_test",
        title="Flatten Test",
        subtitle="Testing Flatten",
        author="Flat Author",
        year=2021,
        publisher="Flat Press",
        publisher_type=PublisherType.UNIVERSITY,
        pages=250,
        catalog_number="FT-001",
        condition=BookCondition.GOOD,
        danger_level=2,
        original_book_title="Original Flatten",
        original_book_author="Original Flat Author",
        tags=["test", "flatten"],
        related_works=["Related Work"],
    )

    flat = book.flatten()

    # Check all required fields are present
    assert flat["title"] == "Flatten Test"
    assert flat["subtitle"] == "Testing Flatten"
    assert flat["author"] == "Flat Author"
    assert flat["year"] == 2021
    assert flat["publisher"] == "Flat Press"
    assert flat["publisher_type"] == PublisherType.UNIVERSITY.value
    assert flat["pages"] == 250
    assert flat["catalog_number"] == "FT-001"
    assert flat["condition"] == BookCondition.GOOD.value
    assert flat["danger_level"] == 2
    assert flat["original_book_title"] == "Original Flatten"
    assert flat["original_book_author"] == "Original Flat Author"
    assert flat["tags"] == ["test", "flatten"]
    assert flat["related_works"] == ["Related Work"]
    assert flat["chain_artifact_type"] == ChainArtifactType.BOOK.value


def test_chain_artifact_type_is_book():
    """Test that chain_artifact_type is always BOOK."""
    book = ReactionBookChainArtifact(
        thread_id="type_test",
        title="Type Test",
        author="Test Author",
        year=2020,
        publisher="Test Publisher",
        publisher_type=PublisherType.UNIVERSITY,
        pages=100,
        catalog_number="TT-001",
        condition=BookCondition.GOOD,
        danger_level=1,
        original_book_title="Original",
        original_book_author="Original Author",
    )

    assert book.chain_artifact_type == ChainArtifactType.BOOK


def test_excerpts_none_vs_empty_list():
    """Test difference between None and empty list for excerpts."""
    book_none = ReactionBookChainArtifact(
        thread_id="none",
        title="No Excerpts",
        author="Test Author",
        year=2020,
        publisher="Test Publisher",
        publisher_type=PublisherType.UNIVERSITY,
        pages=100,
        catalog_number="NE-001",
        condition=BookCondition.GOOD,
        danger_level=1,
        original_book_title="Original",
        original_book_author="Original Author",
        excerpts=None,
    )
    assert book_none.excerpts is None

    book_empty = ReactionBookChainArtifact(
        thread_id="empty",
        title="Empty Excerpts",
        author="Test Author",
        year=2020,
        publisher="Test Publisher",
        publisher_type=PublisherType.UNIVERSITY,
        pages=100,
        catalog_number="EE-001",
        condition=BookCondition.GOOD,
        danger_level=1,
        original_book_title="Original",
        original_book_author="Original Author",
        excerpts=[],
    )
    assert book_empty.excerpts == []


def test_tags_and_related_works_default_to_empty_list():
    """Test that tags and related_works default to empty lists."""
    book = ReactionBookChainArtifact(
        thread_id="defaults",
        title="Default Test",
        author="Test Author",
        year=2020,
        publisher="Test Publisher",
        publisher_type=PublisherType.UNIVERSITY,
        pages=100,
        catalog_number="DT-001",
        condition=BookCondition.GOOD,
        danger_level=1,
        original_book_title="Original",
        original_book_author="Original Author",
    )

    assert book.tags == []
    assert book.related_works == []
    assert isinstance(book.tags, list)
    assert isinstance(book.related_works, list)


def test_missing_required_fields_raises_validation_error():
    """Test that missing required fields raises ValidationError."""
    with pytest.raises(ValidationError):
        # Missing original_book_title
        ReactionBookChainArtifact(
            thread_id="incomplete",
            title="Incomplete",
            author="Test Author",
            year=2020,
            publisher="Test Publisher",
            publisher_type=PublisherType.UNIVERSITY,
            pages=100,
            catalog_number="IC-001",
            condition=BookCondition.GOOD,
            danger_level=1,
            original_book_author="Original Author",
        )


def test_suppression_history():
    """Test suppression_history field."""
    book = ReactionBookChainArtifact(
        thread_id="suppressed",
        title="Forbidden Knowledge",
        author="Censored Author",
        year=1920,
        publisher="Underground Press",
        publisher_type=PublisherType.OCCULT,
        pages=666,
        catalog_number="FK-666",
        condition=BookCondition.BURNED,
        danger_level=5,
        original_book_title="Original Forbidden",
        original_book_author="Original Censored",
        suppression_history="Banned by the Church in 1925, copies burned. Rediscovered in private collection 2015.",
    )

    assert "Banned by the Church" in book.suppression_history
    assert "burned" in book.suppression_history
    assert "Rediscovered" in book.suppression_history
