from app.structures.artifacts.reaction_book_artifact import ReactionBookChainArtifact
from app.structures.enums.book_condition import BookCondition
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.publisher_type import PublisherType


def test_reaction_book_artifact_defaults():
    """Test ReactionBookChainArtifact creation with required fields."""
    book = ReactionBookChainArtifact(
        thread_id="test-thread",
        title="The Cosmic Trigger",
        author="Robert Anton Wilson",
        year=1977,
        publisher="And/Or Press",
        publisher_type=PublisherType.OCCULT,
        pages=267,
        catalog_number="RAW-001",
        condition=BookCondition.GOOD,
        danger_level=3,
        original_book_title="The Original Cosmic Trigger",
        original_book_author="RAW",
    )

    assert book.thread_id == "test-thread"
    assert book.title == "The Cosmic Trigger"
    assert book.author == "Robert Anton Wilson"
    assert book.chain_artifact_type == ChainArtifactType.BOOK
    assert book.chain_artifact_file_type == ChainArtifactFileType.YML
    assert book.original_book_title == "The Original Cosmic Trigger"
    assert book.original_book_author == "RAW"


def test_reaction_book_artifact_optional_fields():
    """Test ReactionBookChainArtifact with optional fields."""
    book = ReactionBookChainArtifact(
        thread_id="test-thread",
        title="Prometheus Rising",
        subtitle="A Guide to Consciousness",
        author="Robert Anton Wilson",
        author_credentials="Discordian Pope",
        year=1983,
        publisher="New Falcon Publications",
        publisher_type=PublisherType.SAMIZDAT,
        edition="2nd",
        pages=280,
        isbn="978-1561840564",
        catalog_number="RAW-002",
        condition=BookCondition.WORN,
        acquisition_date="2024-01-15",
        acquisition_notes="Found at occult bookstore",
        language="English",
        tags=["psychology", "consciousness", "occult"],
        danger_level=4,
        abstract="An exploration of human consciousness through the lens of eight circuit model.",
        notable_quote="Whatever the Thinker thinks, the Prover proves.",
        suppression_history="Briefly banned in certain religious communities",
        related_works=["Cosmic Trigger", "Illuminatus!"],
        excerpts=["Reality is what you can get away with."],
        original_book_title="Prometheus Rising Original",
        original_book_author="RAW",
    )

    assert book.subtitle == "A Guide to Consciousness"
    assert book.author_credentials == "Discordian Pope"
    assert book.isbn == "978-1561840564"
    assert len(book.tags) == 3
    assert book.notable_quote == "Whatever the Thinker thinks, the Prover proves."
    assert len(book.excerpts) == 1


def test_reaction_book_artifact_flatten():
    """Test that flatten() returns correct dictionary representation."""
    book = ReactionBookChainArtifact(
        thread_id="test-thread",
        title="Flatten Test",
        author="Test Author",
        year=2025,
        publisher="Test Pub",
        publisher_type=PublisherType.VANITY,
        pages=150,
        catalog_number="FLAT-001",
        condition=BookCondition.DAMAGED,
        danger_level=2,
        tags=["test", "flatten"],
        original_book_title="Original Flatten",
        original_book_author="Original",
    )

    flattened = book.flatten()

    assert flattened["thread_id"] == "test-thread"
    assert flattened["title"] == "Flatten Test"
    assert flattened["chain_artifact_type"] == ChainArtifactType.BOOK.value
    assert flattened["chain_artifact_file_type"] == ChainArtifactFileType.YML.value
    assert flattened["tags"] == ["test", "flatten"]
    assert flattened["original_book_title"] == "Original Flatten"
    assert flattened["original_book_author"] == "Original"


def test_reaction_book_artifact_enum_values():
    """Test that enum fields work correctly."""
    book = ReactionBookChainArtifact(
        thread_id="test",
        title="Enum Test",
        author="Author",
        year=2025,
        publisher="Pub",
        publisher_type=PublisherType.GOVERNMENT,
        pages=200,
        catalog_number="ENUM-001",
        condition=BookCondition.FRAGMENTARY,
        danger_level=5,
        original_book_title="Original",
        original_book_author="Orig",
    )

    # Verify enum types
    assert isinstance(book.publisher_type, PublisherType)
    assert isinstance(book.condition, BookCondition)
    assert book.publisher_type == PublisherType.GOVERNMENT
    assert book.condition == BookCondition.FRAGMENTARY

    # Verify flattened enum values are strings
    flattened = book.flatten()
    assert flattened["publisher_type"] == "government"
    assert flattened["condition"] == "fragmentary"
