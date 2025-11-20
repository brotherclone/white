from pathlib import Path

import yaml

from app.structures.artifacts.book_artifact import BookArtifact
from tests.mocks.mock_tests.mock_utils import normalize_book_data_enums


def test_book_artifact_mocks():
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "mocks"
        / "red_book_artifact_mock.yml"
    )
    with path.open("r") as f:
        data = yaml.safe_load(f)
    data = normalize_book_data_enums(data)
    book = BookArtifact(**data)
    assert isinstance(book, BookArtifact)
    assert book.publisher_type is not None
    assert book.condition is not None
    assert book.publisher_type.value == "vanity"
    assert book.condition.value == "good"
    assert book.title == "Mock Book Title"
    assert book.author == "John Doe"
    assert book.publisher == "Mock Publishing House"
    assert book.pages == 250
    assert book.author_credentials == "PhD in Mock Studies"
    assert book.year == 2025
    assert book.isbn == "123-4-567-89012-3"
    assert book.catalog_number == "MOCK-CAT-001"
    assert book.acquisition_date == "2025-01-01"
    assert book.acquisition_notes == "Acquired for testing purposes."
    assert book.language == "English"
    assert book.tags == ["mock", "test", "sample"]
    assert book.danger_level == 1
    assert book.abstract == "An abstract for the mock book used in testing."
    assert book.chain_artifact_type == "book"
    assert book.thread_id == "412345"


def test_reaction_book_mocks():
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "mocks"
        / "red_reaction_book_data_mock.yml"
    )
    with path.open("r") as f:
        data = yaml.safe_load(f)
    data = normalize_book_data_enums(data)
    book = BookArtifact(**data)
    assert isinstance(book, BookArtifact)
    assert book.title == "Mock Book Title"
    assert book.author == "John Doe"
    assert book.publisher == "Mock Publishing House"
    assert book.publisher_type.value == "vanity"
    assert book.pages == 250
    assert book.author_credentials == "PhD in Mock Studies"
    assert book.year == 2025
    assert book.isbn == "123-4-567-89012-3"
    assert book.catalog_number == "MOCK-CAT-001"
    assert book.acquisition_date == "2025-01-01"
