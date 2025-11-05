from copy import deepcopy
from pathlib import Path

import yaml
from hypothesis import given
from hypothesis import strategies as st

from app.structures.artifacts.book_artifact import BookArtifact
from app.structures.artifacts.book_data import BookData
from app.structures.artifacts.text_chain_artifact_file import TextChainArtifactFile
from app.structures.concepts.rainbow_table_color import (
    RainbowColorObjectionalMode,
    RainbowColorOntologicalMode,
    RainbowColorTemporalMode,
    RainbowTableColor,
)
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
    assert isinstance(book.book_data, BookData)
    assert isinstance(book.excerpts[0], TextChainArtifactFile)
    assert isinstance(book.excerpts[0].rainbow_color, RainbowTableColor)
    assert book.book_data.publisher_type is not None
    assert book.book_data.condition is not None
    assert book.book_data.publisher_type.value == "Vanity press directly from author"
    assert book.book_data.condition.value == "good"
    assert book.book_data.title == "Mock Book Title"
    assert book.book_data.author == "John Doe"
    assert book.book_data.publisher == "Mock Publishing House"
    assert book.book_data.pages == 250
    assert book.book_data.author_credentials == "PhD in Mock Studies"
    assert book.book_data.year == 2025
    assert book.book_data.isbn == "123-4-567-89012-3"
    assert book.book_data.catalog_number == "MOCK-CAT-001"
    assert book.book_data.acquisition_date == "2025-01-01"
    assert book.book_data.acquisition_notes == "Acquired for testing purposes."
    assert book.book_data.language == "English"
    assert book.book_data.tags == ["mock", "test", "sample"]
    assert book.book_data.danger_level == 1
    assert book.book_data.abstract == "An abstract for the mock book used in testing."
    assert book.chain_artifact_type == "REDChainArtifactBook"
    assert book.thread_id == "412345"
    assert book.excerpts[0].text_content == "This is a mock page for testing purposes."
    assert book.excerpts[0].chain_artifact_file_type.value == "md"
    assert book.excerpts[0].rainbow_color.mnemonic_character_value == "R"
    assert book.excerpts[0].rainbow_color.temporal_mode.value == "Past"
    assert book.excerpts[0].rainbow_color.objectional_mode.value == "Thing"
    assert book.excerpts[0].rainbow_color.ontological_mode[0].value == "Known"
    assert book.excerpts[0].artifact_id == "555"
    assert book.excerpts[0].artifact_name == "book"
    assert book.excerpts[0].file_name == "555_red_book.md"


@given(
    temporal=st.sampled_from([m.value for m in RainbowColorTemporalMode]),
    objectional=st.sampled_from([m.value for m in RainbowColorObjectionalMode]),
    ontological=st.lists(
        st.sampled_from([m.value for m in RainbowColorOntologicalMode]),
        min_size=1,
        max_size=2,
    ),
)
def test_book_artifact_trans_modes(temporal, objectional, ontological):
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "mocks"
        / "red_book_artifact_mock.yml"
    )
    with path.open("r") as f:
        base = yaml.safe_load(f)

    candidate = deepcopy(base)
    candidate = normalize_book_data_enums(candidate)
    rc = candidate["excerpts"][0]["rainbow_color"]
    rc["temporal_mode"] = temporal
    rc["objectional_mode"] = objectional
    rc["ontological_mode"] = ontological

    book = BookArtifact(**candidate)
    assert isinstance(book, BookArtifact)


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
    book = BookData(**data)
    assert isinstance(book, BookData)
    assert book.title == "Mock Book Title"
    assert book.author == "John Doe"
    assert book.publisher == "Mock Publishing House"
    assert book.publisher_type.value == "Vanity press directly from author"
    assert book.pages == 250
    assert book.author_credentials == "PhD in Mock Studies"
    assert book.year == 2025
    assert book.isbn == "123-4-567-89012-3"
    assert book.catalog_number == "MOCK-CAT-001"
    assert book.acquisition_date == "2025-01-01"


def test_reaction_book_page_1_mock():
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "mocks"
        / "red_reaction_book_page_1_mock.yml"
    )
    with path.open("r") as f:
        data = yaml.safe_load(f)
        page = TextChainArtifactFile(**data)
    assert isinstance(page, TextChainArtifactFile)
    assert page.text_content == "Page 1 mock"
    assert page.chain_artifact_file_type.value == "md"
    assert page.rainbow_color.mnemonic_character_value == "R"
    assert page.rainbow_color.temporal_mode.value == "Past"
    assert page.rainbow_color.objectional_mode.value == "Thing"
    assert page.rainbow_color.ontological_mode[0].value == "Known"
    assert page.artifact_id == "555"
    assert page.artifact_name == "book"
    assert page.file_name == "555_red_book.md"


def test_reaction_book_page_2_mock():
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "mocks"
        / "red_reaction_book_page_2_mock.yml"
    )
    with path.open("r") as f:
        data = yaml.safe_load(f)
        page = TextChainArtifactFile(**data)
    assert isinstance(page, TextChainArtifactFile)
    assert page.text_content == "Page 2 mock"
    assert page.chain_artifact_file_type.value == "md"
    assert page.rainbow_color.mnemonic_character_value == "R"
    assert page.rainbow_color.temporal_mode.value == "Past"
    assert page.rainbow_color.objectional_mode.value == "Thing"
    assert page.rainbow_color.ontological_mode[0].value == "Known"
    assert page.artifact_id == "444"
    assert page.artifact_name == "book"
    assert page.file_name == "444_red_book.md"
