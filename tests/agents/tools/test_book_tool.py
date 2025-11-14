import random

from app.agents.tools.book_tool import BookMaker
from app.structures.artifacts.book_data import BookData
from app.structures.enums.book_condition import BookCondition
from app.structures.enums.book_genre import BookGenre
from app.structures.enums.publisher_type import PublisherType


def test_select_genre_uses_choices(monkeypatch):
    monkeypatch.setattr(
        random, "choices", lambda genres, weights=None: [BookGenre.SCIFI]
    )
    assert BookMaker.select_genre() is BookGenre.SCIFI


def test_generate_title_with_and_without_colon():
    # SCIFI branch: explicit split when ':' present
    title, subtitle = BookMaker.generate_title("Stars: A Voyage", BookGenre.SCIFI)
    assert title == "Stars"
    assert subtitle == "A Voyage"

    # Non-SCIFI branch: may return subtitle or None; ensure behavior for no colon
    title2, subtitle2 = BookMaker.generate_title("Lonely Planet", BookGenre.OCCULT)
    assert isinstance(title2, str)
    assert (subtitle2 is None) or isinstance(subtitle2, str)


def test_generate_author_initials_and_credentials(monkeypatch):
    # Make author lists deterministic (single entries)
    monkeypatch.setattr(
        BookMaker, "get_authors_for_genre", lambda g: (["Alice"], ["Smith"])
    )
    # Ensure initials and credentials branches trigger (random.random < thresholds)
    monkeypatch.setattr(random, "random", lambda: 0.05)
    # deterministic choice (pick first element)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])

    name, creds = BookMaker.generate_author(BookGenre.OCCULT)
    # Initial should have been converted to initial form: "A. Smith"
    assert name.startswith("A.")
    assert "Smith" in name
    # Credentials should be populated for OCCULT given the low random value + deterministic choice
    assert creds is not None and isinstance(creds, str)


def test_generate_catalog_number_format():
    code = BookMaker.generate_catalog_number(2020, 12, BookGenre.NOIR)
    assert code == "RA-2020-NOR-0012"


def test_generate_random_book_properties(monkeypatch):
    # Generate a book with force_genre and fixed index, check key invariants
    book = BookMaker.generate_random_book(index=7, force_genre=BookGenre.SCIFI)

    # Catalog number ends with the zero-padded index
    assert book.catalog_number.endswith("0007")
    # Publisher should be one of the allowed publishers for the genre
    pubs = BookMaker.get_publishers_for_genre(BookGenre.SCIFI)
    assert book.publisher in pubs
    # Page count within expected range
    assert 127 <= book.pages <= 847
    # Tags should include the genre value
    assert BookGenre.SCIFI.value in book.tags


def test_format_bibliography_entry_and_card_catalog():
    sample = BookData(
        title="The Hidden Net",
        subtitle="Protocols of Desire",
        author="J. Doe",
        author_credentials="PhD",
        year=1999,
        publisher="Obscura Press",
        publisher_type=PublisherType.OCCULT,
        edition="1st",
        pages=256,
        isbn="978-1-2345-6789-0",
        catalog_number="RA-1999-XYZ-0001",
        condition=BookCondition.GOOD,
        acquisition_date="June 2005",
        acquisition_notes="Donated",
        language="English",
        translated_from=None,
        translator=None,
        tags=["hidden", "net", "science"],
        danger_level=2,
        abstract="An exploration of hidden networks.",
        notable_quote='"We are all nodes."',
        suppression_history=None,
        related_works=[],
    )

    bib = BookMaker.format_bibliography_entry(sample)
    assert "J. Doe" in bib
    assert "(1999)" in bib
    assert "*The Hidden Net" in bib
    assert "ISBN 978-1-2345-6789-0" in bib
    assert "[RA-1999-XYZ-0001]" in bib

    card = BookMaker.format_card_catalog(sample)
    # Basic sanity checks for expected sections
    assert "CATALOG #" in card
    assert "DANGER LEVEL" in card
    assert sample.catalog_number in card
