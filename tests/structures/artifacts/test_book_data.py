import pytest
from pydantic import ValidationError

from app.structures.artifacts.book_data import BookData, BookDataPageCollection
from app.structures.enums.book_condition import BookCondition
from app.structures.enums.publisher_type import PublisherType


def minimal_book_payload():
    # pick a concrete enum member without relying on member names
    publisher_type = list(PublisherType)[0]
    condition = list(BookCondition)[0]
    return {
        "title": "Test Title",
        "author": "Jane Doe",
        "year": 2020,
        "publisher": "Test Publisher",
        "publisher_type": publisher_type,
        "pages": 123,
        "catalog_number": "RA-0001",
        "condition": condition,
        "danger_level": 2,
    }


def test_bookdata_creates_with_minimal_required_fields():
    payload = minimal_book_payload()
    b = BookData(**payload)
    assert b.title == payload["title"]
    assert b.author == payload["author"]
    assert b.year == payload["year"]
    assert b.publisher == payload["publisher"]
    assert b.pages == payload["pages"]
    assert b.catalog_number == payload["catalog_number"]
    assert b.danger_level == payload["danger_level"]


def test_bookdata_defaults_and_collections():
    b = BookData(**minimal_book_payload())
    # defaults
    assert b.edition == "1st"
    assert b.language == "English"
    # list defaults are empty lists
    assert isinstance(b.tags, list) and b.tags == []
    assert isinstance(b.related_works, list) and b.related_works == []


def test_bookdata_optional_fields_set_correctly():
    payload = minimal_book_payload()
    payload.update(
        {
            "subtitle": "A Subtitle",
            "author_credentials": "PhD",
            "isbn": "978-1-23456-789-7",
            "acquisition_date": "2025-01-01",
            "acquisition_notes": "Donated",
            "translated_from": "Latin",
            "translator": "John Translator",
            "tags": ["occult", "history"],
            "abstract": "Short abstract",
            "notable_quote": "Something memorable",
            "suppression_history": "Censored historically",
            "related_works": ["Other Book"],
        }
    )
    b = BookData(**payload)
    assert b.subtitle == "A Subtitle"
    assert b.tags == ["occult", "history"]
    assert b.related_works == ["Other Book"]


def test_bookdata_missing_required_raises_validation_error():
    payload = minimal_book_payload()
    payload.pop("title")
    with pytest.raises(ValidationError):
        BookData(**payload)


def test_bookdata_invalid_year_raises_validation_error():
    payload = minimal_book_payload()
    payload["year"] = "not-a-year"
    with pytest.raises(ValidationError):
        BookData(**payload)


def test_bookdatapagecollection_valid_and_missing_fields():
    pc = BookDataPageCollection(page_1="First page text", page_2="Second page text")
    assert pc.page_1.startswith("First")
    assert pc.page_2.startswith("Second")

    with pytest.raises(ValidationError):
        BookDataPageCollection(page_1="Only one page")
