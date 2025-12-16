import pytest
from pydantic import ValidationError
import datetime

from app.structures.concepts.biographical_event import BiographicalEvent


def test_create_valid_model_with_date():
    m = BiographicalEvent(
        date=datetime.date(1998, 6, 15), description="Graduated from college"
    )
    assert isinstance(m, BiographicalEvent)
    assert m.date == datetime.date(1998, 6, 15)
    assert m.description == "Graduated from college"
    assert m.year is None
    assert m.approximate_date is None
    assert m.category is None
    assert m.tags == []
    assert m.model_dump(exclude_none=True, exclude_unset=True)


def test_create_valid_model_with_year():
    m = BiographicalEvent(year=1998, description="Started new job")
    assert m.year == 1998
    assert m.date is None


def test_create_valid_model_with_approximate_date():
    m = BiographicalEvent(
        approximate_date="Summer 1998", description="Moved to Portland"
    )
    assert m.approximate_date == "Summer 1998"
    assert m.date is None
    assert m.year is None


def test_with_all_optional_fields():
    m = BiographicalEvent(
        date=datetime.date(1998, 6, 15),
        year=1998,
        approximate_date="Mid-June 1998",
        description="Important life event",
        category="career",
        emotional_weight=0.85,
        tags=["major", "career", "life-changing"],
    )
    assert m.category == "career"
    assert m.emotional_weight == 0.85
    assert len(m.tags) == 3
    assert "major" in m.tags


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        BiographicalEvent()


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        BiographicalEvent(description=123)


def test_invalid_date_type_raises():
    with pytest.raises(ValidationError):
        BiographicalEvent(date="not a date", description="test")


def test_emotional_weight_validation():
    with pytest.raises(ValidationError):
        BiographicalEvent(description="test", emotional_weight=1.5)

    with pytest.raises(ValidationError):
        BiographicalEvent(description="test", emotional_weight=-0.1)


def test_valid_emotional_weight_boundaries():
    m1 = BiographicalEvent(description="test", emotional_weight=0.0)
    assert m1.emotional_weight == 0.0

    m2 = BiographicalEvent(description="test", emotional_weight=1.0)
    assert m2.emotional_weight == 1.0


def test_default_tags_empty_list():
    m = BiographicalEvent(description="test")
    assert m.tags == []
    assert isinstance(m.tags, list)


def test_field_descriptions():
    fields = getattr(BiographicalEvent, "model_fields", None)
    assert fields is not None
