import pytest
from pydantic import ValidationError

from app.structures.concepts.alternate_life_detail import AlternateLifeDetail


def test_create_valid_model():
    m = AlternateLifeDetail(category="career", detail="Working at a bookstore")
    assert isinstance(m, AlternateLifeDetail)
    assert m.category == "career"
    assert m.detail == "Working at a bookstore"
    assert m.sensory_elements is None
    assert m.model_dump(exclude_none=True, exclude_unset=True)


def test_with_sensory_elements():
    m = AlternateLifeDetail(
        category="location",
        detail="Living in Portland",
        sensory_elements=["rain", "coffee shops", "Powell's Books"],
    )
    assert len(m.sensory_elements) == 3
    assert "rain" in m.sensory_elements


def test_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        AlternateLifeDetail()

    with pytest.raises(ValidationError):
        AlternateLifeDetail(category="career")

    with pytest.raises(ValidationError):
        AlternateLifeDetail(detail="test")


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        AlternateLifeDetail(category=123, detail="test")


def test_invalid_category_raises():
    with pytest.raises(ValidationError):
        AlternateLifeDetail(category="invalid_category", detail="test")


def test_valid_categories():
    valid_categories = [
        "career",
        "relationship",
        "location",
        "creative",
        "daily_routine",
        "outcome",
    ]
    for cat in valid_categories:
        m = AlternateLifeDetail(category=cat, detail="test detail")
        assert m.category == cat


def test_field_descriptions():
    fields = getattr(AlternateLifeDetail, "model_fields", None)
    assert fields is not None
