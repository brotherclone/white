import pytest
from pydantic import ValidationError

from app.structures.concepts.divergence_point import DivergencePoint


def test_create_valid_model():
    m = DivergencePoint(
        when="After graduating college in 1997",
        what_changed="Took the Greyhound to Portland instead of returning to NJ",
        why_plausible="Had been offered a job at Powell's Books",
    )
    assert isinstance(m, DivergencePoint)
    assert m.when == "After graduating college in 1997"
    assert m.what_changed == "Took the Greyhound to Portland instead of returning to NJ"
    assert m.why_plausible == "Had been offered a job at Powell's Books"
    assert m.model_dump(exclude_none=True, exclude_unset=True)


def test_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        DivergencePoint()

    with pytest.raises(ValidationError):
        DivergencePoint(when="test")

    with pytest.raises(ValidationError):
        DivergencePoint(when="test", what_changed="test")

    with pytest.raises(ValidationError):
        DivergencePoint(what_changed="test", why_plausible="test")


def test_wrong_type_raises():
    with pytest.raises(ValidationError):
        DivergencePoint(when=123, what_changed="test", why_plausible="test")

    with pytest.raises(ValidationError):
        DivergencePoint(when="test", what_changed=123, why_plausible="test")

    with pytest.raises(ValidationError):
        DivergencePoint(when="test", what_changed="test", why_plausible=123)


def test_empty_strings_allowed():
    m = DivergencePoint(when="", what_changed="", why_plausible="")
    assert m.when == ""
    assert m.what_changed == ""
    assert m.why_plausible == ""


def test_field_descriptions():
    fields = getattr(DivergencePoint, "model_fields", None)
    assert fields is not None
