import enum

import pytest

from app.agents.enums.book_condition import BookCondition

def test_members_and_values():
    expected = {
        "PRISTINE": "pristine",
        "GOOD": "good",
        "WORN": "worn",
        "DAMAGED": "damaged",
        "FRAGMENTARY": "fragmentary",
        "RECONSTRUCTED": "reconstructed from copies",
        "BURNED": "partially burned",
    }
    for name, value in expected.items():
        member = getattr(BookCondition, name)
        assert member.value == value

@pytest.mark.parametrize("value,member", [
    ("pristine", BookCondition.PRISTINE),
    ("damaged", BookCondition.DAMAGED),
    ("fragmentary", BookCondition.FRAGMENTARY),
    ("reconstructed from copies", BookCondition.RECONSTRUCTED),
    ("partially burned", BookCondition.BURNED),
])

def test_lookup_by_value(value, member):
    assert BookCondition(value) is member

def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        BookCondition("smells")

def test_values_are_unique():
    values = [m.value for m in BookCondition]
    assert len(values) == len(set(values))

def test_enum_members_are_enum_instances():
    assert isinstance(BookCondition.DAMAGED, enum.Enum)
    assert isinstance(BookCondition.FRAGMENTARY, enum.Enum)
