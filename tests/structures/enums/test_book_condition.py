import enum
import pytest

from app.structures.enums.book_condition import BookCondition

EXPECTED = {
    "PRISTINE": "pristine",
    "GOOD": "good",
    "WORN": "worn",
    "DAMAGED": "damaged",
    "FRAGMENTARY": "fragmentary",
    "RECONSTRUCTED": "reconstructed from copies",
    "BURNED": "partially burned",
}

def test_members_and_values():
    assert set(BookCondition.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(BookCondition, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)

def test_members_are_str_and_enum_and_compare_to_value():
    for member in BookCondition:
        assert isinstance(member, BookCondition)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value

@pytest.mark.parametrize("value,member", [
    ("pristine", BookCondition.PRISTINE),
    ("damaged", BookCondition.DAMAGED),
    ("fragmentary", BookCondition.FRAGMENTARY),
    ("reconstructed from copies", BookCondition.RECONSTRUCTED),
    ("partially burned", BookCondition.BURNED),
])
def test_lookup_by_value(value, member):
    assert BookCondition(value) is member

def test_lookup_by_name():
    assert BookCondition["PRISTINE"] is BookCondition.PRISTINE
    assert BookCondition["GOOD"] is BookCondition.GOOD

def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        BookCondition("pickled")

def test_values_are_unique():
    values = [m.value for m in BookCondition]
    assert len(values) == len(set(values))

def test_enum_members_are_enum_instances():
    assert isinstance(BookCondition.BURNED, enum.Enum)
    assert isinstance(BookCondition.RECONSTRUCTED, enum.Enum)