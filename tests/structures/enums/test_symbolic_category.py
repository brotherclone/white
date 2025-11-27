import enum

import pytest

from app.structures.enums.symbolic_object_category import SymbolicObjectCategory

EXPECTED = {
    "CIRCULAR_TIME": "circular_time",
    "INFORMATION_ARTIFACTS": "information_artifacts",
    "LIMINAL_OBJECTS": "liminal_objects",
    "PSYCHOGEOGRAPHIC": "psychogeographic",
}


def test_members_and_values():
    assert set(SymbolicObjectCategory.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(SymbolicObjectCategory, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in SymbolicObjectCategory:
        assert isinstance(member, SymbolicObjectCategory)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("circular_time", SymbolicObjectCategory.CIRCULAR_TIME),
        ("information_artifacts", SymbolicObjectCategory.INFORMATION_ARTIFACTS),
        ("liminal_objects", SymbolicObjectCategory.LIMINAL_OBJECTS),
        ("psychogeographic", SymbolicObjectCategory.PSYCHOGEOGRAPHIC),
    ],
)
def test_lookup_by_value(value, member):
    assert SymbolicObjectCategory(value) is member


def test_lookup_by_name():
    assert (
        SymbolicObjectCategory["INFORMATION_ARTIFACTS"]
        is SymbolicObjectCategory.INFORMATION_ARTIFACTS
    )
    assert (
        SymbolicObjectCategory["PSYCHOGEOGRAPHIC"]
        is SymbolicObjectCategory.PSYCHOGEOGRAPHIC
    )


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        SymbolicObjectCategory("unknown")


def test_values_are_unique():
    values = [m.value for m in SymbolicObjectCategory]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(SymbolicObjectCategory.CIRCULAR_TIME, enum.Enum)
    assert isinstance(SymbolicObjectCategory.LIMINAL_OBJECTS, enum.Enum)
