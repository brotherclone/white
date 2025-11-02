import enum
import pytest

from app.structures.enums.noise_type import NoiseType

EXPECTED = {
    "WHITE": "white",
    "PINK": "pink",
    "BROWN": "brown",
    "BLUE": "blue",
    "VIOLET": "violet",
    "GREY": "grey",
}


def test_members_and_values():
    assert set(NoiseType.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(NoiseType, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in NoiseType:
        assert isinstance(member, NoiseType)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize("value,member", [
    ("white", NoiseType.WHITE),
    ("pink", NoiseType.PINK),
    ("brown", NoiseType.BROWN),
    ("blue", NoiseType.BLUE),
    ("violet", NoiseType.VIOLET),
    ("grey", NoiseType.GREY),
])
def test_lookup_by_value(value, member):
    assert NoiseType(value) is member


def test_lookup_by_name():
    assert NoiseType["WHITE"] is NoiseType.WHITE
    assert NoiseType["PINK"] is NoiseType.PINK


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        NoiseType("loud")


def test_values_are_unique():
    values = [m.value for m in NoiseType]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(NoiseType.WHITE, enum.Enum)
    assert isinstance(NoiseType.GREY, enum.Enum)
