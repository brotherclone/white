import enum

import pytest

from app.structures.enums.sigil_type import SigilType

EXPECTED = {
    "WORD_METHOD": "word_method",
    "PICTORIAL": "pictorial",
    "MANTRIC": "mantric",
    "ALPHABET_OF_DESIRE": "alphabet_of_desire",
}


def test_members_and_values():
    assert set(SigilType.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(SigilType, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in SigilType:
        assert isinstance(member, SigilType)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("word_method", SigilType.WORD_METHOD),
        ("pictorial", SigilType.PICTORIAL),
        ("mantric", SigilType.MANTRIC),
        ("alphabet_of_desire", SigilType.ALPHABET_OF_DESIRE),
    ],
)
def test_lookup_by_value(value, member):
    assert SigilType(value) is member


def test_lookup_by_name():
    assert SigilType["WORD_METHOD"] is SigilType.WORD_METHOD
    assert SigilType["PICTORIAL"] is SigilType.PICTORIAL


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        SigilType("unknown")


def test_values_are_unique():
    values = [m.value for m in SigilType]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(SigilType.WORD_METHOD, enum.Enum)
    assert isinstance(SigilType.ALPHABET_OF_DESIRE, enum.Enum)
