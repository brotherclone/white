import enum

import pytest

from app.structures.enums.publisher_type import PublisherType

EXPECTED = {
    "UNIVERSITY": "university",
    "OCCULT": "occult",
    "SAMIZDAT": "samizdat",
    "VANITY": "vanity",
    "LOST": "lost",
    "GOVERNMENT": "government",
}


def test_members_and_values():
    assert set(PublisherType.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(PublisherType, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in PublisherType:
        assert isinstance(member, PublisherType)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("university", PublisherType.UNIVERSITY),
        ("occult", PublisherType.OCCULT),
        ("samizdat", PublisherType.SAMIZDAT),
        ("vanity", PublisherType.VANITY),
        ("lost", PublisherType.LOST),
        ("government", PublisherType.GOVERNMENT),
    ],
)
def test_lookup_by_value(value, member):
    assert PublisherType(value) is member


def test_lookup_by_name():
    assert PublisherType["LOST"] is PublisherType.LOST
    assert PublisherType["GOVERNMENT"] is PublisherType.GOVERNMENT


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        PublisherType("random house")


def test_values_are_unique():
    values = [m.value for m in PublisherType]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(PublisherType.SAMIZDAT, enum.Enum)
    assert isinstance(PublisherType.OCCULT, enum.Enum)
