import enum
import pytest

from app.structures.enums.publisher_type import PublisherType

EXPECTED = {
    "UNIVERSITY": "University imprint",
    "OCCULT": "Occult cottage industry publisher",
    "SAMIZDAT": "Samizdat dead drop",
    "VANITY": "Vanity press directly from author",
    "LOST": "Previously deemed lost",
    "GOVERNMENT": "Declassified document from FOIA request",
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


@pytest.mark.parametrize("value,member", [
    ("University imprint", PublisherType.UNIVERSITY),
    ("Occult cottage industry publisher", PublisherType.OCCULT),
    ("Samizdat dead drop", PublisherType.SAMIZDAT),
    ("Vanity press directly from author", PublisherType.VANITY),
    ("Previously deemed lost", PublisherType.LOST),
    ("Declassified document from FOIA request", PublisherType.GOVERNMENT),
])
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