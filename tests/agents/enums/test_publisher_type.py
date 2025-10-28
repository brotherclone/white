import enum
import pytest

from app.agents.enums.publisher_type import PublisherType

def test_members_and_values():
    expected = {
        "UNIVERSITY": "University imprint",
        "OCCULT": "Occult cottage industry publisher",
        "SAMIZDAT": "Samizdat dead drop",
        "VANITY": "Vanity press directly from author",
        "LOST": "Previously deemed lost",
        "GOVERNMENT": "Declassified document from FOIA request",
    }
    for name, value in expected.items():
        member = getattr(PublisherType, name)
        assert member.value == value

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

def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        PublisherType("barnes and fucking noble")

def test_values_are_unique():
    values = [m.value for m in PublisherType]
    assert len(values) == len(set(values))

def test_enum_members_are_enum_instances():
    assert isinstance(PublisherType.SAMIZDAT, enum.Enum)
    assert isinstance(PublisherType.LOST, enum.Enum)
