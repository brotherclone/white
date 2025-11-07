import enum

import pytest

from app.structures.enums.white_facet import WhiteFacet

EXPECTED = {
    "CATEGORICAL": "categorical",
    "RELATIONAL": "relational",
    "PROCEDURAL": "procedural",
    "COMPARATIVE": "comparative",
    "ARCHETYPAL": "archetypal",
    "TECHNICAL": "technical",
    "PHENOMENOLOGICAL": "phenomenological",
}


def test_members_and_values():
    assert set(WhiteFacet.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(WhiteFacet, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in WhiteFacet:
        assert isinstance(member, WhiteFacet)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("categorical", WhiteFacet.CATEGORICAL),
        ("relational", WhiteFacet.RELATIONAL),
        ("procedural", WhiteFacet.PROCEDURAL),
    ],
)
def test_lookup_by_value(value, member):
    assert WhiteFacet(value) is member


def test_lookup_by_name():
    assert WhiteFacet["CATEGORICAL"] is WhiteFacet.CATEGORICAL
    assert WhiteFacet["RELATIONAL"] is WhiteFacet.RELATIONAL


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        WhiteFacet("mystical")


def test_values_are_unique():
    values = [m.value for m in WhiteFacet]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(WhiteFacet.CATEGORICAL, enum.Enum)
    assert isinstance(WhiteFacet.PHENOMENOLOGICAL, enum.Enum)
