import enum
import pytest

from app.structures.enums.temporal_relatioship import TemporalRelationship

EXPECTED = {
    "ACROSS": "spans_across",
    "BLEED_IN": "bleeds_in",
    "BLEED_OUT": "bleeds_out",
    "CONTAINED": "contained",
    "MATCH": "exact_match",
    "UNKNOWN": "unknown",
}


def test_members_and_values():
    assert set(TemporalRelationship.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(TemporalRelationship, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in TemporalRelationship:
        assert isinstance(member, TemporalRelationship)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize("value,member", [
    ("spans_across", TemporalRelationship.ACROSS),
    ("bleeds_in", TemporalRelationship.BLEED_IN),
    ("bleeds_out", TemporalRelationship.BLEED_OUT),
    ("contained", TemporalRelationship.CONTAINED),
    ("exact_match", TemporalRelationship.MATCH),
    ("unknown", TemporalRelationship.UNKNOWN),
])
def test_lookup_by_value(value, member):
    assert TemporalRelationship(value) is member


def test_lookup_by_name():
    assert TemporalRelationship["ACROSS"] is TemporalRelationship.ACROSS
    assert TemporalRelationship["MATCH"] is TemporalRelationship.MATCH


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        TemporalRelationship("later")


def test_values_are_unique():
    values = [m.value for m in TemporalRelationship]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(TemporalRelationship.ACROSS, enum.Enum)
    assert isinstance(TemporalRelationship.UNKNOWN, enum.Enum)

