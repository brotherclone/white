import enum
import pytest

from app.structures.enums.biographical_timeline_detail_level import (
    BiographicalTimelineDetailLevel,
)

EXPECTED = {
    "HIGH": "high",
    "MEDIUM": "medium",
    "LOW": "low",
    "MINIMAL": "minimal",
}


def test_members_and_values():
    assert set(BiographicalTimelineDetailLevel.__members__.keys()) == set(
        EXPECTED.keys()
    )
    for name, expected_value in EXPECTED.items():
        member = getattr(BiographicalTimelineDetailLevel, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in BiographicalTimelineDetailLevel:
        assert isinstance(member, BiographicalTimelineDetailLevel)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("high", BiographicalTimelineDetailLevel.HIGH),
        ("medium", BiographicalTimelineDetailLevel.MEDIUM),
        ("low", BiographicalTimelineDetailLevel.LOW),
        ("minimal", BiographicalTimelineDetailLevel.MINIMAL),
    ],
)
def test_lookup_by_value(value, member):
    assert BiographicalTimelineDetailLevel(value) is member


def test_lookup_by_name():
    assert (
        BiographicalTimelineDetailLevel["HIGH"] is BiographicalTimelineDetailLevel.HIGH
    )
    assert (
        BiographicalTimelineDetailLevel["MINIMAL"]
        is BiographicalTimelineDetailLevel.MINIMAL
    )


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        BiographicalTimelineDetailLevel("invalid_level")


def test_values_are_unique():
    values = [m.value for m in BiographicalTimelineDetailLevel]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(BiographicalTimelineDetailLevel.HIGH, enum.Enum)
    assert isinstance(BiographicalTimelineDetailLevel.MINIMAL, enum.Enum)
