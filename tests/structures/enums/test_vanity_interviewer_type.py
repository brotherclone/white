import enum
import pytest

from app.structures.enums.vanity_interviewer_type import VanityInterviewerType

EXPECTED = {
    "HOSTILE_SKEPTICAL": "hostile_skeptical",
    "EXPERIMENTAL_PURIST": "experimental_purist",
    "VANITY_PRESSING_FAN": "vanity_pressing_fan",
    "EARNEST_BUT_WRONG": "earnest_but_wrong",
}


def test_members_and_values():
    assert set(VanityInterviewerType.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(VanityInterviewerType, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in VanityInterviewerType:
        assert isinstance(member, VanityInterviewerType)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("hostile_skeptical", VanityInterviewerType.HOSTILE_SKEPTICAL),
        ("experimental_purist", VanityInterviewerType.EXPERIMENTAL_PURIST),
        ("vanity_pressing_fan", VanityInterviewerType.VANITY_PRESSING_FAN),
        ("earnest_but_wrong", VanityInterviewerType.EARNEST_BUT_WRONG),
    ],
)
def test_lookup_by_value(value, member):
    assert VanityInterviewerType(value) is member


def test_lookup_by_name():
    assert (
        VanityInterviewerType["HOSTILE_SKEPTICAL"]
        is VanityInterviewerType.HOSTILE_SKEPTICAL
    )
    assert (
        VanityInterviewerType["EXPERIMENTAL_PURIST"]
        is VanityInterviewerType.EXPERIMENTAL_PURIST
    )


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        VanityInterviewerType("unknown")


def test_values_are_unique():
    values = [m.value for m in VanityInterviewerType]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(VanityInterviewerType.VANITY_PRESSING_FAN, enum.Enum)
    assert isinstance(VanityInterviewerType.EARNEST_BUT_WRONG, enum.Enum)
