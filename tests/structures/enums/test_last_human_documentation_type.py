import enum
import pytest

from app.structures.enums.last_human_documentation_type import (
    LastHumanDocumentationType,
)

EXPECTED = {
    "DEATH": "death",
    "DISPLACEMENT": "displacement",
    "ADAPTATION_FAILURE": "adaptation_failure",
    "RESILIENCE": "resilience",
    "WITNESS": "witness",
}


def test_members_and_values():
    assert set(LastHumanDocumentationType.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(LastHumanDocumentationType, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in LastHumanDocumentationType:
        assert isinstance(member, LastHumanDocumentationType)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("death", LastHumanDocumentationType.DEATH),
        ("displacement", LastHumanDocumentationType.DISPLACEMENT),
        ("adaptation_failure", LastHumanDocumentationType.ADAPTATION_FAILURE),
        ("resilience", LastHumanDocumentationType.RESILIENCE),
        ("witness", LastHumanDocumentationType.WITNESS),
    ],
)
def test_lookup_by_value(value, member):
    assert LastHumanDocumentationType(value) is member


def test_lookup_by_name():
    assert LastHumanDocumentationType["DEATH"] is LastHumanDocumentationType.DEATH
    assert (
        LastHumanDocumentationType["RESILIENCE"]
        is LastHumanDocumentationType.RESILIENCE
    )


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        LastHumanDocumentationType("invalid_type")


def test_values_are_unique():
    values = [m.value for m in LastHumanDocumentationType]
    assert len(values) == len(set(values))
