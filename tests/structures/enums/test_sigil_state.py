import enum
import pytest

from app.structures.enums.sigil_state import SigilState

EXPECTED = {
    "CREATED": "created",
    "AWAITING_CHARGE": "awaiting charge",
    "CHARGING": "charging",
    "CHARGED": "charged",
    "BURIED": "buried",
    "UNKNOWN": "unknown",
}


def test_members_and_values():
    assert set(SigilState.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(SigilState, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in SigilState:
        assert isinstance(member, SigilState)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize("value,member", [
    ("created", SigilState.CREATED),
    ("awaiting charge", SigilState.AWAITING_CHARGE),
    ("charging", SigilState.CHARGING),
    ("charged", SigilState.CHARGED),
    ("buried", SigilState.BURIED),
    ("unknown", SigilState.UNKNOWN),
])
def test_lookup_by_value(value, member):
    assert SigilState(value) is member


def test_lookup_by_name():
    assert SigilState["CREATED"] is SigilState.CREATED
    assert SigilState["CHARGED"] is SigilState.CHARGED


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        SigilState("erased")


def test_values_are_unique():
    values = [m.value for m in SigilState]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(SigilState.CREATED, enum.Enum)
    assert isinstance(SigilState.UNKNOWN, enum.Enum)
