import pytest
from app.agents.enums.sigil_state import SigilState

def test_members_and_values():
    expected = {
        "CREATED": "created",
        "AWAITING_CHARGE": "awaiting charge",
        "CHARGING": "charging",
        "CHARGED": "charged",
        "BURIED": "buried",
        "UNKNOWN": "unknown",
    }
    for name, value in expected.items():
        member = SigilState[name]
        assert member.value == value
        assert isinstance(member.value, str)

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

def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        SigilState("invalid")

def test_values_are_unique():
    values = [m.value for m in SigilState]
    assert len(values) == len(set(values))

