import pytest
from app.agents.enums.noise_type import NoiseType

def test_members_and_values():
    expected = {
        "WHITE": "white",
        "PINK": "pink",
        "BROWN": "brown",
        "BLUE": "blue",
        "VIOLET": "violet",
        "GREY": "grey",
    }
    for name, value in expected.items():
        member = NoiseType[name]
        assert member.value == value
        assert isinstance(member.value, str)

@pytest.mark.parametrize("value,member", [
    ("white", NoiseType.WHITE),
    ("pink", NoiseType.PINK),
    ("brown", NoiseType.BROWN),
    ("blue", NoiseType.BLUE),
    ("violet", NoiseType.VIOLET),
    ("grey", NoiseType.GREY),
])
def test_lookup_by_value(value, member):
    assert NoiseType(value) is member

def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        NoiseType("invalid")

def test_values_are_unique():
    values = [m.value for m in NoiseType]
    assert len(values) == len(set(values))

