import pytest
from app.agents.enums.gnosis_method import GnosisMethod

def test_members_and_values():
    expected = {
        "EXHAUSTION": "exhaustion",
        "ECSTASY": "ecstasy",
        "OBSESSION": "obsession",
        "SENSORY_OVERLOAD": "sensory_overload",
        "MEDITATION": "meditation",
        "CHAOS": "chaos",
    }
    for name, value in expected.items():
        member = GnosisMethod[name]
        assert member.value == value
        assert isinstance(member.value, str)

@pytest.mark.parametrize("value,member", [
    ("exhaustion", GnosisMethod.EXHAUSTION),
    ("ecstasy", GnosisMethod.ECSTASY),
    ("obsession", GnosisMethod.OBSESSION),
    ("sensory_overload", GnosisMethod.SENSORY_OVERLOAD),
    ("meditation", GnosisMethod.MEDITATION),
    ("chaos", GnosisMethod.CHAOS),
])
def test_lookup_by_value(value, member):
    assert GnosisMethod(value) is member

def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        GnosisMethod("invalid")

def test_values_are_unique():
    values = [m.value for m in GnosisMethod]
    assert len(values) == len(set(values))

