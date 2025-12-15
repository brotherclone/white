import enum
import pytest

from app.structures.enums.gnosis_method import GnosisMethod

EXPECTED = {
    "EXHAUSTION": "exhaustion",
    "ECSTASY": "ecstasy",
    "OBSESSION": "obsession",
    "SENSORY_OVERLOAD": "sensory_overload",
    "MEDITATION": "meditation",
    "CHAOS": "chaos",
}


def test_members_and_values():
    assert set(GnosisMethod.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(GnosisMethod, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in GnosisMethod:
        assert isinstance(member, GnosisMethod)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("exhaustion", GnosisMethod.EXHAUSTION),
        ("ecstasy", GnosisMethod.ECSTASY),
        ("obsession", GnosisMethod.OBSESSION),
        ("sensory_overload", GnosisMethod.SENSORY_OVERLOAD),
        ("meditation", GnosisMethod.MEDITATION),
        ("chaos", GnosisMethod.CHAOS),
    ],
)
def test_lookup_by_value(value, member):
    assert GnosisMethod(value) is member


def test_lookup_by_name():
    assert GnosisMethod["EXHAUSTION"] is GnosisMethod.EXHAUSTION
    assert GnosisMethod["ECSTASY"] is GnosisMethod.ECSTASY


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        GnosisMethod("dancing")


def test_values_are_unique():
    values = [m.value for m in GnosisMethod]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(GnosisMethod.EXHAUSTION, enum.Enum)
    assert isinstance(GnosisMethod.CHAOS, enum.Enum)
