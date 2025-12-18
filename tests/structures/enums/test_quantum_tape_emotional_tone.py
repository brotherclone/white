import enum

import pytest

from app.structures.enums.quantum_tape_emotional_tone import QuantumTapeEmotionalTone

EXPECTED = {
    "WISTFUL": "wistful",
    "MELANCHOLY": "melancholy",
    "BITTERSWEET": "bittersweet",
    "NOSTALGIC": "nostalgic",
    "PEACEFUL": "peaceful",
    "RESTLESS": "restless",
}


def test_members_and_values():
    assert set(QuantumTapeEmotionalTone.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(QuantumTapeEmotionalTone, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in QuantumTapeEmotionalTone:
        assert isinstance(member, QuantumTapeEmotionalTone)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("wistful", QuantumTapeEmotionalTone.WISTFUL),
        ("melancholy", QuantumTapeEmotionalTone.MELANCHOLY),
        ("bittersweet", QuantumTapeEmotionalTone.BITTERSWEET),
        ("nostalgic", QuantumTapeEmotionalTone.NOSTALGIC),
        ("peaceful", QuantumTapeEmotionalTone.PEACEFUL),
        ("restless", QuantumTapeEmotionalTone.RESTLESS),
    ],
)
def test_lookup_by_value(value, member):
    assert QuantumTapeEmotionalTone(value) is member


def test_lookup_by_name():
    assert QuantumTapeEmotionalTone["RESTLESS"] is QuantumTapeEmotionalTone.RESTLESS
    assert QuantumTapeEmotionalTone["NOSTALGIC"] is QuantumTapeEmotionalTone.NOSTALGIC


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        QuantumTapeEmotionalTone("Unknown")


def test_values_are_unique():
    values = [m.value for m in QuantumTapeEmotionalTone]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(QuantumTapeEmotionalTone.PEACEFUL, enum.Enum)
    assert isinstance(QuantumTapeEmotionalTone.BITTERSWEET, enum.Enum)
