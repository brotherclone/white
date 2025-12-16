import enum

import pytest

from app.structures.enums.quantum_tape_recording_quality import (
    QuantumTapeRecordingQuality,
)

EXPECTED = {"LP": "lp", "SP": "sp", "EP": "ep"}


def test_members_and_values():
    assert set(QuantumTapeRecordingQuality.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(QuantumTapeRecordingQuality, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in QuantumTapeRecordingQuality:
        assert isinstance(member, QuantumTapeRecordingQuality)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("lp", QuantumTapeRecordingQuality.LP),
        ("sp", QuantumTapeRecordingQuality.SP),
        ("ep", QuantumTapeRecordingQuality.EP),
    ],
)
def test_lookup_by_value(value, member):
    assert QuantumTapeRecordingQuality(value) is member


def test_lookup_by_name():
    assert QuantumTapeRecordingQuality["LP"] is QuantumTapeRecordingQuality.LP
    assert QuantumTapeRecordingQuality["SP"] is QuantumTapeRecordingQuality.SP


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        QuantumTapeRecordingQuality("Unknown")


def test_values_are_unique():
    values = [m.value for m in QuantumTapeRecordingQuality]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(QuantumTapeRecordingQuality.LP, enum.Enum)
    assert isinstance(QuantumTapeRecordingQuality.EP, enum.Enum)
