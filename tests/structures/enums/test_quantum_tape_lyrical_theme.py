import enum
import pytest

from app.structures.enums.quantum_tape_lyrical_theme import QuantumTapeLyricalTheme

EXPECTED = {
    "COUNTERFACTUAL_NOSTALGIA": "counterfactual_nostalgia",
    "ROADS_NOT_TAKEN": "roads_not_taken",
    "TAPED_OVER_MEMORIES": "taped_over_memories",
    "PARALLEL_LIVES": "parallel_lives",
    "TEMPORAL_BLEEDING": "temporal_bleeding",
    "CASSETTE_ARCHAEOLOGY": "cassette_archaeology",
    "GLIMPSES_OF_PAST": "glimpses_of_past",
    "DEJA_VU": "deja_vu",
    "BEING_THERE": "being_there",
    "BEING_WATCHED": "being_watched",
    "LOSS": "loss",
}


def test_members_and_values():
    assert set(QuantumTapeLyricalTheme.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(QuantumTapeLyricalTheme, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in QuantumTapeLyricalTheme:
        assert isinstance(member, QuantumTapeLyricalTheme)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("counterfactual_nostalgia", QuantumTapeLyricalTheme.COUNTERFACTUAL_NOSTALGIA),
        ("roads_not_taken", QuantumTapeLyricalTheme.ROADS_NOT_TAKEN),
        ("taped_over_memories", QuantumTapeLyricalTheme.TAPED_OVER_MEMORIES),
        ("parallel_lives", QuantumTapeLyricalTheme.PARALLEL_LIVES),
        ("temporal_bleeding", QuantumTapeLyricalTheme.TEMPORAL_BLEEDING),
        ("cassette_archaeology", QuantumTapeLyricalTheme.CASSETTE_ARCHAEOLOGY),
        ("glimpses_of_past", QuantumTapeLyricalTheme.GLIMPSES_OF_PAST),
        ("deja_vu", QuantumTapeLyricalTheme.DEJA_VU),
        ("being_there", QuantumTapeLyricalTheme.BEING_THERE),
        ("being_watched", QuantumTapeLyricalTheme.BEING_WATCHED),
        ("loss", QuantumTapeLyricalTheme.LOSS),
    ],
)
def test_lookup_by_value(value, member):
    assert QuantumTapeLyricalTheme(value) is member


def test_lookup_by_name():
    assert (
        QuantumTapeLyricalTheme["COUNTERFACTUAL_NOSTALGIA"]
        is QuantumTapeLyricalTheme.COUNTERFACTUAL_NOSTALGIA
    )
    assert QuantumTapeLyricalTheme["LOSS"] is QuantumTapeLyricalTheme.LOSS


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        QuantumTapeLyricalTheme("invalid_theme")


def test_values_are_unique():
    values = [m.value for m in QuantumTapeLyricalTheme]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(QuantumTapeLyricalTheme.COUNTERFACTUAL_NOSTALGIA, enum.Enum)
    assert isinstance(QuantumTapeLyricalTheme.LOSS, enum.Enum)
