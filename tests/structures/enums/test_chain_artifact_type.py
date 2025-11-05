import enum

import pytest

from app.structures.enums.chain_artifact_type import ChainArtifactType

EXPECTED = {
    "TRANSCRIPT": "transcript",
    "INSTRUCTIONS_TO_HUMAN": "instructions_to_human",
    "SIGIL_DESCRIPTION": "sigil_description",
    "DOCUMENT": "doc",
    "AUDIO_MOSIAC": "audio_mos",
    "RANDOM_AUDIO_BY_COLOR_SEGMENT": "col_audio_",
    "NOISE_MIXED_AUDIO": "noise_mixed",
}


def test_members_and_values():
    assert set(ChainArtifactType.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(ChainArtifactType, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in ChainArtifactType:
        assert isinstance(member, ChainArtifactType)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("transcript", ChainArtifactType.TRANSCRIPT),
        ("instructions_to_human", ChainArtifactType.INSTRUCTIONS_TO_HUMAN),
        ("sigil_description", ChainArtifactType.SIGIL_DESCRIPTION),
        ("doc", ChainArtifactType.DOCUMENT),
        ("audio_mos", ChainArtifactType.AUDIO_MOSIAC),
        ("col_audio_", ChainArtifactType.RANDOM_AUDIO_BY_COLOR_SEGMENT),
        ("noise_mixed", ChainArtifactType.NOISE_MIXED_AUDIO),
    ],
)
def test_lookup_by_value(value, member):
    assert ChainArtifactType(value) is member


def test_lookup_by_name():
    assert ChainArtifactType["DOCUMENT"] is ChainArtifactType.DOCUMENT
    assert (
        ChainArtifactType["RANDOM_AUDIO_BY_COLOR_SEGMENT"]
        is ChainArtifactType.RANDOM_AUDIO_BY_COLOR_SEGMENT
    )


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        ChainArtifactType("pickled")


def test_values_are_unique():
    values = [m.value for m in ChainArtifactType]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(ChainArtifactType.AUDIO_MOSIAC, enum.Enum)
    assert isinstance(ChainArtifactType.INSTRUCTIONS_TO_HUMAN, enum.Enum)
