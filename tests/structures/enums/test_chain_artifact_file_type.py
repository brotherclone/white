import enum

import pytest

from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType

EXPECTED = {
    "AUDIO": "wav",
    "PNG": "png",
    "MARKDOWN": "md",
    "JSON": "json",
}


def test_members_and_values():
    assert set(ChainArtifactFileType.__members__.keys()) == set(EXPECTED.keys())
    for name, expected_value in EXPECTED.items():
        member = getattr(ChainArtifactFileType, name)
        assert member.value == expected_value
        assert isinstance(member.value, str)


def test_members_are_str_and_enum_and_compare_to_value():
    for member in ChainArtifactFileType:
        assert isinstance(member, ChainArtifactFileType)
        assert isinstance(member, str)
        assert isinstance(member, enum.Enum)
        assert member == member.value


@pytest.mark.parametrize(
    "value,member",
    [
        ("wav", ChainArtifactFileType.AUDIO),
        ("png", ChainArtifactFileType.PNG),
        ("md", ChainArtifactFileType.MARKDOWN),
        ("json", ChainArtifactFileType.JSON),
    ],
)
def test_lookup_by_value(value, member):
    assert ChainArtifactFileType(value) is member


def test_lookup_by_name():
    assert ChainArtifactFileType["MARKDOWN"] is ChainArtifactFileType.MARKDOWN
    assert ChainArtifactFileType["PNG"] is ChainArtifactFileType.PNG


def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        ChainArtifactFileType("EXCEL")


def test_values_are_unique():
    values = [m.value for m in ChainArtifactFileType]
    assert len(values) == len(set(values))


def test_enum_members_are_enum_instances():
    assert isinstance(ChainArtifactFileType.AUDIO, enum.Enum)
    assert isinstance(ChainArtifactFileType.JSON, enum.Enum)
