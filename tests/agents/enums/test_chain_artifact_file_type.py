import enum
import pytest
from app.agents.enums.chain_artifact_file_type import ChainArtifactFileType

def test_members_and_values():
    expected = {
        "MARKDOWN": "md",
        "AUDIO": "wav",
        "JSON": "json",
        "PNG": "png",
    }
    for name, value in expected.items():
        member = ChainArtifactFileType[name]
        assert member.name == name
        assert member.value == value
        assert isinstance(member.value, str)

@pytest.mark.parametrize("value,member", [
    ("md", ChainArtifactFileType.MARKDOWN),
    ("wav", ChainArtifactFileType.AUDIO),
    ("json", ChainArtifactFileType.JSON),
    ("png", ChainArtifactFileType.PNG),
])
def test_lookup_by_value(value, member):
    assert ChainArtifactFileType(value) is member

def test_lookup_by_name():
    assert ChainArtifactFileType["MARKDOWN"] is ChainArtifactFileType.MARKDOWN

def test_invalid_value_raises_value_error():
    with pytest.raises(ValueError):
        ChainArtifactFileType("exe")

def test_values_are_unique():
    values = [m.value for m in ChainArtifactFileType]
    assert len(values) == len(set(values))

def test_enum_members_are_enum_instances():
    assert isinstance(ChainArtifactFileType.MARKDOWN, enum.Enum)
    assert isinstance(ChainArtifactFileType.AUDIO, enum.Enum)